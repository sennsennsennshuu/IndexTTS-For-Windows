"""构建可解压运行的 Windows 发行包，严格排除模型和本机产生的数据。"""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
import copy
import argparse


VERSION = "1.2.0"
OUTPUT = Path(f"build/release_v{VERSION}")
PREFIX = "IndexTTS-For-Windows"
WEIGHT_SUFFIXES = {".pth", ".pt", ".t7", ".safetensors", ".ckpt", ".onnx", ".gguf"}
SKIP_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints", ".pytest_cache", ".cache"}


def should_include(path):
    """只允许明确列出的目录；包内不得带入用户路径、缓存、语音权重或开发静态库。"""
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    if path.suffix.lower() in {".pyc", ".pyo", ".lib", ".pdb", ".log"}:
        return False
    if path.name == "direct_url.json":
        return False
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        # site-packages 内的 .pth 是 Python 的路径配置，不是神经网络权重。
        return path.suffix.lower() == ".pth" and path.parts[0] == "even" and "site-packages" in path.parts
    if path.parts[0] == "even":
        # torch.testing、numpy.testing 是运行时公开模块，不能按目录名 testing 删除。
        if any(part in {"tests", "test", "include", "libs", "cmake"} for part in path.parts):
            return False
        if path.parts[:2] == ("even", "Scripts"):
            # 源机器生成的 exe 带绝对解释器路径；发行版统一使用 python -m。
            return False
        if path.parts[:3] in {("even", "Library", "lib"), ("even", "Library", "include")}:
            return False
    return True


def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def collect_files():
    files = [Path(name) for name in ("run.bat", "start.py", "requirements.txt")]
    for folder in ("app", "vendor", "even"):
        files.extend(path for path in Path(folder).rglob("*") if path.is_file() and should_include(path))
    files.extend(Path("docs") / name for name in (
        "README.md", "RELEASE_v1.2.0.md", "MODEL_SETUP.md", "model_manifest.json",
        "desktop_result.png", "desktop_emotion.png", "desktop_window.png", "gitignore"))
    return sorted(files)


def split_archives(archive):
    """将已压缩文件分配为独立 ZIP，避免反复压缩数 GB 的 CUDA DLL。

    使用当前随项目交付的 CPython zipfile 元数据写入目录，原样复制每个本地文件记录；
    每个输出都有自己的中央目录，可由 Windows 直接解压，并经过全量 CRC 验收。
    """
    with zipfile.ZipFile(archive) as source:
        entries = sorted(source.infolist(), key=lambda info: info.header_offset)
        spans = {entry.filename: ((entries[index + 1].header_offset if index + 1 < len(entries) else source.start_dir)
                                  - entry.header_offset) for index, entry in enumerate(entries)}
        program = [entry for entry in entries if not entry.filename.startswith(f"{PREFIX}/even/")]
        runtime = [entry for entry in entries if entry.filename.startswith(f"{PREFIX}/even/")]
        groups = [(OUTPUT / f"{PREFIX}-v{VERSION}-program-no-models.zip", program)]
        batch, batch_size = [], 0
        runtime_index = 1
        for entry in runtime:
            size = spans[entry.filename]
            if batch and batch_size + size > 1800 * 1024**2:
                groups.append((OUTPUT / f"{PREFIX}-v{VERSION}-runtime-{runtime_index:02d}.zip", batch))
                runtime_index += 1
                batch, batch_size = [], 0
            batch.append(entry)
            batch_size += size
        if batch:
            groups.append((OUTPUT / f"{PREFIX}-v{VERSION}-runtime-{runtime_index:02d}.zip", batch))
        for target, group in groups:
            if target.exists():
                raise RuntimeError(f"输出文件已存在：{target}")
            print(f"写入独立 ZIP：{target.name}，{len(group)} 项", flush=True)
            with zipfile.ZipFile(target, "w", allowZip64=True) as output, archive.open("rb") as original:
                for entry in group:
                    # 发行说明可在压缩环境期间补充，重新写入此小文件即可保持包内说明一致。
                    if entry.filename == f"{PREFIX}/docs/RELEASE_v1.2.0.md":
                        output.write(Path("docs/RELEASE_v1.2.0.md"), entry.filename, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
                        continue
                    metadata = copy.copy(entry)
                    original.seek(entry.header_offset)
                    metadata.header_offset = output.fp.tell()
                    output._writecheck(metadata)
                    output._didModify = True
                    remaining = spans[entry.filename]
                    while remaining:
                        data = original.read(min(8 * 1024**2, remaining))
                        if not data:
                            raise RuntimeError("源压缩记录不完整")
                        output.fp.write(data)
                        remaining -= len(data)
                    output.filelist.append(metadata)
                    output.NameToInfo[metadata.filename] = metadata
                    output.start_dir = output.fp.tell()
        return [target for target, group in groups]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-compressed", action="store_true", help="复用已完整生成的本地中间 ZIP，仅生成独立附件和清单")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / f"{PREFIX}-v{VERSION}-win64-no-models.zip"
    if archive.exists() and not args.reuse_compressed:
        raise SystemExit("当前发行包已存在；请先确认是否需要保留，构建工具不会自动覆盖已验证成品。")
    files = collect_files()
    uncompressed = sum(path.stat().st_size for path in files)
    print(f"开始打包：{len(files)} 个文件，{uncompressed / 1024**3:.2f} GiB，排除全部模型。", flush=True)
    if not args.reuse_compressed:
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as output:
            for index, path in enumerate(files):
                output.write(path, f"{PREFIX}/{path.as_posix()}")
                if index % 2000 == 0 or path.stat().st_size > 200 * 1024**2:
                    print(f"[{index + 1}/{len(files)}] {path.as_posix()}  当前压缩包 {output.fp.tell()/1024**2:.1f} MiB", flush=True)
            # 空模型目录只提供放置位置，压缩包中没有任何权重文件。
            output.writestr(f"{PREFIX}/models/IndexTTS-2/", b"")
    report = {
        "version": VERSION, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "file_count": len(files), "uncompressed_bytes": uncompressed,
        "contains_models": False, "contains_python_runtime": True,
        "excluded": ["models/**", "*.pt", "*.t7", "*.safetensors", "*.ckpt", "*.onnx", "*.gguf",
                     "even/**/*.lib", "even/**/include/**", "even/**/tests/**", "logs/**", "cache/**", "outputs/**", "test/**"],
        "intermediate_archive": archive.name,
    }
    assets = split_archives(archive)
    report["archives"] = [{"name": item.name, "bytes": item.stat().st_size, "sha256": sha256(item)} for item in assets]
    if any(item.stat().st_size >= 2 * 1024**3 for item in assets):
        raise SystemExit("存在超过 2 GiB 的附件，已停止发布")
    (OUTPUT / "release_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "SHA256SUMS.txt").write_text("".join(f"{item['sha256']}  {item['name']}\n" for item in report["archives"]), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
