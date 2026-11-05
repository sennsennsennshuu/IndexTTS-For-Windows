"""在真实项目环境生成完整包、台语包和轻量包，校验通过后再提交成品。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import APP_VERSION, MODEL_DIR, MINNAN_DIR
from tools.build_release import collect_files, PREFIX, sha256


REPORT_DIR = Path("build/distribution")
TARGETS = {"full": Path("../indextts.zip"), "language": Path("../model.zip"),
           "release": Path("../release.zip")}
SKIP_PARTS = {"__pycache__", ".git", ".cache", ".pytest_cache"}


def walk_files(root):
    """白名单目录内枚举实际文件，拒绝外链，不混入缓存和 Python 字节码。"""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"分发必需目录不存在：{root}")
    for path in root.rglob("*"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"分发内容存在符号链接，不能直接制作可移植副本：{path}")
        if path.is_file() and path.suffix.lower() not in {".pyc", ".pyo", ".tmp", ".lock", ".part"}:
            if path.name != "package_manifest.json":
                yield path


def language_files():
    """正式语言包保持原交付目录完整；试训归档与个人音色不属于分发内容。"""
    files = [MINNAN_DIR / "language.json"]
    for child in ("model", "words", "translation"):
        files.extend(walk_files(MINNAN_DIR / child))
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("台语语言包缺少说明或必需文件")
    return sorted(set(files))


def plan(kind):
    """独立定义三种范围，禁止将整个项目递归压缩而带入凭据或旧版模型。"""
    if kind == "language":
        files = language_files()
        prefix = ""
        manifest_name = (MINNAN_DIR / "package_manifest.json").as_posix()
    else:
        files = collect_files()
        if kind == "full":
            files += list(walk_files("even")) + list(walk_files(MODEL_DIR)) + language_files()
        files = sorted(set(files))
        prefix = PREFIX + "/"
        manifest_name = prefix + "docs/package_manifest.json"
    # 此处检查完整组件，避免“模型已打包”实际只包含目录或下载指针。
    if kind in {"full", "language"}:
        required = [MINNAN_DIR / "model/adapter/adapter_model.safetensors",
                    MINNAN_DIR / "model/extra_state.pt", MINNAN_DIR / "words/entries.jsonl",
                    MINNAN_DIR / "translation/model.safetensors"]
        if kind == "full":
            required += [Path("even/python.exe"), MODEL_DIR / "gpt.pth", MODEL_DIR / "codec.pth"]
        if not set(required).issubset(files) or any(path.stat().st_size < 1024 for path in required):
            raise ValueError("完整分发必需组件缺失或过小")
        if not any(path.suffix == ".gguf" for path in files):
            raise ValueError("缺少完整档台语翻译权重")
    for path in files:
        if (path.is_absolute() or ".." in path.parts or path.name == "ssh.txt"
                or path.parts[0] in {"cache", "logs", "outputs", "training", ".git"}
                or path.parts[:2] in {("models", "voice_library"), ("models", "IndexTTS-2")}):
            raise ValueError(f"分发范围包含禁止内容：{path}")
    return files, prefix, manifest_name


def build(kind):
    files, prefix, manifest_name = plan(kind)
    destination = TARGETS[kind]
    candidate = destination.with_suffix(".candidate.zip")
    if candidate.exists():
        raise FileExistsError(f"存在上次候选文件，请先审查：{candidate}")
    total = sum(path.stat().st_size for path in files)
    print(f"开始 {kind}: {len(files)} 文件，原始 {total / 1024**3:.2f} GiB", flush=True)
    entries = {}
    started = time.monotonic()
    last_report = started
    processed = 0
    manifest = {"format": 1, "kind": kind, "app_version": APP_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "files": entries}
    with zipfile.ZipFile(candidate, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as package:
        for index, path in enumerate(files, 1):
            before = path.stat()
            name = prefix + path.as_posix()
            info = zipfile.ZipInfo.from_file(path, name)
            # 大权重及大型动态库用存储模式，避免无效压缩耗时；源码、词库等照常压缩。
            binary = path.suffix.lower() in {".pth", ".pt", ".safetensors", ".gguf", ".dll", ".pyd"}
            info.compress_type = zipfile.ZIP_STORED if binary and before.st_size >= 32 * 1024**2 else zipfile.ZIP_DEFLATED
            # 当前分发运行时固定 Python 3.11，其 ZipInfo 使用此字段设置流式压缩级别。
            info._compresslevel = 1
            checksum = hashlib.sha256()
            with path.open("rb") as source, package.open(info, "w", force_zip64=True) as target:
                for block in iter(lambda: source.read(8 * 1024**2), b""):
                    target.write(block)
                    checksum.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"文件在打包过程中变化，停止提交成品：{path}")
            entries[name] = {"bytes": before.st_size, "sha256": checksum.hexdigest()}
            processed += before.st_size
            if time.monotonic() - last_report >= 20:
                print(f"写入 {kind}: {index}/{len(files)}，{processed / 1024**3:.2f}/{total / 1024**3:.2f} GiB", flush=True)
                last_report = time.monotonic()
        package.writestr(manifest_name, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"开始逐文件 CRC／SHA-256 校验：{kind}", flush=True)
    verified = 0
    with zipfile.ZipFile(candidate) as package:
        names = package.namelist()
        if len(names) != len(set(names)) or set(names) != set(entries) | {manifest_name}:
            raise RuntimeError("ZIP 清单存在重复或遗漏")
        if json.loads(package.read(manifest_name)) != manifest:
            raise RuntimeError("包内清单与实际构建不一致")
        for index, (name, expected) in enumerate(entries.items(), 1):
            checksum = hashlib.sha256()
            length = 0
            # ZipExtFile 读至结尾会同时校验 CRC；逐项摘要还防止错包和漏文件。
            with package.open(name) as source:
                for block in iter(lambda: source.read(8 * 1024**2), b""):
                    checksum.update(block)
                    length += len(block)
            if length != expected["bytes"] or checksum.hexdigest() != expected["sha256"]:
                raise RuntimeError(f"包内文件摘要错误：{name}")
            verified += length
            if time.monotonic() - last_report >= 20:
                print(f"校验 {kind}: {index}/{len(entries)}，{verified / 1024**3:.2f}/{total / 1024**3:.2f} GiB", flush=True)
                last_report = time.monotonic()
    digest = sha256(candidate)
    candidate.replace(destination)
    report = {key: value for key, value in manifest.items() if key != "files"}
    report.update(path=destination.as_posix(), bytes=destination.stat().st_size, sha256=digest,
                  payload_files=len(entries), payload_bytes=total, manifest_entry=manifest_name,
                  crc_passed=True, all_payload_hashes_verified=True,
                  seconds=round(time.monotonic() - started, 2))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{kind}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("all", *TARGETS), default="all")
    parser.add_argument("--plan", action="store_true", help="仅核对范围，不写 ZIP")
    args = parser.parse_args()
    kinds = ("release", "language", "full") if args.kind == "all" else (args.kind,)
    for kind in kinds:
        if args.plan:
            files, _, _ = plan(kind)
            print(json.dumps({"kind": kind, "files": len(files), "bytes": sum(p.stat().st_size for p in files),
                              "target": TARGETS[kind].as_posix()}, ensure_ascii=False))
        else:
            build(kind)


if __name__ == "__main__":
    main()
