"""构建 Windows 轻量程序包：只分发 2.5 程序和环境安装器，排除全部模型。"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import APP_VERSION, MODEL_VERSION

VERSION = APP_VERSION
OUTPUT = Path(f"build/release_v{VERSION}_light")
PREFIX = "IndexTTS-For-Windows"
WEIGHT_SUFFIXES = {".pth", ".pt", ".t7", ".safetensors", ".ckpt", ".onnx", ".gguf", ".bin"}
SKIP_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints", ".pytest_cache", ".cache"}


def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def collect_files():
    """白名单限制分发范围，环境、模型、缓存、日志与个人音频均不能进入 ZIP。"""
    files = [Path(name) for name in ("run.bat", "start.py", "requirements.txt")]
    for folder in ("app", "vendor", "tools/bootstrap"):
        files += [path for path in Path(folder).rglob("*") if path.is_file()
                  and not any(part in SKIP_PARTS for part in path.parts)
                  and path.suffix.lower() not in WEIGHT_SUFFIXES | {".pyc", ".pyo", ".log", ".tmp"}]
    files += [Path("tools/setup_environment.ps1"), Path("tools/download_models.py"),
              Path("tools/download_minnan_translation.py"), Path("tools/build_minnan_words.py"),
              Path("tools/build_distribution.py")]
    files += [Path("docs") / name for name in (
        "README.md", f"RELEASE_v{VERSION}.md", "MODEL_SETUP.md", "model_manifest.json",
        "upstream_v25.json", "desktop_result.png", "desktop_emotion.png", "desktop_window.png", "gitignore",
        "MINNAN_LANGUAGE_PACK.md", "MINNAN_WORDS_SOURCES.md", "DISTRIBUTION.md",
        "minnan_layout_normal.png", "minnan_layout_compact.png")]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("打包缺少必需文件：" + ", ".join(missing))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="构建不含模型的 IndexTTS 2.5 轻量包")
    parser.add_argument("--lightweight", action="store_true", help="轻量分发（当前唯一分发方式）")
    parser.add_argument("--replace", action="store_true", help="构建候选包，CRC 验证通过后原子替换本地成品")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / f"{PREFIX}-v{VERSION}-win64-no-models.zip"
    if archive.exists() and not args.replace:
        raise RuntimeError("发行包已存在，请先确认已有构建；不会直接覆盖已验证成品")
    files = collect_files()
    hashes = {path.as_posix(): sha256(path) for path in files}
    candidate = archive.with_suffix(".candidate.zip")
    with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for path in files:
            package.write(path, f"{PREFIX}/{path.as_posix()}")
        package.writestr(f"{PREFIX}/models/IndexTTS-{MODEL_VERSION}/", b"")
    # 新成品完整确认前保留旧包；仅在候选包 CRC 全量通过后提交替换。
    with zipfile.ZipFile(candidate) as package:
        if package.testzip() is not None:
            raise RuntimeError("候选程序包 CRC 失败，已有成品保持原样")
    candidate.replace(archive)
    artifact = {"name": archive.name, "bytes": archive.stat().st_size, "sha256": sha256(archive)}
    report = {"version": VERSION, "model_version": MODEL_VERSION,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
              "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
              "file_count": len(files), "contains_models": False, "contains_python_runtime": False,
              "first_launch_installs_even": True, "bootstrap_sha256": sha256(Path("tools/bootstrap/uv.exe")),
              "payload_sha256": hashes, "archives": [artifact]}
    (OUTPUT / "release_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "SHA256SUMS.txt").write_text(f"{artifact['sha256']}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "payload_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
