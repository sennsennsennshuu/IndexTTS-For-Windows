"""模型完整性检查；可选与指定源目录比较，后续检查只需项目内校验清单。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import prepare_runtime, MODEL_DIR
prepare_runtime()


def digest(path):
    """分块校验大权重，避免将整个模型读入内存。"""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-source", type=Path, help="可选的原始模型目录，相对于项目根目录")
    parser.add_argument("--create-manifest", action="store_true", help="首次迁移时创建项目内清单")
    args = parser.parse_args()
    manifest_path = Path("docs/model_manifest.json")
    if args.create_manifest and not args.compare_source:
        parser.error("创建清单必须同时指定 --compare-source")
    if args.create_manifest:
        files = {}
        for path in sorted(MODEL_DIR.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(MODEL_DIR).as_posix()
            checksum = digest(path)
            original = args.compare_source / relative
            if not original.is_file() or digest(original) != checksum:
                raise RuntimeError(f"模型与源文件不一致：{relative}")
            files[relative] = {"bytes": path.stat().st_size, "sha256": checksum}
        report = {"algorithm": "sha256", "source_comparison_passed": True,
                  "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files.values()),
                  "files": files}
        manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in report["files"].items():
            path = MODEL_DIR / relative
            if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                raise RuntimeError(f"模型完整性检查失败：{relative}")
    print(f"模型完整性检查通过：{report['file_count']} 个文件，{report['total_bytes']} 字节")


if __name__ == "__main__":
    main()
