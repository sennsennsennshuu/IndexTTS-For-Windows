"""校验当前 2.5 轻量包：无模型、无旧入口、全量文件摘要、CRC 及解压窗口。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.build_release import PREFIX, VERSION, OUTPUT, WEIGHT_SUFFIXES, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lightweight", action="store_true", help="当前分发方式为轻量包")
    parser.add_argument("--extract-smoke", action="store_true", help="在项目 build/ 中解压并检查真实窗口")
    parser.add_argument("--validation-dir", type=Path, help="指定项目 build/ 内新的验收目录")
    args = parser.parse_args()
    report = json.loads((OUTPUT / "release_manifest.json").read_text(encoding="utf-8"))
    assert report["version"] == VERSION and report["model_version"] == "2.5"
    target = args.validation_dir or Path(f"build/validation_v{VERSION}")
    assert target.resolve().is_relative_to(Path("build").resolve()), "验收目录必须位于项目 build/ 内"
    if args.extract_smoke and target.exists():
        raise RuntimeError("解压目录已存在，请先确认上次验收结果再清理")
    names = []
    for item in report["archives"]:
        archive = OUTPUT / item["name"]
        assert sha256(archive) == item["sha256"] and archive.stat().st_size == item["bytes"]
        with zipfile.ZipFile(archive) as package:
            names += package.namelist()
            for name in package.namelist():
                parts = Path(name).parts
                assert parts[0] == PREFIX and ".." not in parts and not Path(name).is_absolute()
                if name.endswith("/"):
                    continue
                assert parts[1] not in {"models", "even", "cache", "logs", "outputs", "test", ".git"}
                assert Path(name).suffix.lower() not in WEIGHT_SUFFIXES | {".pyc", ".wav", ".mp3", ".lib", ".pdb"}
                relative = Path(*parts[1:]).as_posix()
                assert hashlib.sha256(package.read(name)).hexdigest() == report["payload_sha256"][relative]
            assert package.testzip() is None
            assert b"\r\n" in package.read(f"{PREFIX}/run.bat")
            if args.extract_smoke:
                package.extractall(target)
    assert {Path(name).name for name in names if len(Path(name).parts) == 2 and not name.endswith("/")} == {"run.bat", "start.py", "requirements.txt"}
    assert f"{PREFIX}/vendor/indextts/infer_v2_5.py" in names
    assert f"{PREFIX}/vendor/indextts/infer_v2.py" not in names
    assert f"{PREFIX}/tools/bootstrap/uv.exe" in names
    assert f"{PREFIX}/tools/download_models.py" in names
    report.update(crc_passed=True, models_excluded_verified=True, all_payload_hashes_verified=True,
                  old_engine_excluded=True, root_files=["requirements.txt", "run.bat", "start.py"])
    if args.extract_smoke:
        code = '''
import os, sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
from app.paths import prepare_runtime, validate_models
prepare_runtime()
from PyQt6.QtWidgets import QApplication
from app.window import IndexTTSWindow
app=QApplication([])
window=IndexTTSWindow()
window.show(); app.processEvents()
assert not window.grab().isNull()
assert window.language_combo.count()==6 and window.duration_spin.value()==1.0
assert window.language_combo.findData('ZH-NAN-TW')>=0
assert hasattr(window,'voice_library_panel')
try:
    validate_models()
except FileNotFoundError:
    pass
else:
    raise AssertionError('Missing models were not reported')
window.close()
print('Extracted 2.5 desktop, six languages, voice library, no external models: OK')
'''
        result = subprocess.run([sys.executable, "-X", "utf8", "-I", "-B", "-c", code, str(target / PREFIX)],
                                env=dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1"),
                                capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        report["extracted_ui_smoke"] = result.stdout.strip()
        report["smoke_runtime"] = "existing project even; fresh dependency installation tested separately"
    (OUTPUT / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "payload_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
