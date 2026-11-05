"""三类真实分发包的范围与提取启动验收，不重复构建阶段的逐文件 CRC／摘要读取。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_distribution import TARGETS, REPORT_DIR
from tools.build_release import PREFIX


def verify(kind, smoke=False):
    archive = TARGETS[kind]
    report = json.loads((REPORT_DIR / f"{kind}.json").read_text(encoding="utf-8"))
    assert report["crc_passed"] and report["all_payload_hashes_verified"]
    assert archive.stat().st_size == report["bytes"]
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        assert len(names) == len(set(names))
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
        assert all(Path(name).name != "ssh.txt" for name in names)
        if kind == "language":
            root = "models/extra_language/zh-nan-tw/"
            assert all(name.startswith(root) for name in names)
            for required in ("model/adapter/adapter_model.safetensors", "model/extra_state.pt",
                             "model/delivery_manifest.json", "words/entries.jsonl", "language.json"):
                assert root + required in names, required
            assert any(name.endswith(".gguf") for name in names)
            assert not any(name.startswith(root + "validation/") for name in names)
        else:
            root_files = {Path(name).name for name in names if len(Path(name).parts) == 2}
            assert root_files == {"run.bat", "start.py", "requirements.txt"}, root_files
            assert all(name.startswith(PREFIX + "/") for name in names)
            window_source = package.read(PREFIX + "/app/window.py").decode("utf-8")
            assert "zeroshot 语音合成" in window_source and 'label("VOICE STUDIO"' not in window_source
            assert "minnan_disclaimer" not in window_source
            assert PREFIX + "/app/voice_library.py" in names
            assert PREFIX + "/vendor/indextts/infer_v2_5.py" in names
            assert PREFIX + "/vendor/indextts/infer_v2.py" not in names
            assert b"\r\n" in package.read(PREFIX + "/run.bat")
            relative = [Path(name).parts[1:] for name in names]
            if kind == "release":
                assert not any(parts[0] in {"models", "even", "cache", "logs", "outputs"} for parts in relative)
            else:
                assert PREFIX + "/even/python.exe" in names
                assert PREFIX + "/models/IndexTTS-2.5/gpt.pth" in names
                assert PREFIX + "/models/extra_language/zh-nan-tw/model/extra_state.pt" in names
                assert not any(parts[:2] in {("models", "IndexTTS-2"), ("models", "voice_library")} for parts in relative)
        result = {"kind": kind, "path": archive.as_posix(), "structure_passed": True,
                  "files": len(names), "sha256": report["sha256"]}
        if smoke and kind != "language":
            target = Path("build/distribution_validation") / kind
            if target.exists():
                raise FileExistsError(f"保留已有提取验收目录，请先审查：{target}")
            target.mkdir(parents=True)
            for name in names:
                parts = Path(name).parts
                extract = True
                if kind == "full":
                    # 启动抽检只提取程序、Python 标准库与 Qt，避免再复制约 20 GiB 模型和环境。
                    # 包内全部其他文件已由构建程序逐文件读取并校验 SHA-256。
                    extract = parts[1] not in {"models"}
                    if parts[1] == "even" and "site-packages" in parts:
                        index = parts.index("site-packages") + 1
                        module = parts[index]
                        extract = module.startswith(("PyQt6", "_distutils_hack")) or len(parts) == index + 1 and module.endswith(".pth")
                if extract:
                    package.extract(name, target)
            extracted = target / PREFIX
            runtime = extracted / "even/python.exe" if kind == "full" else Path(sys.executable)
            code = r'''
import json, sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
from app.paths import prepare_runtime
prepare_runtime()
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QLabel
from app.window import IndexTTSWindow
app=QApplication([])
window=IndexTTSWindow()
window.show()
app.processEvents()
assert window.language_combo.count()==6
assert hasattr(window,'voice_library_panel')
assert any(label.text()=='zeroshot 语音合成' for label in window.findChildren(QLabel))
assert not window.grab().isNull()
window.close()
print(json.dumps({'python':sys.executable,'prefix':sys.prefix,'window_opened':True,'language_count':6}))
'''
            checked = subprocess.run([str(runtime), "-X", "utf8", "-I", "-B", "-c", code, str(extracted)],
                                     env=dict(os.environ, PYTHONUTF8="1", QT_QPA_PLATFORM="windows"),
                                     capture_output=True, text=True, encoding="utf-8", timeout=60)
            if checked.returncode:
                raise RuntimeError(checked.stdout + checked.stderr)
            result["extracted_ui_smoke"] = json.loads(checked.stdout.strip())
            result["runtime_from_archive"] = kind == "full"
            result["extraction_scope"] = "program and bundled Python/Qt, no model inference" if kind == "full" else "complete lightweight package; existing project runtime"
    (REPORT_DIR / f"{kind}_acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("all", *TARGETS), default="all")
    parser.add_argument("--extract-smoke", action="store_true")
    args = parser.parse_args()
    for kind in TARGETS if args.kind == "all" else (args.kind,):
        verify(kind, args.extract_smoke)


if __name__ == "__main__":
    main()
