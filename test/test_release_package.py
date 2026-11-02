"""发行包验收：检查模型排除、路径安全、CRC、SHA-256 和解压后的独立环境。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.build_release import PREFIX, VERSION, OUTPUT, WEIGHT_SUFFIXES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-smoke", action="store_true", help="完整解压到项目 build/，运行依赖与窗口冒烟检查")
    parser.add_argument("--lightweight", action="store_true", help="校验轻量包及首次启动环境安装器")
    args = parser.parse_args()
    output = Path("build/release_v1.2.0_light") if args.lightweight else OUTPUT
    report = json.loads((output / "release_manifest.json").read_text(encoding="utf-8"))
    names = []
    target = Path("build/light_release_validation") if args.lightweight else Path("build/release_validation")
    if args.extract_smoke and (target / PREFIX).exists():
        raise RuntimeError("解压验收目录已存在，请先明确清理后重试")
    for artifact in report["archives"]:
        archive = output / artifact["name"]
        checksum = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        assert checksum.hexdigest() == artifact["sha256"], "压缩包 SHA-256 不一致"
        assert archive.stat().st_size < 2 * 1024**3, "单附件超过 2 GiB"
        with zipfile.ZipFile(archive) as package:
            current_names = package.namelist()
            assert not set(names).intersection(current_names), "多个 ZIP 中存在重名文件"
            names.extend(current_names)
            for name in current_names:
                parts = Path(name).parts
                assert parts[0] == PREFIX and ".." not in parts, f"非法压缩路径：{name}"
                if name.endswith("/"):
                    continue
                suffix = Path(name).suffix.lower()
                assert parts[1] not in {"models", "cache", "logs", "outputs", "test", ".git"}, f"误包含本地数据：{name}"
                assert suffix not in {".lib", ".pdb", ".pyc", ".wav", ".mp3"}, f"不应包含此文件：{name}"
                if suffix in WEIGHT_SUFFIXES:
                    assert suffix == ".pth" and parts[1] == "even" and "site-packages" in parts, f"误包含模型：{name}"
            if f"{PREFIX}/run.bat" in current_names:
                assert b"\r\n" in package.read(f"{PREFIX}/run.bat"), "批处理文件缺少 CRLF 换行"
            print(f"{archive.name}：结构与 SHA-256 通过，开始 CRC 检查与解压。", flush=True)
            assert package.testzip() is None, "压缩包 CRC 校验失败"
            if args.extract_smoke:
                package.extractall(target)
    root_files = {Path(name).name for name in names if len(Path(name).parts) == 2 and not name.endswith("/")}
    assert root_files == {"run.bat", "start.py", "requirements.txt"}, root_files
    if args.lightweight:
        assert not any(name.startswith(f"{PREFIX}/even/") for name in names)
        assert f"{PREFIX}/tools/bootstrap/uv.exe" in names
        assert f"{PREFIX}/tools/setup_environment.ps1" in names
        assert f"{PREFIX}/tools/bootstrap/LICENSE-MIT" in names
    else:
        assert f"{PREFIX}/even/python.exe" in names
        assert f"{PREFIX}/even/Lib/site-packages/torch/lib/torch_cuda.dll" in names
        assert f"{PREFIX}/even/Lib/site-packages/torch/testing/__init__.py" in names
    assert f"{PREFIX}/docs/MODEL_SETUP.md" in names
    report["crc_passed"] = True
    report["models_excluded_verified"] = True
    report["root_files"] = sorted(root_files)
    if args.extract_smoke and args.lightweight:
        root = target / PREFIX
        assert (root / "tools/bootstrap/uv.exe").is_file()
        assert "setup_environment.ps1" in (root / "run.bat").read_text(encoding="utf-8")
        assert ".indextts_setup_pending" in (root / "run.bat").read_text(encoding="utf-8")
        runtime_probe = Path("build/light_setup_validation/even/python.exe")
        result = subprocess.run([str(runtime_probe), "-X", "utf8", "-I", "-B", "-c", "import sys; print(sys.version.split()[0])"],
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 0 and result.stdout.startswith("3.11."), result.stderr
        resolver_log = Path("logs/light_setup_validation.log").read_text(encoding="utf-8")
        assert "完整依赖解析检查通过" in resolver_log
        regression_log = Path("logs/release_regression.log").read_text(encoding="utf-8")
        assert "Ran 16 tests" in regression_log and "\nOK\n" in regression_log
        report["bootstrap_python_installed"] = True
        report["bootstrap_python_version"] = result.stdout.strip()
        report["dependencies_resolved"] = True
        report["regression_tests_passed"] = 16
    elif args.extract_smoke:
        root = Path("build/release_validation") / PREFIX
        python = root / "even/python.exe"
        environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1")
        code = r'''
from pathlib import Path
import sys, json
root=Path(sys.executable).resolve().parent.parent
sys.path.insert(0,str(root))
from app.paths import prepare_runtime
prepare_runtime()
import torch, torchaudio, transformers, wetext, soundfile, librosa
from indextts.infer_v2 import IndexTTS2
from PyQt6.QtWidgets import QApplication
from app.window import IndexTTSWindow
assert Path(sys.prefix).resolve()==root/'even'
assert Path(torch.__file__).resolve().is_relative_to(root/'even')
app=QApplication([])
window=IndexTTSWindow()
window.show()
app.processEvents()
for mode in range(4):
    window.emo_method_combo.setCurrentIndex(mode)
    app.processEvents()
    assert window.grab().isNull() is False
window.close()
print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,
                  'cuda_available':torch.cuda.is_available(),'four_ui_modes':True,
                  'relocated_python':str(sys.executable)},ensure_ascii=False))
'''
        result = subprocess.run([str(python), "-X", "utf8", "-I", "-B", "-c", code],
                                env=environment, text=True, encoding="utf-8", capture_output=True, timeout=100)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        report["relocation_smoke"] = json.loads(result.stdout.strip().splitlines()[-1])
        # 没有模型的发行包必须明确报缺失，不能偷偷使用源项目或全局模型。
        check = subprocess.run([str(python), "-X", "utf8", "-I", "-B", str(root / "start.py"), "--check"],
                               env=environment, text=True, encoding="utf-8", capture_output=True, timeout=20)
        assert check.returncode == 1 and "缺少项目内模型文件" in check.stderr, check.stdout + check.stderr
        report["missing_models_reported"] = True
        pip_check = subprocess.run([str(python), "-I", "-B", "-m", "pip", "check"],
                                   env=environment, text=True, capture_output=True, timeout=30)
        assert pip_check.returncode == 0, pip_check.stdout + pip_check.stderr
        report["pip_check"] = pip_check.stdout.strip()
    (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
