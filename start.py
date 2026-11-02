"""IndexTTS 独立项目入口。默认启动桌面窗口，--check 检查环境与模型。"""
import argparse
import json
import logging
from pathlib import Path
import sys


def main():
    """固定工作目录后再导入第三方库，防止缓存落入项目外。"""
    from app.paths import prepare_runtime, validate_models
    prepare_runtime()
    logging.basicConfig(filename="logs/indextts.log", encoding="utf-8",
                        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="IndexTTS2 独立桌面语音合成")
    parser.add_argument("--check", action="store_true", help="检查解释器、核心依赖及模型，不启动窗口")
    args = parser.parse_args()
    expected = Path("even/python.exe").resolve()
    if Path(sys.executable).resolve() != expected:
        print("请使用项目内环境启动：.\\even\\python.exe -I -B start.py", file=sys.stderr)
        return 1
    try:
        count = validate_models()
        if args.check:
            import torch
            import torchaudio
            import transformers
            import wetext
            from PyQt6 import QtCore, QtMultimedia
            from indextts.infer_v2 import IndexTTS2
            print(json.dumps({"python": sys.version.split()[0], "executable": sys.executable,
                              "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                              "transformers": transformers.__version__, "qt": QtCore.QT_VERSION_STR,
                              "cuda_available": torch.cuda.is_available(),
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                              "model_files_checked": count}, ensure_ascii=False, indent=2))
            return 0
        from PyQt6.QtWidgets import QApplication
        from app.window import IndexTTSWindow
        app = QApplication(sys.argv[:1])
        app.setApplicationName("IndexTTS")
        window = IndexTTSWindow()
        window.show()
        return app.exec()
    except Exception as exc:
        logging.exception("启动失败")
        print(f"启动失败：{exc}\n详细日志：logs/indextts.log", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # -I 隔离模式不包含脚本目录，这里仅加入当前入口所属的项目。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
