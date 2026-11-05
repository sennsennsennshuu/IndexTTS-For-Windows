"""真实 Windows 窗口布局验收：普通、最小和最大化状态，不加载翻译或语音模型。"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import prepare_runtime


def main():
    prepare_runtime()
    from PyQt6.QtCore import QTimer, QPoint
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication, QLabel, QStyle, QStyleOptionComboBox
    from app.window import IndexTTSWindow

    app = QApplication(["IndexTTS 布局验收"])
    window = IndexTTSWindow()
    window.language_combo.setCurrentIndex(window.language_combo.findData("ZH-NAN-TW"))
    window.text_input.setPlainText("我今天下午要去医院看医生。")
    window._cancel_minnan_preview()
    # 仅为观察布局填入此前已验证的固定译文，不在界面验收中重复加载翻译模型。
    window._set_minnan_preview_text("我今仔日下晡欲去病院看醫生。")
    window.minnan_status.setText("台语转换结果可直接编辑。")
    window.minnan_status_hint.setText("选择输入方式与翻译模型后，可继续编辑原文和译文。")
    result = {"qt_platform": QGuiApplication.platformName(), "fixture_preview": True, "states": []}
    stages = (("normal", (1180, 820)), ("compact", (1000, 700)), ("maximized", None))
    position = 0

    def finish(error=None):
        if error:
            result["error"] = str(error)
        result["passed"] = not error and len(result["states"]) == len(stages)
        Path("logs/minnan_ui_layout_verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        window.close()
        app.exit(0 if result["passed"] else 1)

    def arrange():
        name, size = stages[position]
        if size:
            window.showNormal()
            window.resize(*size)
        else:
            window.showMaximized()
        QTimer.singleShot(160, capture)

    def capture():
        nonlocal position
        try:
            name, requested = stages[position]
            if requested:
                assert [window.width(), window.height()] == list(requested)
            else:
                assert window.isMaximized()
            labels = window.findChildren(QLabel)
            assert not any(label.text() == "VOICE STUDIO" for label in labels)
            assert any(label.text() == "zeroshot 语音合成" for label in labels)
            assert not window.findChildren(QLabel, "minnanWarning")
            record = {"name": name, "requested": requested,
                      "actual": [window.width(), window.height()], "fields": {}}
            for key in ("minnan_mode_combo", "minnan_translation_combo"):
                combo = getattr(window, key)
                option = QStyleOptionComboBox()
                combo.initStyleOption(option)
                edit = combo.style().subControlRect(
                    QStyle.ComplexControl.CC_ComboBox, option,
                    QStyle.SubControl.SC_ComboBoxEditField, combo)
                text_width = combo.fontMetrics().horizontalAdvance(combo.currentText())
                assert edit.width() >= text_width, f"{name}/{key} 文本空间不足"
                record["fields"][key] = {"text_width": text_width, "available_text_width": edit.width()}
            assert not window.minnan_mode_combo.geometry().intersects(
                window.minnan_translation_combo.geometry())
            # 小窗口优先保证整组标签与选择框同屏，译文和原文通过正文滚动访问。
            controls_top = window.minnan_controls.mapTo(window.editor_scroll.widget(), QPoint(0, 0)).y()
            window.editor_scroll.verticalScrollBar().setValue(max(0, controls_top - 6))
            app.processEvents()
            viewport = window.editor_scroll.viewport()
            for field in (window.minnan_mode_label, window.minnan_translation_label,
                          window.minnan_mode_combo, window.minnan_translation_combo):
                assert viewport.rect().contains(field.mapTo(viewport, field.rect().topLeft()))
                assert viewport.rect().contains(field.mapTo(viewport, field.rect().bottomRight()))
            screenshot = f"docs/minnan_layout_{name}.png"
            assert window.grab().save(screenshot)
            record["screenshot"] = screenshot
            record["editor_scroll_range"] = window.editor_scroll.verticalScrollBar().maximum()
            result["states"].append(record)
            position += 1
            if position == len(stages):
                finish()
            else:
                arrange()
        except Exception as exc:
            finish(exc)

    window.show()
    QTimer.singleShot(160, arrange)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
