"""台语界面回归：显隐、异步预览、未命中提示、参数传递与线程生命周期。"""
from dataclasses import dataclass
from pathlib import Path
import os
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import prepare_runtime
prepare_runtime()
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QStyle, QStyleOptionComboBox
from app.window import IndexTTSWindow, MINNAN_LANGUAGE_CODE


@dataclass
class FakePreparedMinnan:
    """模拟约定的 PreparedMinnan，只暴露界面实际消费的字段。"""
    text: str
    hanzi: str = ""
    tailo: str = ""
    matches: list | None = None
    unresolved: list | None = None
    warnings: list | None = None
    method: str = "test"


class FakeMinnanProcessor:
    """可配置延迟、结果和异常的处理器；用于验证线程行为而非词库质量。"""
    delay = 0.0
    prepared = None
    error = None
    calls = []

    def prepare(self, text, mode="mandarin", translation_mode="full"):
        type(self).calls.append((text, mode, translation_mode))
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.prepared or FakePreparedMinnan(text=text, matches=[])


class MinnanUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # 核心 API 可能由另一并行任务写入；测试注入独立替身，保证结果确定。
        cls.fake_module = types.ModuleType("app.minnan_words")
        cls.fake_module.MinnanTextProcessor = FakeMinnanProcessor
        cls.original_module = sys.modules.get("app.minnan_words")
        sys.modules["app.minnan_words"] = cls.fake_module

    @classmethod
    def tearDownClass(cls):
        if cls.original_module is None:
            sys.modules.pop("app.minnan_words", None)
        else:
            sys.modules["app.minnan_words"] = cls.original_module

    def setUp(self):
        FakeMinnanProcessor.delay = 0.0
        FakeMinnanProcessor.error = None
        FakeMinnanProcessor.prepared = None
        FakeMinnanProcessor.calls = []
        self.window = IndexTTSWindow()
        self.window.auto_play.setChecked(False)

    def tearDown(self):
        self.wait_preview()
        self.window.close()
        self.app.processEvents()

    def select_minnan(self, text=""):
        """真实 LANGUAGES 尚未合并时补入同一语言代码，不修改 paths.py。"""
        combo = self.window.language_combo
        index = -1
        for i in range(combo.count()):
            code = str(combo.itemData(i) or "").upper().replace("_", "-")
            if code == MINNAN_LANGUAGE_CODE:
                index = i
                break
        if index < 0:
            combo.addItem("台语（台湾闽南语）", MINNAN_LANGUAGE_CODE)
            index = combo.count() - 1
        self.window.text_input.setPlainText(text)
        combo.setCurrentIndex(index)
        self.app.processEvents()

    def wait_preview(self, timeout=4.0):
        """同时等待防抖计时器和 QThread finished 清理完成。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if (self.window.minnan_preview_worker is None and
                    not self.window._minnan_preview_timer.isActive() and
                    self.window._pending_minnan_preview is None):
                return
            time.sleep(0.01)
        self.fail("台语预览线程未在期限内结束")

    def test_panel_is_hidden_for_existing_languages_and_shown_for_minnan(self):
        self.window.show()
        self.assertFalse(self.window.minnan_panel.isVisible())
        self.select_minnan()
        self.assertTrue(self.window.minnan_panel.isVisible())
        self.assertEqual(self.window.minnan_mode_combo.currentData(), "mandarin")
        self.assertFalse(self.window.minnan_preview.isReadOnly())
        self.assertFalse(self.window.findChildren(QLabel, "minnanWarning"))
        labels = [self.window.minnan_mode_combo.itemText(i)
                  for i in range(self.window.minnan_mode_combo.count())]
        self.assertEqual(labels, ["普通话（自动转台语，推荐）", "台语汉字", "带声调台罗"])
        self.assertTrue(self.window.minnan_translation_combo.isVisible())
        self.assertEqual(
            [self.window.minnan_translation_combo.itemText(i)
             for i in range(self.window.minnan_translation_combo.count())],
            ["完整", "精简", "不使用"])
        self.assertEqual(self.window.minnan_translation_combo.currentData(), "full")
        self.assertIn("313", self.window.minnan_translation_combo.itemData(
            1, Qt.ItemDataRole.ToolTipRole))
        self.assertIn("未命中", self.window.minnan_translation_combo.itemData(
            2, Qt.ItemDataRole.ToolTipRole))
        self.window.minnan_mode_combo.setCurrentIndex(1)
        self.app.processEvents()
        self.assertFalse(self.window.minnan_translation_combo.isVisible())
        self.window.minnan_mode_combo.setCurrentIndex(0)
        self.app.processEvents()
        # 切回任一原语言，台语区域必须消失且不保留旧结果。
        self.window.language_combo.setCurrentIndex(0)
        self.app.processEvents()
        self.assertFalse(self.window.minnan_panel.isVisible())

    def test_normal_window_fits_controls_without_forcing_larger_height(self):
        """复现非全屏拥挤：保留指定窗口尺寸、完整显示选项，并验证滚动可达性。"""
        self.window.show()
        self.select_minnan()
        for width, height in ((1000, 700), (1180, 820), (1500, 900)):
            with self.subTest(size=(width, height)):
                self.window.resize(width, height)
                self.app.processEvents()
                self.assertEqual((self.window.width(), self.window.height()), (width, height))
                first = self.window.minnan_mode_combo
                second = self.window.minnan_translation_combo
                self.assertFalse(first.geometry().intersects(second.geometry()))
                for combo in (first, second):
                    option = QStyleOptionComboBox()
                    combo.initStyleOption(option)
                    edit = combo.style().subControlRect(
                        QStyle.ComplexControl.CC_ComboBox, option,
                        QStyle.SubControl.SC_ComboBoxEditField, combo)
                    self.assertGreaterEqual(
                        edit.width(), combo.fontMetrics().horizontalAdvance(combo.currentText()))
                self.window.editor_scroll.ensureWidgetVisible(self.window.minnan_controls, 0, 4)
                self.app.processEvents()
                viewport = self.window.editor_scroll.viewport()
                center = first.mapTo(viewport, first.rect().center())
                self.assertTrue(viewport.rect().contains(center))
                # 原文和台语译文仍可编辑；通过滚动能到达原文，不截断任一输入框。
                self.window.editor_scroll.ensureWidgetVisible(self.window.text_input, 0, 4)
                self.app.processEvents()
                center = self.window.text_input.mapTo(viewport, self.window.text_input.rect().center())
                self.assertTrue(viewport.rect().contains(center))
                before = first.width()
                self.window.minnan_mode_combo.setCurrentIndex(1)
                self.app.processEvents()
                self.assertGreater(first.width(), before)
                self.window.minnan_mode_combo.setCurrentIndex(0)
                self.app.processEvents()

    def test_async_preview_keeps_source_and_displays_prepared_text(self):
        FakeMinnanProcessor.prepared = FakePreparedMinnan(
            text="guá beh khì", hanzi="我欲去", tailo="guá beh khì",
            matches=[{"source": "我"}, {"source": "要"}], unresolved=[])
        self.select_minnan("我要去")
        self.wait_preview()
        self.assertEqual(FakeMinnanProcessor.calls, [("我要去", "mandarin", "full")])
        self.assertEqual(self.window.text_input.toPlainText(), "我要去")
        self.assertEqual(self.window.minnan_preview.toPlainText(), "guá beh khì")
        self.assertEqual(self.window.minnan_status.property("state"), "done")
        self.assertIn("2 条来源对照", self.window.minnan_status.text())
        self.assertIn("请检查", self.window.minnan_status.text())

    def test_unresolved_terms_are_explicit_and_never_silently_use_mandarin(self):
        FakeMinnanProcessor.prepared = FakePreparedMinnan(
            text="我 beh khì", matches=[{"source": "要"}],
            unresolved=["我", {"text": "明天"}])
        self.select_minnan("我要去明天")
        self.wait_preview()
        self.assertEqual(self.window.minnan_status.property("state"), "error")
        self.assertIn("2 处未命中", self.window.minnan_status.text())
        self.assertIn("我、明天", self.window.minnan_status_hint.text())
        self.assertIn("请检查", self.window.minnan_status.text())
        self.assertNotEqual(self.window.minnan_preview.toPlainText(), "我要去明天")

    def test_preview_exception_is_visible_and_does_not_keep_auto_result(self):
        FakeMinnanProcessor.error = RuntimeError("词库文件缺失")
        self.select_minnan("我要去")
        self.wait_preview()
        self.assertEqual(self.window.minnan_status.property("state"), "error")
        self.assertIn("暂不可用", self.window.minnan_status.text())
        self.assertIn("词库文件缺失", self.window.minnan_status_hint.text())
        self.assertEqual(self.window.minnan_preview.toPlainText(), "")

    def test_manual_edit_survives_late_worker_result_and_drives_generation(self):
        FakeMinnanProcessor.delay = 0.18
        FakeMinnanProcessor.prepared = FakePreparedMinnan(text="guá beh khì", matches=[])
        self.select_minnan("我要去")
        self.window._minnan_preview_timer.stop()
        self.window._start_minnan_preview()
        self.assertIsNotNone(self.window.minnan_preview_worker)
        self.window.minnan_preview.setPlainText("guá ài khì")
        self.assertTrue(self.window._minnan_preview_edited)
        self.wait_preview()
        self.assertEqual(self.window.minnan_preview.toPlainText(), "guá ài khì")
        params = self.window.collect_minnan_params()
        self.assertEqual(params["minnan_text"], "guá ài khì")
        self.assertEqual(params["minnan_input_mode"], "tailo")

    def test_generation_passes_prepared_text_and_callback_result(self):
        FakeMinnanProcessor.prepared = FakePreparedMinnan(
            text="guá beh khì", matches=[])
        self.select_minnan("我要去")
        self.wait_preview()
        self.window.speaker_audio_input.setText("test/fixtures/reference.wav")
        engine = Mock()
        engine.is_loaded = True
        engine.infer.return_value = "outputs/2.5验收_情感模式0.wav"
        def fake_infer(text, *args, **kwargs):
            # 模拟引擎准备台语文本后回调；界面必须只更新结果区。
            engine.text_prepared_callback("guá beh khì")
            return engine.infer.return_value
        engine.infer.side_effect = fake_infer
        self.window.engine = engine
        with patch.object(self.window, "get_indextts_model_path", return_value="models"), \
             patch.object(self.window, "play_generated_audio"):
            self.window.generate_speech()
            self.wait_generation()
        self.assertEqual(engine.infer.call_args.kwargs["minnan_input_mode"], "tailo")
        self.assertEqual(engine.infer.call_args.kwargs["lang"], MINNAN_LANGUAGE_CODE)
        self.assertEqual(engine.infer.call_args.args[0], "guá beh khì")
        self.assertEqual(self.window.text_input.toPlainText(), "我要去")
        self.assertEqual(self.window.minnan_preview.toPlainText(), "guá beh khì")
        self.assertIsNone(engine.text_prepared_callback)

        engine.reset_mock()
        engine.infer.side_effect = None
        self.window.language_combo.setCurrentIndex(0)
        self.app.processEvents()
        with patch.object(self.window, "get_indextts_model_path", return_value="models"), \
             patch.object(self.window, "play_generated_audio"):
            self.window.generate_speech()
            self.wait_generation()
        self.assertNotIn("minnan_input_mode", engine.infer.call_args.kwargs)
        self.assertNotIn("minnan_translation_mode", engine.infer.call_args.kwargs)

    def test_translation_mode_is_passed_to_preview_and_engine(self):
        """默认 full 会同时进入 prepare 和 engine.infer 参数。"""
        FakeMinnanProcessor.prepared = FakePreparedMinnan(text="guá beh khì", matches=[])
        self.select_minnan("我要去")
        self.wait_preview()
        self.assertEqual(FakeMinnanProcessor.calls, [("我要去", "mandarin", "full")])
        params = self.window.collect_minnan_params()
        self.assertEqual(params["minnan_translation_mode"], "full")

        # 精简档必须同时进入新一轮 prepare 和引擎参数，不能沿用完整档译文。
        self.window.minnan_translation_combo.setCurrentIndex(1)
        self.wait_preview()
        self.assertEqual(FakeMinnanProcessor.calls[-1], ("我要去", "mandarin", "compact"))
        self.assertEqual(
            self.window.collect_minnan_params()["minnan_translation_mode"], "compact")

        self.window.speaker_audio_input.setText("test/fixtures/reference.wav")
        engine = Mock()
        engine.is_loaded = True
        engine.infer.return_value = "outputs/2.5验收_情感模式0.wav"
        self.window.engine = engine
        with patch.object(self.window, "get_indextts_model_path", return_value="models"), \
             patch.object(self.window, "play_generated_audio"):
            self.window.generate_speech()
            self.wait_generation()
        self.assertEqual(engine.infer.call_args.kwargs["minnan_translation_mode"], "compact")

    def test_translation_mode_switch_invalidates_inflight_result(self):
        """旧档位返回时 request_id 已失效，不能覆盖新档位结果。"""
        FakeMinnanProcessor.delay = 0.22
        FakeMinnanProcessor.prepared = FakePreparedMinnan(text="full-old", matches=[])
        self.select_minnan("我要去")
        self.window._minnan_preview_timer.stop()
        self.window._start_minnan_preview()
        self.assertIsNotNone(self.window.minnan_preview_worker)
        self.window.minnan_translation_combo.setCurrentIndex(1)
        self.app.processEvents()
        self.wait_preview()
        self.assertEqual(FakeMinnanProcessor.calls[0], ("我要去", "mandarin", "full"))
        self.assertEqual(FakeMinnanProcessor.calls[-1], ("我要去", "mandarin", "compact"))
        self.assertEqual(self.window.collect_minnan_params()["minnan_translation_mode"], "compact")
        self.assertEqual(self.window._minnan_preview_translation_mode, "compact")

    def test_direct_hanzi_and_tailo_use_none_and_hide_selector(self):
        """直接输入台语时不加载翻译模型，并传 none 给引擎。"""
        self.select_minnan("我欲去")
        self.window.minnan_mode_combo.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(self.window.collect_minnan_params()["minnan_translation_mode"], "none")
        self.assertFalse(self.window.minnan_translation_combo.isVisible())

    def test_none_translation_mode_surfaces_dictionary_miss(self):
        """不使用翻译器时，词库未命中必须报错且不能保留普通话结果。"""
        FakeMinnanProcessor.error = RuntimeError("词库未命中：逐家")
        self.select_minnan("逐家好")
        self.window.minnan_translation_combo.setCurrentIndex(2)
        self.wait_preview()
        self.assertEqual(FakeMinnanProcessor.calls[-1], ("逐家好", "mandarin", "none"))
        self.assertEqual(self.window.minnan_preview.toPlainText(), "")
        self.assertEqual(self.window.minnan_status.property("state"), "error")
        self.assertIn("未命中", self.window.minnan_status_hint.text())

    def test_non_minnan_has_no_translation_mode_parameters(self):
        """原五语言调用不得携带台语翻译参数。"""
        self.window.text_input.setPlainText("你好")
        self.assertEqual(self.window.collect_minnan_params(), {})

    def test_generation_waits_for_checked_mandarin_translation(self):
        """普通话模式尚未有结果时，不能把原文直接送进台语合成。"""
        FakeMinnanProcessor.delay = 0.3
        self.select_minnan("我要去")
        # 保留真实防抖计时器，先验证“输入后立即点生成”的典型路径。
        self.window.speaker_audio_input.setText("test/fixtures/reference.wav")
        engine = Mock()
        engine.is_loaded = True
        self.window.engine = engine
        with patch.object(self.window, "get_indextts_model_path", return_value="models"):
            self.window.generate_speech()
        self.assertIsNone(self.window.worker)
        # 输入后立即生成时防抖可能尚未触发；先取消，避免测试关闭时留下排队请求。
        self.window._cancel_minnan_preview()
        self.wait_preview()
        engine.infer.assert_not_called()
        self.assertEqual(self.window.status_label.property("state"), "error")
        self.assertIn("先等待台语转换完成", self.window.status_label.text())

    def test_worker_clears_text_prepared_callback_after_failure(self):
        """生成失败也必须解除引擎回调，避免下次请求误投递到已结束线程。"""
        from app.worker import IndexTTSWorker
        class FailingEngine:
            is_loaded = True
            progress_callback = None
            text_prepared_callback = None
            def infer(self, *args, **kwargs):
                raise RuntimeError("预期失败")
        engine = FailingEngine()
        worker = IndexTTSWorker("我要去", "reference.wav", {}, {}, engine)
        worker.start()
        self.assertTrue(worker.wait(3000))
        self.assertIsNone(engine.progress_callback)
        self.assertIsNone(engine.text_prepared_callback)

    def test_close_waits_for_preview_thread(self):
        FakeMinnanProcessor.delay = 0.2
        self.select_minnan("我要去")
        self.window._minnan_preview_timer.stop()
        self.window._start_minnan_preview()
        self.assertIsNotNone(self.window.minnan_preview_worker)
        self.assertFalse(self.window.close())
        self.assertTrue(self.window._close_when_finished)
        deadline = time.monotonic() + 4
        while self.window.minnan_preview_worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertIsNone(self.window.minnan_preview_worker)
        self.app.processEvents()
        self.assertFalse(self.window.isVisible())

    def wait_generation(self, timeout=4.0):
        deadline = time.monotonic() + timeout
        while self.window.worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertIsNone(self.window.worker)


if __name__ == "__main__":
    unittest.main(verbosity=2)
