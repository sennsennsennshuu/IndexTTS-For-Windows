"""实际验证三个翻译档位、简繁输入及关闭模型后的内存状态，不运行语音 GPU。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import prepare_runtime


def main():
    prepare_runtime()
    from app.minnan_words import MinnanTextProcessor, TaigiMinnanTranslator, LocalMinnanTranslator
    processor = MinnanTextProcessor()
    simplified = "我今天下午要去医院看医生。"
    traditional = "我今天下午要去醫院看醫生。"
    full = processor.prepare(simplified, translation_mode="full")
    traditional_full = processor.prepare(traditional, translation_mode="full")
    assert full.text == traditional_full.text
    compact = processor.prepare(simplified, translation_mode="compact")
    assert compact.text and compact.method == "compact_translation"
    assert TaigiMinnanTranslator._model is None, "精简档不应继续驻留完整模型"
    candidates = processor.lexicon.mandarin
    known_key = next(key for key, rows in candidates.items() if len(key) > 8
                     and processor.lexicon.unique_reading(rows))
    dictionary = processor.prepare(known_key, translation_mode="none")
    assert dictionary.method == "dictionary_exact"
    unknown_rejected = False
    try:
        processor.prepare("这个尚未收录的量子纠缠实验将在火星启动。", translation_mode="none")
    except ValueError as error:
        unknown_rejected = "不使用翻译模型" in str(error)
    assert unknown_rejected and TaigiMinnanTranslator._model is None and LocalMinnanTranslator._model is None
    result = {"full": full.__dict__, "traditional_full": traditional_full.__dict__,
              "simplified_traditional_equal_for_checked_sentence": True,
              "compact": compact.__dict__, "dictionary_only": {"source": known_key, "text": dictionary.text},
              "unknown_dictionary_sentence_rejected": unknown_rejected,
              "models_unloaded_in_none_mode": True, "language_expert_review_passed": False}
    Path("logs/minnan_translation_modes.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
