"""实际比较两种普通话转台语模型，保存原始输出；不以非空字符串冒充语言质量验收。"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import prepare_runtime


def main():
    prepare_runtime()
    from app.minnan_words import LocalMinnanTranslator, TaigiMinnanTranslator, MinnanTextProcessor
    sentences = ["我很早起床。", "你把水倒得太满了。", "我今天下午要去医院看医生。", "你好，谢谢你的帮忙。"]
    results = []
    for backend, model in (("marian", LocalMinnanTranslator), ("taigi_llama_2_q4_k_m", TaigiMinnanTranslator)):
        for text in sentences:
            start = time.monotonic()
            output = model.translate(text)
            assert output.strip() and "[TRANS]" not in output
            row = {"backend": backend, "source": text, "output": output, "seconds": time.monotonic()-start}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if backend == "marian":
            # 对照模型已完成验证，释放 CPU 权重后再加载 7B，降低本机峰值内存。
            LocalMinnanTranslator._model = None
            LocalMinnanTranslator._tokenizer = None
            gc.collect()
    processor = MinnanTextProcessor()
    prepared = processor.prepare("我今天下午要去医院看医生。")
    result = {"cases": results, "actual_default": prepared.__dict__, "language_expert_review_passed": False}
    Path("logs/minnan_translation_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding="utf-8")
    print(json.dumps({"actual_default": prepared.__dict__}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
