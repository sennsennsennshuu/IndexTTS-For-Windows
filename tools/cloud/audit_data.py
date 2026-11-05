"""训练前检查完整文本窗口、语言编号和数据隔离，不靠训练中途崩溃发现错配。"""

from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import configure, encode_text
from prepare_data import canonical_hanzi


def main() -> None:
    configure()
    from indextts.utils.tokenizer import LANGUAGE_DICT, get_tokenizer

    if LANGUAGE_DICT["minnan"] != 100:
        raise ValueError("语言表与训练约定不同")
    tokenizer = get_tokenizer(multilingual=True, model_dir="models/IndexTTS-2.5")
    groups = {}
    lengths = []
    counts = Counter()
    for split in ("train", "valid", "test"):
        keys = {"audio": set(), "text": set(), "speaker": set()}
        path = Path(f"training/data/minnan/{split}.cloud.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["split"] != split:
                raise ValueError("清单 split 字段不一致")
            audio = Path(row["audio"])
            if audio.is_absolute() or not audio.resolve().is_relative_to(Path("/root/autodl-tmp")):
                raise ValueError(f"音频路径越出 tmp 数据盘：{audio}")
            if not audio.is_file():
                raise FileNotFoundError(audio)
            tokens = encode_text(tokenizer, row["text"])
            if max(tokens) >= 60509 or min(tokens) < 0:
                raise ValueError("文本 token ID 超出固定基础词表")
            lengths.append(len(tokens))
            counts[f"{split}:{row['text_type']}"] += 1
            keys["audio"].add(row["audio_sha256"])
            keys["text"].add(canonical_hanzi(row["hanzi"]))
            if row.get("speaker_id"):
                keys["speaker"].add(row["speaker_id"])
        groups[split] = keys
    for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
        for kind in ("audio", "text", "speaker"):
            if groups[first][kind] & groups[second][kind]:
                raise ValueError(f"{first}/{second} 存在 {kind} 泄漏")
    lengths.sort()
    report = {"passed": True, "view_rows": len(lengths), "unique_audio": sum(len(group["audio"]) for group in groups.values()),
              "token_max": max(lengths), "token_p99": lengths[int(len(lengths) * 0.99)], "counts": dict(counts),
              "text_prefix": "<|zh|> ", "language_id": 100, "vocabulary_reindexed": False}
    Path("logs/data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
