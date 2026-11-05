"""下载固定修订的普通话转台语模型；文件与缓存全部保存在项目相对目录。"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import time

import requests


REPOSITORY = "linguardia/opus-mt-zh-nan"
REVISION = "0c13daaf7758f9863b704571cb48512955a856a6"
DESTINATION = Path("models/extra_language/zh-nan-tw/translation")
FILES = ("README.md", "added_tokens.json", "config.json", "generation_config.json",
         "linguardia_meta.json", "model.safetensors", "source.spm", "target.spm",
         "tokenizer_config.json", "vocab.json")


def checksum(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def matches_upstream(path: Path, expected: str) -> bool:
    """HF 普通文件使用 Git blob SHA-1，LFS 权重使用文件 SHA-256，不混用两种摘要。"""
    if not path.is_file() or not expected:
        return False
    if len(expected) == 64:
        return checksum(path) == expected
    if len(expected) == 40:
        data = path.read_bytes()
        return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == expected
    raise ValueError("上游返回了无法识别的文件摘要")


def main() -> None:
    """先下载部分文件再原子提交；远端提供 LFS SHA 时额外逐项核验。"""
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("marian", "taigi"), default="taigi")
    args = parser.parse_args()
    if args.backend == "taigi":
        repository = "RichardErkhov/Bohanlu_-_Taigi-Llama-2-Translator-7B-gguf"
        revision = "57f8ce7b2109cbfc02cf30f78e3465dc98678e21"
        destination = DESTINATION / "taigi"
        files = ("README.md", "Taigi-Llama-2-Translator-7B.Q4_K_M.gguf")
        license_text = "CC-BY-NC-SA-4.0; original Bohanlu/Taigi-Llama-2-Translator-7B; non-commercial use only"
    else:
        repository, revision, destination, files = REPOSITORY, REVISION, DESTINATION, FILES
        license_text = "Apache-2.0; training data attribution CC-BY-SA-4.0 and CC0-1.0 in README.md"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    session = requests.Session()
    for filename in files:
        target = destination / filename
        url = f"https://huggingface.co/{repository}/resolve/{revision}/{filename}"
        metadata = session.head(url, allow_redirects=False, timeout=40)
        metadata.raise_for_status()
        expected = metadata.headers.get("x-linked-etag", "").strip('"')
        if not matches_upstream(target, expected):
            temporary = target.with_name(target.name + ".part")
            # 上次已下载完但未原子提交时直接核验，避免重复请求整个文件。
            if matches_upstream(temporary, expected):
                temporary.replace(target)
                records.append({"path": filename, "bytes": target.stat().st_size, "sha256": checksum(target)})
                print(json.dumps(records[-1], ensure_ascii=False), flush=True)
                continue
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                last_report = time.monotonic()
                with temporary.open("ab" if append else "wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            output.write(chunk)
                        if time.monotonic() - last_report >= 15:
                            print(json.dumps({"file": filename, "downloaded": output.tell()}), flush=True)
                            last_report = time.monotonic()
            if expected and not matches_upstream(temporary, expected):
                raise ValueError(f"翻译模型摘要不符：{filename}")
            temporary.replace(target)
        records.append({"path": filename, "bytes": target.stat().st_size, "sha256": checksum(target)})
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    if args.backend == "taigi":
        # 量化仓库可能没有完整模型卡；另存原作者声明与提示格式，避免遗失限制。
        card_url = "https://huggingface.co/Bohanlu/Taigi-Llama-2-Translator-7B/raw/main/README.md"
        card = session.get(card_url, timeout=40)
        card.raise_for_status()
        path = destination / "ORIGINAL_MODEL_CARD.md"
        path.write_bytes(card.content)
        records.append({"path": path.name, "bytes": path.stat().st_size, "sha256": checksum(path)})
    value = {"repository": repository, "revision": revision, "downloaded_unix": time.time(),
             "backend": args.backend, "purpose": "Mandarin to Taiwanese Hokkien translation",
             "license": license_text,
             "files": records}
    (destination / "download_manifest.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
