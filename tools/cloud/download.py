"""依据本地已验证的官方清单下载云端基础模型，所有路径均相对训练根目录。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request


ROOT = Path("models/IndexTTS-2.5")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(name: str, entry: dict) -> dict:
    """保留部分下载，按预期长度及 SHA-256 校验后再替换正式模型。"""
    target = ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == entry["bytes"] and digest(target) == entry["sha256"]:
        return {"path": name, "status": "verified_existing"}
    repo, revision = entry["repo"], entry["revision"]
    remote = name
    if name.startswith("hf_cache/w2v-bert-2.0/"):
        remote = name.removeprefix("hf_cache/w2v-bert-2.0/")
    elif name.startswith("hf_cache/bigvgan/"):
        remote = name.removeprefix("hf_cache/bigvgan/")
    elif name.startswith("hf_cache/"):
        remote = name.removeprefix("hf_cache/")
    temporary = target.with_name(target.name + ".part")
    last_error = None
    for attempt in range(8):
        endpoint = ("https://hf-mirror.com", "https://huggingface.co")[attempt % 2]
        offset = temporary.stat().st_size if temporary.exists() else 0
        try:
            if offset > entry["bytes"]:
                raise ValueError("断点文件大于预期，拒绝追加")
            if offset < entry["bytes"]:
                headers = {"User-Agent": "IndexTTS-Minnan-Training/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                url = f"{endpoint}/{repo}/resolve/{revision}/{remote}?download=true"
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    if offset and response.status != 206:
                        offset = 0
                    if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("下载端点返回了错误的续传范围")
                    print(f"Downloading {name}: {offset}/{entry['bytes']}", flush=True)
                    last_print = time.monotonic()
                    with temporary.open("ab" if offset else "wb") as stream:
                        while chunk := response.read(4 * 1024 * 1024):
                            stream.write(chunk)
                            offset += len(chunk)
                            if offset > entry["bytes"]:
                                raise ValueError("响应长度超过清单")
                            if time.monotonic() - last_print > 25:
                                print(f"Progress {name}: {offset}/{entry['bytes']}", flush=True)
                                last_print = time.monotonic()
            if temporary.stat().st_size != entry["bytes"] or digest(temporary) != entry["sha256"]:
                # 摘要不匹配是实质失败，保留现场，不自动当成有效模型或无限续传。
                raise ValueError(f"模型摘要不匹配：{name}")
            temporary.replace(target)
            print(f"Verified {name}", flush=True)
            return {"path": name, "status": "downloaded_and_verified"}
        except Exception as error:
            last_error = error
            print(f"Retry {attempt + 1} {name}: {type(error).__name__}: {error}", flush=True)
            time.sleep(2)
    raise RuntimeError(f"下载失败：{name}") from last_error


def main() -> None:
    manifest = json.loads(Path("training/base_model_manifest.json").read_text(encoding="utf-8"))
    # 台语训练不需要文本情感大模型，不为此下载 Qwen 权重。
    entries = {name: entry for name, entry in manifest["files"].items() if not name.startswith("qwen0.6bemo4-merge/")}
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = [executor.submit(fetch, name, entry) for name, entry in entries.items()]
        for job in as_completed(jobs):
            results.append(job.result())
    Path("logs/model_verification.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    Path("logs/models_complete").touch()
    print(f"All {len(results)} model files verified", flush=True)


if __name__ == "__main__":
    main()
