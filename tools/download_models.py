"""将官方 2.5 权重下载到项目内；断点续传、校验后才替换成正式文件。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

import requests


MODEL_REPO = "IndexTeam/IndexTTS-2.5"
MODEL_REVISION = "c39ce5ba981572cb187443877ff559dfb246ce63"
TARGET = Path("models/IndexTTS-2.5")
AUXILIARY = {
    "facebook/w2v-bert-2.0": {
        "config.json": "hf_cache/w2v-bert-2.0/config.json",
        "model.safetensors": "hf_cache/w2v-bert-2.0/model.safetensors",
        "preprocessor_config.json": "hf_cache/w2v-bert-2.0/preprocessor_config.json",
    },
    "funasr/campplus": {"campplus_cn_common.bin": "hf_cache/campplus_cn_common.bin"},
    "nvidia/bigvgan_v2_22khz_80band_256x": {
        "config.json": "hf_cache/bigvgan/config.json",
        "bigvgan_generator.pt": "hf_cache/bigvgan/bigvgan_generator.pt",
    },
}


def file_digest(path, algorithm="sha256"):
    """流式计算大文件摘要；普通 Git 文件使用带头部的 Git blob SHA-1。"""
    checksum = hashlib.new("sha1" if algorithm == "git-sha1" else algorithm)
    if algorithm == "git-sha1":
        checksum.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def metadata(repo, revision="main"):
    """从官方 API 固定仓库修订和每个文件的预期大小、摘要。"""
    response = requests.get(
        f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=40)
    response.raise_for_status()
    return response.json()


def make_entry(repo, revision, item, destination):
    lfs = item.get("lfs") or {}
    return {"repo": repo, "revision": revision, "remote": item["rfilename"],
            "path": destination, "bytes": item["size"],
            "algorithm": "sha256" if lfs else "git-sha1",
            "digest": lfs.get("sha256") or item["blobId"]}


def matches(path, entry):
    return (path.is_file() and path.stat().st_size == entry["bytes"]
            and file_digest(path, entry["algorithm"]) == entry["digest"])


def download(entry, reusable):
    """重用相同摘要文件；网络失败保留 .part，下一次继续传输。"""
    target = TARGET / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if matches(target, entry):
        print(f"已校验：{entry['path']}", flush=True)
        return
    source = reusable.get((entry["algorithm"], entry["digest"]))
    if source and matches(source, entry):
        shutil.copy2(source, target)
        print(f"复用相同权重：{entry['path']}", flush=True)
        return
    partial = target.with_name(target.name + ".part")
    last_error = None
    for attempt in range(4):
        endpoint = ("https://hf-mirror.com", "https://huggingface.co")[attempt % 2]
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > entry["bytes"]:
                partial.unlink()
                offset = 0
            if offset != entry["bytes"]:
                url = f"{endpoint}/{entry['repo']}/resolve/{entry['revision']}/{entry['remote']}"
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        offset = 0  # 镜像不支持续传时重新写，不能把完整响应追加到旧文件。
                    if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("服务器返回了错误的续传范围")
                    print(f"下载：{entry['path']}，从 {offset / 1024**2:.1f} MiB 开始", flush=True)
                    written, progress = offset, offset
                    with partial.open("ab" if offset else "wb") as stream:
                        for block in response.iter_content(4 * 1024 * 1024):
                            stream.write(block)
                            written += len(block)
                            if written > entry["bytes"]:
                                raise RuntimeError("响应超出预期文件大小")
                            if written - progress >= 128 * 1024**2:
                                print(f"进度：{entry['path']} {written / entry['bytes']:.0%}", flush=True)
                                progress = written
            if not matches(partial, entry):
                if partial.stat().st_size >= entry["bytes"]:
                    partial.unlink()  # 完整但摘要不符的响应不可继续复用。
                raise RuntimeError(f"文件大小或摘要不一致：{entry['path']}")
            partial.replace(target)
            print(f"下载并校验通过：{entry['path']}", flush=True)
            return
        except (OSError, requests.RequestException, RuntimeError) as exc:
            last_error = exc
            print(f"重试 {entry['path']}：{exc}", flush=True)
            time.sleep(1)
    raise RuntimeError(f"下载失败：{entry['path']}：{last_error}")


def main():
    parser = argparse.ArgumentParser(description="下载官方 IndexTTS 2.5 模型，不上传任何权重")
    parser.add_argument("--reuse-dir", type=Path, help="可选的项目内现有模型目录，仅复用哈希相同的文件")
    args = parser.parse_args()
    reusable = {}
    if args.reuse_dir:
        if not args.reuse_dir.resolve().is_relative_to(Path.cwd().resolve()):
            parser.error("复用目录必须在当前项目内")
        previous = json.loads(Path("docs/model_manifest.json").read_text(encoding="utf-8"))
        for relative, record in previous["files"].items():
            reusable[("sha256", record["sha256"])] = args.reuse_dir / relative
    info = metadata(MODEL_REPO, MODEL_REVISION)
    entries = [make_entry(MODEL_REPO, info["sha"], item, item["rfilename"])
               for item in info["siblings"]
               if item["rfilename"] not in {".gitattributes", "README.md"}]
    revisions = {MODEL_REPO: info["sha"]}
    for repo, files in AUXILIARY.items():
        info = metadata(repo)
        revisions[repo] = info["sha"]
        by_name = {item["rfilename"]: item for item in info["siblings"]}
        entries.extend(make_entry(repo, info["sha"], by_name[name], target) for name, target in files.items())
    # 小型辅助配置可按实际摘要匹配复用，避免为了几个 KB 再访问下载端。
    if args.reuse_dir:
        for source in args.reuse_dir.rglob("*"):
            if source.is_file() and source.stat().st_size < 4 * 1024**2:
                reusable[("git-sha1", file_digest(source, "git-sha1"))] = source
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, entry, reusable) for entry in entries]
        for future in as_completed(futures):
            future.result()
    files = {}
    for entry in entries:
        path = TARGET / entry["path"]
        files[entry["path"]] = {"bytes": path.stat().st_size, "sha256": file_digest(path),
                                "repo": entry["repo"], "revision": entry["revision"]}
    manifest = {"model_version": "2.5", "algorithm": "sha256", "upstream_digest_verified": True,
                "revisions": revisions, "file_count": len(files),
                "total_bytes": sum(item["bytes"] for item in files.values()), "files": files}
    Path("docs/model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"全部模型校验完成：{len(files)} 个文件。", flush=True)


if __name__ == "__main__":
    main()
