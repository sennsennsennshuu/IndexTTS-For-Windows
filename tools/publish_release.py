"""发布已验证的 v1.2.0 发行包；凭据仅从现有 Git 登录状态读取到内存。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import requests

REPOSITORY = "sennsennsennshuu/IndexTTS-For-Windows"
TAG = "v1.2.0"
OUTPUT = Path("build/release_v1.2.0")


def get_session():
    """使用 Git 已有凭据，不将密码写文件、命令参数或工具输出。"""
    result = subprocess.run(["git", "-c", "credential.interactive=never", "credential", "fill"],
                            input="protocol=https\nhost=github.com\n\n", text=True, capture_output=True,
                            env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never"), timeout=30)
    if result.returncode:
        raise RuntimeError("无法使用已有 Git 登录状态")
    credential = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = credential.get("password")
    if not token:
        raise RuntimeError("Git 登录状态缺少访问凭据")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                            "X-GitHub-Api-Version": "2026-03-10", "User-Agent": "IndexTTS-Release-Publisher"})
    return session


def api(session, method, suffix, data=None):
    url = f"https://api.github.com/repos/{REPOSITORY}" + (f"/{suffix}" if suffix else "")
    response = session.request(method, url, json=data, timeout=(20, 45))
    if response.status_code >= 300:
        raise RuntimeError(f"GitHub {method} {suffix}: HTTP {response.status_code}: {response.text[:600]}")
    return response.json() if response.content else None


class UploadStream:
    """流式上传大附件，避免把整个压缩包载入内存；记录可恢复的进度。"""
    def __init__(self, path):
        self.file = path.open("rb")
        self.size = path.stat().st_size
        self.name = path.name
        self.sent = 0
        self.next_report = 32 * 1024**2
        self.last_report = time.monotonic()

    def __len__(self):
        return self.size

    def read(self, amount=-1):
        data = self.file.read(amount)
        self.sent += len(data)
        if self.sent >= self.next_report or time.monotonic() - self.last_report > 20:
            print(f"上传 {self.name}: {self.sent/1024**2:.1f}/{self.size/1024**2:.1f} MiB ({100*self.sent/self.size:.1f}%)", flush=True)
            self.next_report = self.sent + 32 * 1024**2
            self.last_report = time.monotonic()
        return data

    def close(self):
        self.file.close()


def upload_asset(session, release, path):
    """同名已完成附件会按大小及 SHA-256 复用，失败残留只在 draft 中重传。"""
    checksum = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024**2), b""):
            checksum.update(block)
    expected_digest = "sha256:" + checksum.hexdigest()
    assets = api(session, "GET", f"releases/{release['id']}/assets")
    for asset in assets:
        if asset["name"] != path.name:
            continue
        if asset["state"] == "uploaded" and asset["size"] == path.stat().st_size and asset.get("digest") == expected_digest:
            print(f"附件已验证，复用：{path.name}", flush=True)
            return asset
        if not release["draft"]:
            raise RuntimeError("已发布版本存在不一致附件，停止覆盖")
        api(session, "DELETE", f"releases/assets/{asset['id']}")
    url = release["upload_url"].split("{")[0] + "?name=" + quote(path.name)
    stream = UploadStream(path)
    try:
        response = session.post(url, data=stream, headers={"Content-Type": "application/octet-stream",
                                                        "Content-Length": str(len(stream))}, timeout=(30, 60))
    finally:
        stream.close()
    if response.status_code != 201:
        raise RuntimeError(f"附件上传失败 {path.name}: HTTP {response.status_code}: {response.text[:400]}")
    asset = response.json()
    if asset["size"] != path.stat().st_size or asset.get("digest") != expected_digest:
        raise RuntimeError(f"远端附件 SHA-256 或大小不一致：{path.name}")
    print(f"附件校验完成：{path.name} ({asset['size']} B)", flush=True)
    return asset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True, help="已推送并打上 v1.2.0 标签的提交")
    parser.add_argument("--lightweight", action="store_true", help="发布用户选择的首次启动安装 even/ 的轻量包")
    args = parser.parse_args()
    output = Path("build/release_v1.2.0_light") if args.lightweight else OUTPUT
    verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    assert verification["crc_passed"] and verification["models_excluded_verified"]
    if args.lightweight:
        assert verification["bootstrap_python_installed"] and verification["dependencies_resolved"]
        assert verification["regression_tests_passed"] == 16
    else:
        assert verification["relocation_smoke"]["four_ui_modes"] and verification["missing_models_reported"]
        assert verification["pip_check"] == "No broken requirements found."
    manifest = json.loads((output / "release_manifest.json").read_text(encoding="utf-8"))
    archives = [output / item["name"] for item in manifest["archives"]]
    assert all(path.stat().st_size < 2 * 1024**3 for path in archives)
    session = get_session()
    reference = api(session, "GET", "git/ref/tags/" + TAG)
    if reference["object"]["sha"] != args.commit:
        raise RuntimeError("v1.2.0 标签与待发布提交不一致")
    repo = api(session, "GET", "")
    assert repo["id"] == 1241927776
    releases = api(session, "GET", "releases?per_page=30")
    release = next((r for r in releases if r["tag_name"] == TAG), None)
    if release is not None and not release["draft"]:
        raise RuntimeError("v1.2.0 已发布，本工具不直接覆盖已发布版本")
    if release is None:
        release = next((r for r in releases if r["id"] == 324201366 and r["draft"]), None)
    body = Path("docs/RELEASE_v1.2.0.md").read_text(encoding="utf-8")
    body += f"\n\n![新版语音工作台](https://raw.githubusercontent.com/{REPOSITORY}/{TAG}/docs/desktop_result.png)\n"
    payload = {"tag_name": TAG, "target_commitish": args.commit, "name": "v1.2.0 · Windows 深色工作台（不含模型）",
               "body": body, "draft": True, "prerelease": False}
    release = api(session, "PATCH", f"releases/{release['id']}", payload) if release else api(session, "POST", "releases", payload)
    (output / "draft_state.json").write_text(json.dumps({"release_id": release["id"], "tag": TAG}, indent=2), encoding="utf-8")
    print(f"Release 草稿已准备：id={release['id']} / {TAG}", flush=True)
    attachments = [*archives, output / "SHA256SUMS.txt", output / "release_manifest.json", Path("docs/RELEASE_v1.2.0.md")]
    # 两个环境附件互不依赖，可并行上传；程序和说明先上传，便于独立核对。
    small = [path for path in attachments if path.stat().st_size < 100 * 1024**2]
    large = [path for path in attachments if path not in small]
    uploaded = [upload_asset(session, release, path) for path in small]
    with ThreadPoolExecutor(max_workers=2) as pool:
        uploaded.extend(pool.map(lambda path: upload_asset(session, release, path), large))
    # 新附件均上传且服务端摘要一致后，再移除旧草稿中的前版成品。
    expected_names = {p.name for p in attachments}
    for asset in api(session, "GET", f"releases/{release['id']}/assets"):
        if asset["name"] not in expected_names:
            api(session, "DELETE", f"releases/assets/{asset['id']}")
            print(f"已清理旧草稿附件：{asset['name']}", flush=True)
    published = api(session, "PATCH", f"releases/{release['id']}", {"draft": False, "make_latest": "true"})
    latest = api(session, "GET", "releases/latest")
    assert latest["id"] == published["id"] and latest["tag_name"] == TAG and not latest["draft"]
    result = {"repository": REPOSITORY, "tag": TAG, "commit": args.commit, "release_id": published["id"],
              "url": published["html_url"], "published_at": published["published_at"], "is_latest": True,
              "models_uploaded": False, "assets": [{key: a.get(key) for key in ("id", "name", "size", "digest", "browser_download_url")}
                                                     for a in latest["assets"]]}
    Path("docs/release_publish.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
