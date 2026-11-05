"""从项目内凭据建立云端连接；不把密码放进命令行、输出或上传包。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import time

import paramiko


def connect() -> paramiko.SSHClient:
    """只解析 SSH 地址、端口与密码字段，不执行凭据文件中的任意文字。"""
    config = Path("ssh.txt").read_text(encoding="utf-8-sig")
    address = re.search(r"([\w.-]+)@([\w.-]+)", config)
    port = re.search(r"-p\s+(\d+)", config)
    password = re.search(r"(?:密码|password)\s*[:：=]\s*(.+)", config, re.I)
    if not all((address, port, password)):
        raise ValueError("ssh.txt 缺少服务器、端口或密码字段")
    client = paramiko.SSHClient()
    host_keys = Path("cache/cloud_known_hosts")
    host_keys.parent.mkdir(parents=True, exist_ok=True)
    if host_keys.exists():
        client.load_host_keys(str(host_keys))
    # 首次连接的主机指纹已人工审计并保存在项目缓存，后续连接拒绝指纹变化。
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        address.group(2), port=int(port.group(1)), username=address.group(1),
        password=password.group(1).strip(), timeout=25, banner_timeout=30,
        auth_timeout=25, look_for_keys=False, allow_agent=False,
    )
    client.get_transport().set_keepalive(20)
    return client


def execute(client: paramiko.SSHClient, command: str) -> int:
    """同步排空两个输出通道，避免长任务卡在 stderr 管道。"""
    channel = client.get_transport().open_session()
    channel.exec_command(command)
    while True:
        if channel.recv_ready():
            sys.stdout.buffer.write(channel.recv(65536))
            sys.stdout.buffer.flush()
        if channel.recv_stderr_ready():
            sys.stderr.buffer.write(channel.recv_stderr(65536))
            sys.stderr.buffer.flush()
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            return channel.recv_exit_status()
        time.sleep(0.05)


def ensure_directory(sftp: paramiko.SFTPClient, path: str) -> None:
    """逐层创建目标目录，已经存在的目录不受影响。"""
    current = PurePosixPath("/")
    for component in PurePosixPath(path).parts[1:]:
        current /= component
        try:
            sftp.stat(str(current))
        except FileNotFoundError:
            sftp.mkdir(str(current))


def upload(client: paramiko.SSHClient, source: Path, destination: str) -> None:
    """上传至临时文件，完成后用 SHA-256 校验并原子提交，支持中断续传。"""
    if not source.is_file():
        raise FileNotFoundError(source)
    local_size = source.stat().st_size
    with source.open("rb") as stream:
        local_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    sftp = client.open_sftp()
    ensure_directory(sftp, str(PurePosixPath(destination).parent))
    partial = destination + ".part"
    try:
        offset = sftp.stat(partial).st_size
    except FileNotFoundError:
        offset = 0
    if offset > local_size:
        raise ValueError("远端续传文件大于本地源文件；保留现场，拒绝覆盖")
    with source.open("rb") as incoming, sftp.open(partial, "ab" if offset else "wb") as outgoing:
        outgoing.set_pipelined(True)
        incoming.seek(offset)
        last_report = time.monotonic()
        while chunk := incoming.read(1024 * 1024):
            outgoing.write(chunk)
            offset += len(chunk)
            if time.monotonic() - last_report >= 30:
                print(json.dumps({"file": source.name, "sent": offset, "total": local_size}), flush=True)
                last_report = time.monotonic()
    verify = (
        "import hashlib,os; "
        f"p={partial!r}; "
        "h=hashlib.file_digest(open(p,'rb'),'sha256').hexdigest(); "
        f"assert h=={local_hash!r}, 'SHA256 mismatch'; "
        f"os.replace(p,{destination!r}); "
        "print('VERIFIED',h,os.path.getsize(" + repr(destination) + "))"
    )
    status = execute(client, "/root/miniconda3/bin/python -c " + shlex.quote(verify))
    sftp.close()
    if status:
        raise RuntimeError("上传摘要验证失败，临时文件已保留")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--command")
    run.add_argument("--script", type=Path)
    run.add_argument("--python", default="/root/miniconda3/bin/python")
    put = subparsers.add_parser("put")
    put.add_argument("source", type=Path)
    put.add_argument("destination")
    get = subparsers.add_parser("get")
    get.add_argument("source")
    get.add_argument("destination", type=Path)
    args = parser.parse_args()
    with connect() as client:
        if args.action == "run":
            command = args.command
            if args.script:
                code = args.script.read_text(encoding="utf-8-sig")
                command = shlex.quote(args.python) + " -u -c " + shlex.quote(code)
            if not command:
                parser.error("run 需要 --command 或 --script")
            raise SystemExit(execute(client, command))
        if args.action == "put":
            upload(client, args.source, args.destination)
        if args.action == "get":
            args.destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.destination.with_name(args.destination.name + ".part")
            with client.open_sftp() as sftp:
                sftp.get(args.source, str(temporary))
            checksum_code = "import hashlib; print(hashlib.file_digest(open(" + repr(args.source) + ",'rb'),'sha256').hexdigest())"
            _, output, errors = client.exec_command("/root/miniconda3/bin/python -c " + shlex.quote(checksum_code))
            remote_hash = output.read().decode().strip()
            if output.channel.recv_exit_status() != 0:
                raise RuntimeError("无法获取远端下载摘要")
            with temporary.open("rb") as stream:
                local_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if local_hash != remote_hash:
                raise ValueError("下载 SHA-256 不匹配，保留临时文件并停止")
            temporary.replace(args.destination)
            print("Downloaded and SHA256 verified:", args.destination, local_hash)


if __name__ == "__main__":
    main()
