"""将本机回环端口转发到云端 TensorBoard；凭据仅由 remote.connect 读取。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import socketserver
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect


def status(state: str, port: int, **extra) -> None:
    """状态中只保存进程和本机入口，不记录服务器凭据。"""
    path = Path("logs/tensorboard_tunnel.json")
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps({"pid": os.getpid(), "state": state,
                                    "url": f"http://127.0.0.1:{port}",
                                    "updated_unix": time.time(), **extra}, indent=2), encoding="utf-8")
    temporary.replace(path)


class ForwardServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    transport = None
    remote_port = 6008


class ForwardHandler(socketserver.BaseRequestHandler):
    def handle(self):
        channel = None
        try:
            transport = self.server.transport
            if transport is None or not transport.is_active():
                return
            channel = transport.open_channel("direct-tcpip", ("127.0.0.1", self.server.remote_port),
                                             self.request.getpeername(), timeout=20)
            if channel is None:
                return
            # 双向转发 HTTP 数据；只绑定本机回环地址，不开放公网监听。
            while transport.is_active():
                ready, _, _ = select.select([self.request, channel], [], [], 10)
                for incoming in ready:
                    chunk = incoming.recv(65536)
                    if not chunk:
                        return
                    (channel if incoming is self.request else self.request).sendall(chunk)
        except (OSError, EOFError):
            # 浏览器刷新或 SSH 重连时连接可以正常结束；主循环负责恢复隧道。
            pass
        except Exception as error:
            print(f"forward_error: {type(error).__name__}", flush=True)
        finally:
            if channel is not None:
                channel.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=16007)
    parser.add_argument("--remote-port", type=int, default=6008)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    Path("logs").mkdir(exist_ok=True)
    # 端口绑定本身充当单实例锁；已有隧道时不会启动第二份。
    with ForwardServer(("127.0.0.1", args.port), ForwardHandler) as server:
        server.remote_port = args.remote_port
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            while True:
                try:
                    with connect() as client:
                        server.transport = client.get_transport()
                        status("connected", args.port)
                        print(f"TensorBoard ready: http://127.0.0.1:{args.port}", flush=True)
                        while server.transport.is_active():
                            time.sleep(5)
                except Exception as error:
                    status("reconnecting", args.port, error_type=type(error).__name__)
                    print(f"reconnecting: {type(error).__name__}", flush=True)
                finally:
                    server.transport = None
                time.sleep(5)
        finally:
            server.shutdown()
            status("stopped", args.port)


if __name__ == "__main__":
    main()
