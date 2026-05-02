"""MCPConnector — stdio 子进程管理器，实现 MCP JSON-RPC 2.0 协议。

每个 MCPConnector 实例对应一个正在运行的 MCP Server 子进程。
通过 stdin/stdout 进行全双工 JSON-RPC 通信。

用法示例::

    # 文件操作 MCP
    conn = MCPConnector(
        name="filesystem",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp/mcp"],
    )
    conn.start()
    tools = conn.list_tools()
    result = conn.call_tool("read_file", {"path": "/tmp/mcp/data.txt"})
    conn.stop()

    # SQLite MCP
    conn = MCPConnector(
        name="sqlite",
        command=["npx", "-y", "@berthojoris/mcp-sqlite-server",
                 "sqlite:///home/olvan/test.db", "list,read,write"],
    )
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """MCP 协议或运行时错误."""


class MCPConnector:
    """管理单个 MCP Server 子进程的生命周期，并提供同步 RPC 接口。

    Attributes:
        name: 连接器标识名，仅用于日志。
        command: 启动 MCP Server 的完整命令列表，如
                 ``["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp/mcp"]``。
        env: 额外的环境变量字典（叠加到当前进程环境上）。
        startup_timeout: 等待子进程启动的超时秒数。
        call_timeout: 单次 RPC 调用的超时秒数。
        _allow_paths: 白名单路径前缀列表（用于安全审计，不做实际拦截）。
    """

    def __init__(
        self,
        name: str,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        startup_timeout: float = 15.0,
        call_timeout: float = 30.0,
        allow_paths: Optional[List[str]] = None,
    ):
        self.name = name
        self.command = command
        self.extra_env = env or {}
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout
        self._allow_paths = allow_paths or []

        self._proc: Optional[subprocess.Popen] = None
        self._req_id = 0
        self._lock = threading.Lock()
        self._started = False

    # ─── 生命周期 ──────────────────────────────────────────

    def start(self) -> "MCPConnector":
        """启动 MCP Server 子进程并完成 JSON-RPC initialize 握手。"""
        if self._started:
            return self

        import os

        env = os.environ.copy()
        env.update(self.extra_env)

        logger.info("[MCP:%s] 启动子进程: %s", self.name, " ".join(self.command))
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,           # 行缓冲
        )

        # 等待子进程就绪（简单探测：发 initialize 请求）
        deadline = time.time() + self.startup_timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                resp = self._rpc(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "TrinityGuard", "version": "1.0"},
                    },
                    timeout=5.0,
                )
                # 发送 initialized 通知（单向，无需等待响应）
                self._notify("notifications/initialized", {})
                logger.info("[MCP:%s] 握手成功: %s", self.name, resp.get("serverInfo", {}))
                self._started = True
                return self
            except Exception as exc:
                last_err = exc
                time.sleep(0.5)

        self._proc.kill()
        raise MCPError(
            f"[MCP:{self.name}] 启动超时 ({self.startup_timeout}s). 最后错误: {last_err}"
        )

    def stop(self):
        """终止 MCP Server 子进程。"""
        if self._proc and self._proc.poll() is None:
            logger.info("[MCP:%s] 终止子进程", self.name)
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ─── 对外 API ─────────────────────────────────────────

    def list_tools(self) -> List[Dict[str, Any]]:
        """返回 MCP Server 提供的工具列表（含 name、description、inputSchema）。"""
        resp = self._rpc("tools/list", {})
        return resp.get("tools", [])

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """调用 MCP 工具并返回结果内容。

        Args:
            tool_name: 工具名称（来自 list_tools）。
            arguments: 工具参数字典。

        Returns:
            工具返回内容（已解析为 Python 对象）。

        Raises:
            MCPError: 若调用失败或超时。
        """
        logger.debug("[MCP:%s] call_tool %s args=%s", self.name, tool_name, arguments)
        resp = self._rpc(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout=self.call_timeout,
        )

        # MCP 规范：content 是 List[{type, text}]
        content = resp.get("content", [])
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "\n".join(texts) if texts else content
        return content

    def is_alive(self) -> bool:
        """检查子进程是否仍在运行。"""
        return self._proc is not None and self._proc.poll() is None

    # ─── 内部 JSON-RPC ────────────────────────────────────

    def _next_id(self) -> int:
        with self._lock:
            self._req_id += 1
            return self._req_id

    def _send(self, message: dict):
        """向子进程 stdin 写入一行 JSON。"""
        line = json.dumps(message, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise MCPError(f"[MCP:{self.name}] 子进程管道已断开") from exc

    def _recv(self, timeout: float) -> dict:
        """从子进程 stdout 读取一行 JSON（带超时）。"""
        import queue

        result_q: queue.Queue = queue.Queue()

        def _reader():
            try:
                line = self._proc.stdout.readline()
                result_q.put(("ok", line))
            except Exception as exc:
                result_q.put(("err", exc))

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        try:
            status, payload = result_q.get(timeout=timeout)
        except queue.Empty:
            raise MCPError(f"[MCP:{self.name}] 读取超时 ({timeout}s)")

        if status == "err":
            raise MCPError(f"[MCP:{self.name}] 读取错误: {payload}") from payload

        line = payload.strip()
        if not line:
            raise MCPError(f"[MCP:{self.name}] 收到空响应")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise MCPError(f"[MCP:{self.name}] JSON 解析失败: {line!r}") from exc

    def _rpc(self, method: str, params: dict, timeout: Optional[float] = None) -> dict:
        """发送请求并等待响应（同步）。"""
        req_id = self._next_id()
        request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        timeout = timeout or self.call_timeout

        with self._lock:
            self._send(request)
            # 循环读取，忽略通知（无 id 字段）直到收到对应 id 的响应
            deadline = time.time() + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise MCPError(f"[MCP:{self.name}] RPC 超时: {method}")
                msg = self._recv(remaining)

                # 跳过 JSON-RPC 通知（服务端主动推送，无 id）
                if "id" not in msg:
                    continue
                if msg["id"] != req_id:
                    logger.debug("[MCP:%s] 忽略非预期响应 id=%s", self.name, msg.get("id"))
                    continue

                if "error" in msg:
                    err = msg["error"]
                    raise MCPError(
                        f"[MCP:{self.name}] RPC 错误 [{err.get('code')}]: {err.get('message')}"
                    )
                return msg.get("result", {})

    def _notify(self, method: str, params: dict):
        """发送单向通知（无需响应）。"""
        notification = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self._send(notification)
        except MCPError as exc:
            logger.warning("[MCP:%s] 通知发送失败: %s", self.name, exc)