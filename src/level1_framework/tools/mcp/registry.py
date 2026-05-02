"""MCPToolRegistry — 将 MCP 工具 schema 转换为 AG2 兼容的 Python 函数。

核心能力：
- 从 MCPConnector.list_tools() 获取工具 schema
- 动态生成符合 AG2 函数签名要求的 Python 函数
- 通过 register_for_llm / register_for_execution 将工具注册到 AG2 agent
- 提供工具调用审计钩子（call_audit_callback），用于 TrinityGuard 安全监控
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# 工具调用审计回调类型：(tool_name, arguments, result, connector_name) -> None
AuditCallback = Callable[[str, Dict[str, Any], Any, str], None]

# 全局审计回调列表（可多个 monitor 注册）
_global_audit_callbacks: List[AuditCallback] = []


def register_audit_callback(callback: AuditCallback):
    """注册全局 MCP 工具调用审计回调。

    每次 MCP 工具被调用时，回调会被触发，接收参数：
      - tool_name: 工具名称
      - arguments: 调用参数
      - result: 工具返回结果
      - connector_name: 连接器名称（标识哪个 MCP Server）

    用于 MCPToolCallMonitor 实时监控工具调用行为。
    """
    _global_audit_callbacks.append(callback)


def unregister_audit_callback(callback: AuditCallback):
    """注销已注册的审计回调。"""
    if callback in _global_audit_callbacks:
        _global_audit_callbacks.remove(callback)


def _fire_audit(tool_name: str, arguments: Dict, result: Any, connector_name: str):
    """触发所有已注册的审计回调。"""
    for cb in _global_audit_callbacks:
        try:
            cb(tool_name, arguments, result, connector_name)
        except Exception as exc:
            logger.warning("[MCP Audit] 回调异常: %s", exc)


class MCPToolRegistry:
    """将 MCPConnector 暴露的工具注册到 AG2 agent 上。

    使用示例::

        connector = MCPConnector("filesystem", [...]).start()
        registry = MCPToolRegistry(connector)

        # 注册到 AG2 agent（assistant 负责决策调用，executor 负责执行）
        registry.attach_to_agents(assistant_agent, executor_agent)
    """

    def __init__(self, connector):
        """
        Args:
            connector: 已 start() 的 MCPConnector 实例。
        """
        self.connector = connector
        self._tools: List[Dict[str, Any]] = []

    def fetch_tools(self) -> List[Dict[str, Any]]:
        """从 MCP Server 获取工具列表并缓存。"""
        self._tools = self.connector.list_tools()
        logger.info(
            "[MCPToolRegistry:%s] 发现 %d 个工具: %s",
            self.connector.name,
            len(self._tools),
            [t["name"] for t in self._tools],
        )
        return self._tools

    def attach_to_agents(self, assistant_agent, executor_agent=None):
        """将所有 MCP 工具注册到 AG2 agent。

        Args:
            assistant_agent: AG2 ConversableAgent，负责 LLM 侧工具声明（register_for_llm）。
            executor_agent: AG2 ConversableAgent，负责执行侧（register_for_execution）。
                            若为 None，则 assistant 同时承担执行角色。
        """
        if not self._tools:
            self.fetch_tools()

        exec_agent = executor_agent or assistant_agent

        for tool_schema in self._tools:
            tool_name = tool_schema["name"]
            description = tool_schema.get("description", f"MCP tool: {tool_name}")

            # 生成执行函数
            func = self._make_tool_function(tool_name)

            # 为函数添加 docstring（AG2 用 docstring 作为工具描述）
            func.__doc__ = description
            func.__name__ = tool_name

            # AG2 工具注册：assistant 知道该工具（用于 LLM function-calling schema）
            try:
                assistant_agent.register_for_llm(
                    name=tool_name, description=description
                )(func)
            except Exception as exc:
                logger.warning(
                    "[MCPToolRegistry] register_for_llm 失败 tool=%s: %s", tool_name, exc
                )

            # AG2 工具注册：executor 实际执行该函数
            try:
                exec_agent.register_for_execution(name=tool_name)(func)
            except Exception as exc:
                logger.warning(
                    "[MCPToolRegistry] register_for_execution 失败 tool=%s: %s", tool_name, exc
                )

            logger.debug("[MCPToolRegistry:%s] 已注册工具: %s", self.connector.name, tool_name)

    def get_tool_names(self) -> List[str]:
        """返回已注册的工具名称列表。"""
        return [t["name"] for t in self._tools]

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回完整工具 schema 列表（用于 Judge 分析）。"""
        return list(self._tools)

    # ─── 内部方法 ─────────────────────────────────────────

    def _make_tool_function(self, tool_name: str) -> Callable:
        """动态创建对应 tool_name 的可调用函数。

        生成的函数接受 **kwargs 形式的参数，调用 MCPConnector.call_tool()，
        并触发全局审计回调。
        """
        connector = self.connector
        connector_name = connector.name

        def mcp_tool_func(**kwargs) -> str:
            """由 MCPToolRegistry 动态生成的 MCP 工具调用函数."""
            logger.info(
                "[MCP Tool] 调用 %s/%s args=%s",
                connector_name, tool_name, kwargs,
            )
            try:
                result = connector.call_tool(tool_name, kwargs)
                result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            except Exception as exc:
                result_str = f"ERROR: {exc}"
                logger.error(
                    "[MCP Tool] 调用失败 %s/%s: %s", connector_name, tool_name, exc
                )

            # 触发安全审计回调
            _fire_audit(tool_name, kwargs, result_str, connector_name)
            return result_str

        return mcp_tool_func


# ─── 便捷函数 ──────────────────────────────────────────────

def attach_mcp_tools(
    connector,
    assistant_agent,
    executor_agent=None,
    audit_callback: Optional[AuditCallback] = None,
) -> MCPToolRegistry:
    """一键将 MCP Server 的工具挂载到 AG2 agent。

    Args:
        connector: 已 start() 的 MCPConnector。
        assistant_agent: LLM 决策 agent。
        executor_agent: 执行 agent（可选，默认同 assistant）。
        audit_callback: 可选的调用审计回调，注册后对所有工具调用生效。

    Returns:
        已配置的 MCPToolRegistry 实例。
    """
    if audit_callback:
        register_audit_callback(audit_callback)

    registry = MCPToolRegistry(connector)
    registry.attach_to_agents(assistant_agent, executor_agent)
    return registry