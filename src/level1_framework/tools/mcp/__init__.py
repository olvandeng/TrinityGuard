"""MCP (Model Context Protocol) tool support for TrinityGuard.

This package provides:
- MCPConnector: manages stdio subprocess for any MCP server
- MCPToolRegistry: builds AG2-compatible tool functions from MCP tool schemas
- attach_mcp_tools: one-call helper that wires MCP tools onto an AG2 agent
"""

from .connector import MCPConnector
from .registry import MCPToolRegistry, attach_mcp_tools

__all__ = ["MCPConnector", "MCPToolRegistry", "attach_mcp_tools"]