"""
agent/tool_registry.py

Central registry for all agent tools.
The orchestrator calls get_definitions() to get tool schemas for the LLM,
and get(name) to execute a specific tool.

Adding a new tool:
    1. Create agent/tools/your_tool.py with a class extending BaseTool
    2. Register it here in default_registry()
    That's it — the orchestrator picks it up automatically.
"""

from agent.tools.base import BaseTool


class ToolRegistry:
    """
    Holds all available tools.
    Orchestrator uses this to:
      - Pass tool definitions to the LLM (get_definitions)
      - Execute a tool the LLM chose (get + execute)
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_definitions(self) -> list[dict]:
        """
        Return all tool schemas in OpenAI function-calling format.
        Passed directly to provider.complete(tools=...).
        Gemini provider converts these to its own format internally.
        """
        return [tool.definition() for tool in self._tools.values()]


def default_registry() -> ToolRegistry:
    """
    Build and return the default registry with all MVP tools registered.
    Import and call this in main.py to get a ready-to-use registry.

    Usage:
        from agent.tool_registry import default_registry
        registry = default_registry()
        orchestrator = Orchestrator(provider=provider, registry=registry)
    """
    from agent.tools.web_search import WebSearchTool
    from agent.tools.db_query import DBQueryTool
    from agent.tools.summarizer import SummarizerTool

    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(DBQueryTool())
    registry.register(SummarizerTool())
    return registry