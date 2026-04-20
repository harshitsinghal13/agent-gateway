"""
agent/tools/base.py

Every tool must extend BaseTool.
This enforces the interface the orchestrator and registry expect.

To create a new tool:
    class MyTool(BaseTool):
        name = "my_tool"
        description = "Does something useful"

        def definition(self) -> dict:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string", "description": "The input"}
                        },
                        "required": ["input"]
                    }
                }
            }

        async def execute(self, input: str) -> str:
            return f"Result for {input}"
"""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    Abstract base for all agent tools.

    Subclasses must define:
        name        — unique string key used by the LLM to call this tool
        description — shown to the LLM, should be clear and specific
        definition() — returns OpenAI function-calling schema
        execute()    — runs the tool, returns a string result
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def definition(self) -> dict:
        """
        Return the tool schema in OpenAI function-calling format.
        This is what the LLM sees when deciding which tool to call.
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """
        Run the tool with the arguments the LLM provided.
        Always returns a string — the orchestrator passes this
        back to the LLM as a tool result message.
        Raise an exception on failure — orchestrator catches it.
        """
        ...

    def __repr__(self):
        return f"<Tool: {self.name}>"
        