"""
agent/orchestrator.py

ReAct loop orchestrator for AgentGateway.
Reason → Act → Observe → repeat until done.

The orchestrator:
  1. Receives a task (user message) and a provider
  2. Sends messages to the LLM with available tools
  3. If LLM returns a ToolCall → executes the tool → feeds result back
  4. If LLM returns LLMResponse → task is done, return final answer
  5. Streams a TraceEvent at every step via an async queue

Usage:
    orchestrator = Orchestrator(provider=GeminiProvider())
    async for event in orchestrator.run(task="summarize AI news"):
        print(event)
"""

import json
import asyncio
from providers.schema import LLMResponse, ToolCall, ToolResult, TraceEvent
from agent.tool_registry import ToolRegistry


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ITERATIONS = 10       # hard cap — prevents infinite loops
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
To complete tasks, reason step by step and use tools when needed.
When you have enough information, provide a final answer directly without calling any more tools.
/no_think"""


# ── Orchestrator ──────────────────────────────────────────────────────────────
"""
    Runs the ReAct loop for a given task.
    Args:
        provider:  Any provider implementing .complete() — Gemini or Ollama
        registry:  ToolRegistry instance with available tools registered
"""
class Orchestrator:
    def __init__(self, provider, registry: ToolRegistry | None = None):
        self.provider = provider
        self.registry = registry or ToolRegistry()

    async def run(self, task: str, session_id: str = "default"):
        """
        Run the ReAct loop for a task.
        Yields TraceEvent at every step — thought, tool_call, tool_result, final_answer.

        Args:
            task:        The user's task as a plain string
            session_id:  Used for Redis session tracking later

        Yields:
            TraceEvent — one per reasoning/action step

        Usage:
            async for event in orchestrator.run("find AI news"):
                print(event.event_type, event.content)
        """

        # ── Build initial message history ──────────────────────────────────
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": task},
        ]

        # get tool definitions in OpenAI format for the LLM
        tool_definitions = self.registry.get_definitions()

        iteration = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1

            # ── Step 1: Ask the LLM what to do next ───────────────────────
            yield TraceEvent(
                event_type="thought",
                content=f"Thinking... (step {iteration})",
                metadata={"iteration": iteration}
            )

            response = await self.provider.complete(
                messages=messages,
                tools=tool_definitions if tool_definitions else None,
            )

            # ── Step 2: LLM wants to call a tool ──────────────────────────
            if isinstance(response, ToolCall):
                yield TraceEvent(
                    event_type="tool_call",
                    content=f"Calling {response.name}({json.dumps(response.arguments)})",
                    metadata={
                        "tool": response.name,
                        "arguments": response.arguments,
                        "provider": response.provider,
                    }
                )

                # execute the tool
                tool_result = await self._execute_tool(response)

                yield TraceEvent(
                    event_type="tool_result",
                    content=tool_result.output if tool_result.success else f"Error: {tool_result.error}",
                    metadata={
                        "tool": tool_result.tool_name,
                        "success": tool_result.success,
                    }
                )

                # add tool call + result to message history so LLM has context
                messages.append({
                    "role": "assistant",
                    "content": f"I'll call {response.name} with {json.dumps(response.arguments)}"
                })
                messages.append({
                    "role": "tool",
                    "content": tool_result.output if tool_result.success else f"Error: {tool_result.error}"
                })

                # loop back — LLM will reason again with the tool result
                continue

            # ── Step 3: LLM returned a final answer ───────────────────────
            if isinstance(response, LLMResponse):
                # check if model is describing a tool call instead of making one
                if response.content.startswith("I'll call") or "function_call" in response.content:
                    # treat as thought, not final answer — loop again
                    messages.append({"role": "assistant", "content": response.content})
                    continue
                yield TraceEvent(
                    event_type="final_answer",
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "model": response.model,
                        "provider": response.provider,
                        "iterations": iteration,
                    }
                )
                return

        # ── Hit max iterations without a final answer ──────────────────────
        yield TraceEvent(
            event_type="final_answer",
            content="Max iterations reached without a final answer. Please try a simpler task.",
            metadata={"iterations": iteration, "hit_limit": True}
        )

    """
    method to execute a tool call and return the result.
    """
    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        
        tool = self.registry.get(tool_call.name)

        if not tool:
            return ToolResult(
                tool_name=tool_call.name,
                output="",
                success=False,
                error=f"Tool '{tool_call.name}' not found in registry. Available: {self.registry.list_names()}"
            )

        try:
            output = await tool.execute(**tool_call.arguments)
            return ToolResult(
                tool_name=tool_call.name,
                output=output,
                success=True,
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_call.name,
                output="",
                success=False,
                error=str(e)
            )