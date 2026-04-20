"""
providers/gemini.py

Gemini Flash provider for AgentGateway.
Mirrors the same interface as ollama.py.

Setup:
    pip install google-generativeai
    Set GEMINI_API_KEY in .env
    Get free key at: aistudio.google.com
"""

import time
import os
import google.generativeai as genai

from typing import AsyncGenerator
from  providers.schema import LLMResponse, ToolCall, ProviderHealth


# ── Tool format conversion ────────────────────────────────────────────────────
# Gemini uses a different tool schema than OpenAI format.
# Ollama and the agent orchestrator use OpenAI format internally.
# This converts before sending to Gemini.

def _to_gemini_tools(tools: list[dict]) -> list:
    """
    Convert OpenAI-format tool definitions to Gemini FunctionDeclaration format.

    OpenAI format (what agent uses internally):
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": { "query": { "type": "string" } },
                    "required": ["query"]
                }
            }
        }

    Gemini format:
        genai.protos.Tool(function_declarations=[
            genai.protos.FunctionDeclaration(
                name="web_search",
                description="Search the web",
                parameters=genai.protos.Schema(...)
            )
        ])
    """
    declarations = []
    for tool in tools:
        fn = tool.get("function", tool)
        declarations.append(
            genai.protos.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        k: genai.protos.Schema(
                            type=genai.protos.Type[v.get("type", "string").upper()],
                            description=v.get("description", ""),
                        )
                        for k, v in fn.get("parameters", {}).get("properties", {}).items()
                    },
                    required=fn.get("parameters", {}).get("required", []),
                )
            )
        )
    return [genai.protos.Tool(function_declarations=declarations)]


def _to_gemini_messages(messages: list[dict]) -> tuple[str | None, list]:
    """
    Convert OpenAI-format messages to Gemini format.

    Gemini separates system prompt from conversation history.
    Returns (system_instruction, gemini_history).

    OpenAI roles:  system / user / assistant / tool
    Gemini roles:  user / model
    """
    system_instruction = None
    history = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_instruction = content

        elif role == "user":
            history.append({"role": "user", "parts": [content]})

        elif role == "assistant":
            history.append({"role": "model", "parts": [content]})

        elif role == "tool":
            # tool results go back as user messages in Gemini
            history.append({
                "role": "user",
                "parts": [f"Tool result: {content}"]
            })

    return system_instruction, history


# ── Main provider class ───────────────────────────────────────────────────────

class GeminiProvider:
    """
    Wraps Google Gemini API for AgentGateway.
    Identical interface to OllamaProvider.

    Usage:
        provider = GeminiProvider()
        response = await provider.complete(messages, tools)
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        timeout: int = 60,
    ):
        self.model = model
        self.timeout = timeout
        self.provider_name = "gemini"

        # configure SDK — reads GEMINI_API_KEY from env if not passed directly
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Get a key at aistudio.google.com and add it to .env"
            )
        genai.configure(api_key=api_key)

    # ── Health check ──────────────────────────────────────────────────────────

    async def is_healthy(self) -> ProviderHealth:
        """
        Ping Gemini with a minimal request to verify the API key works.
        Go router uses this for automatic failover decisions.
        """
        start = time.monotonic()
        try:
            client = genai.GenerativeModel(self.model)
            client.generate_content("ping")
            latency_ms = (time.monotonic() - start) * 1000

            return ProviderHealth(
                provider=self.provider_name,
                healthy=True,
                model=self.model,
                latency_ms=round(latency_ms, 2),
            )

        except Exception as e:
            return ProviderHealth(
                provider=self.provider_name,
                healthy=False,
                model=self.model,
                error=str(e),
            )

    # ── Core completion ───────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse | ToolCall:
        """
        Send messages to Gemini, get a response back.

        Args:
            messages:    OpenAI-format message list
            tools:       OpenAI-format tool definitions (auto-converted to Gemini format)
            temperature: 0.0 = deterministic, 1.0 = creative
            max_tokens:  Max response length

        Returns:
            LLMResponse  — model replied with text
            ToolCall     — model wants to call a tool
        """

        system_instruction, history = _to_gemini_messages(messages)

        client = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_instruction,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            tools=_to_gemini_tools(tools) if tools else None,
        )

        # Gemini uses a chat session to maintain history
        chat = client.start_chat(history=history[:-1] if history else [])

        # send the last user message
        last_message = history[-1]["parts"][0] if history else ""
        response = chat.send_message(last_message)

        candidate = response.candidates[0]
        part = candidate.content.parts[0]

        # ── Tool call response ─────────────────────────────────────────────
        if hasattr(part, "function_call") and part.function_call.name:
            fc = part.function_call
            return ToolCall(
                name=fc.name,
                arguments=dict(fc.args),
                provider=self.provider_name,
            )

        # ── Text response ──────────────────────────────────────────────────
        # Gemini provides real token counts via usage_metadata
        usage = response.usage_metadata

        return LLMResponse(
            content=response.text,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            model=self.model,
            provider=self.provider_name,
        )

    # ── Streaming completion ──────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Gemini as they're generated.
        SSE handler iterates this and pushes each chunk to the client.

        Usage:
            async for chunk in provider.stream(messages):
                await sse_send(chunk)
        """

        system_instruction, history = _to_gemini_messages(messages)

        client = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_instruction,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        chat = client.start_chat(history=history[:-1] if history else [])
        last_message = history[-1]["parts"][0] if history else ""

        # stream=True returns a generator of response chunks
        response = chat.send_message(last_message, stream=True)

        for chunk in response:
            try:
                text = chunk.text
                if text:
                    yield text
            except Exception:
                # some chunks have no text (e.g. finish_reason only)
                continue

    # ── Token counting ────────────────────────────────────────────────────────

    async def count_tokens(self, text: str) -> int:
        """
        Use Gemini's real token counting API.
        More accurate than the ~4 chars/token heuristic used by ollama.py.
        Cost tracker calls this before each request.
        """
        client = genai.GenerativeModel(self.model)
        result = client.count_tokens(text)
        return result.total_tokens


# ── Quick test ────────────────────────────────────────────────────────────────

async def _test():
    """Run with: python providers/gemini.py"""
    from dotenv import load_dotenv
    load_dotenv()

    provider = GeminiProvider()

    print("Checking health...")
    health = await provider.is_healthy()
    print(f"Healthy: {health.healthy}")
    if not health.healthy:
        print(f"Error: {health.error}")
        return
    print(f"Latency: {health.latency_ms}ms")

    print("\nTesting completion...")
    messages = [{"role": "user", "content": "Say an essay on dog in 10 sentence."}]
    result = await provider.complete(messages)
    print(f"Response: {result.content}")
    print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")

    print("\nTesting token count...")
    count = await provider.count_tokens("Say a Quote in 2 sentence.")
    print(f"Token count: {count}")

    print("\nTesting streaming...")
    async for chunk in provider.stream(messages):
        print(chunk, end="", flush=True)
    print()

    print("\nTesting tool call...")
    tools = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }]
    messages_with_tool = [{"role": "user", "content": "Search for latest AI news"}]
    result = await provider.complete(messages_with_tool, tools=tools)
    if isinstance(result, ToolCall):
        print(f"Tool called: {result.name}")
        print(f"Arguments: {result.arguments}")
    else:
        print(f"Text response: {result.content}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_test())