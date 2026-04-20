import time
import httpx
import json
from typing import AsyncGenerator

from providers.schema import LLMResponse, ToolCall, ProviderHealth


class OllamaProvider:

    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434", timeout: int = 120):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.provider_name = "ollama"

    # ── Health check ──────────────────────────────────────────────────────────

    async def is_healthy(self) -> ProviderHealth:
        """
        Ping Ollama and return a ProviderHealth.
        """
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                latency_ms = (time.monotonic() - start) * 1000

                if response.status_code != 200:
                    return ProviderHealth(
                        provider=self.provider_name,
                        healthy=False,
                        model=self.model,
                        error=f"HTTP {response.status_code}",
                    )

                data = response.json()
                available = [m["name"] for m in data.get("models", [])]
                model_found = any(self.model in m for m in available)
                print(self.model, available, model_found)
                if not model_found:
                    return ProviderHealth(
                        provider=self.provider_name,
                        healthy=False,
                        model=self.model,
                        error=f"Model '{self.model}' not pulled. Run: ollama pull {self.model}",
                    )

                return ProviderHealth(
                    provider=self.provider_name,
                    healthy=True,
                    model=self.model,
                    latency_ms=round(latency_ms, 2),
                )

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            return ProviderHealth(
                provider=self.provider_name,
                healthy=False,
                model=self.model,
                error=f"Ollama unreachable: {str(e)}. Run: ollama serve",
            )

    # ── List available models ─────────────────────────────────────────────────

    async def list_models(self) -> list[str]:
        """Return all models pulled locally in Ollama."""
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]

    # ── Core completion ───────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse | ToolCall:
        """
        Send messages to Ollama, get a response back.

        Args:
            messages:    OpenAI-format message list
                         [{"role": "user", "content": "..."}]
            tools:       Tool definitions in OpenAI format (optional).
                         Requires llama3.1+ or mistral-nemo for tool calling.
            temperature: 0.0 = deterministic, 1.0 = creative
            max_tokens:  Max response length

        Returns:
            LLMResponse  — model replied with text
            ToolCall     — model wants to call a tool

        Mirror this exact signature in gemini.py.
        """

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        message = data.get("message", {})

        # ── Tool call response ─────────────────────────────────────────────
        if message.get("tool_calls"):
            tool = message["tool_calls"][0]
            return ToolCall(
                name=tool["function"]["name"],
                arguments=tool["function"]["arguments"],
                provider=self.provider_name,
            )

        # ── Text response ──────────────────────────────────────────────────
        return LLMResponse(
            content=message.get("content", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            model=self.model,
            provider=self.provider_name,
        )

    # ── Streaming completion ──────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Ollama as they're generated.
        SSE handler iterates this and pushes each chunk to the client.

        Usage:
            async for chunk in provider.stream(messages):
                await sse_send(chunk)

        Mirror this exact signature in gemini.py.
        """

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content

                    if chunk.get("done"):
                        break

    # ── Token counting ────────────────────────────────────────────────────────

    async def count_tokens(self, text: str) -> int:
        """
        Rough token estimate for Ollama models.
        ~4 chars per token is the standard LLaMA heuristic.
        gemini.py can use Gemini's real count_tokens API instead.
        """
        return len(text) // 4


# ── Quick test ────────────────────────────────────────────────────────────────

async def _test():
    """Run with: python providers/ollama.py"""
    provider = OllamaProvider(model="qwen3:4b")

    print("Checking health...")
    health = await provider.is_healthy()
    print(f"Healthy: {health.healthy}")
    if not health.healthy:
        print(f"Error: {health.error}")
        return
    print(f"Latency: {health.latency_ms}ms")

    print("\nListing models...")
    models = await provider.list_models()
    print(f"Available: {models}")

    print("\nTesting completion...")
    messages = [{"role": "user", "content": "Say hello in 2 sentence."}]
    #result = await provider.complete(messages)
    #print(f"Response: {result.content}")
    #print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")

    print("\nTesting streaming...")
    async for chunk in provider.stream(messages):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(_test())