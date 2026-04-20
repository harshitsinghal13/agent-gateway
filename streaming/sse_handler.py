"""
streaming/sse_handler.py

Converts agent TraceEvents and LLM token chunks into SSE format
and streams them to the client via FastAPI StreamingResponse.

SSE format:
    data: <json payload>\n\n

Client reads this with EventSource or fetch + ReadableStream.
"""

import json
import asyncio
from fastapi.responses import StreamingResponse
from providers.schema import TraceEvent


def _event(data: dict) -> str:
    """Format a dict as an SSE event string."""
    return f"data: {json.dumps(data)}\n\n"


def _ping() -> str:
    """Keep-alive ping — prevents connection timeout on slow agents."""
    return ": ping\n\n"


async def stream_agent(orchestrator, task: str) -> StreamingResponse:
    """
    Run the agent and stream every TraceEvent to the client via SSE.

    Usage in FastAPI route:
        return await stream_agent(orchestrator, task="find AI news")

    Client receives a stream of SSE events:
        data: {"type": "thought", "content": "Thinking...", "metadata": {}}
        data: {"type": "tool_call", "content": "Calling web_search...", "metadata": {...}}
        data: {"type": "tool_result", "content": "Found results...", "metadata": {...}}
        data: {"type": "final_answer", "content": "Here is...", "metadata": {...}}
        data: {"type": "done"}
    """

    async def generator():
        try:
            async for event in orchestrator.run(task=task):
                yield _event({
                    "type": event.event_type,
                    "content": event.content,
                    "metadata": event.metadata,
                })

                # small yield to flush buffer immediately
                await asyncio.sleep(0)

            # signal completion to client
            yield _event({"type": "done"})

        except Exception as e:
            yield _event({"type": "error", "content": str(e)})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


async def stream_completion(provider, messages: list[dict]) -> StreamingResponse:
    """
    Stream raw LLM token chunks directly — no agent, no tools.
    Used by the /v1/chat/completions endpoint for simple completions.

    Client receives:
        data: {"type": "token", "content": "Hello"}
        data: {"type": "token", "content": " world"}
        data: {"type": "done"}
    """

    async def generator():
        try:
            async for chunk in provider.stream(messages):
                yield _event({"type": "token", "content": chunk})
                await asyncio.sleep(0)

            yield _event({"type": "done"})

        except Exception as e:
            yield _event({"type": "error", "content": str(e)})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )