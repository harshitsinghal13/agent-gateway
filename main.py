"""
main.py

FastAPI app — entry point for AgentGateway agent layer.
Wires providers, registry, orchestrator, and SSE handler together.

Run:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /agent/run         — run agent task, stream trace via SSE
    POST /chat              — simple LLM completion, stream tokens via SSE
    GET  /health            — provider health check
"""

import os
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import jwt
from providers.gemini import GeminiProvider
from providers.ollama import OllamaProvider
from agent.tool_registry import default_registry
from agent.orchestrator import Orchestrator
from streaming.sse_handler import stream_agent, stream_completion
from fastapi.staticfiles import StaticFiles
from typing import Annotated

load_dotenv()

app = FastAPI(title="AgentGateway", version="0.1.0")
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

# allow all origins for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Providers ─────────────────────────────────────────────────────────────────
# Gemini is primary, Ollama is fallback.
# For local dev just use Gemini — it's faster.

gemini = GeminiProvider()
ollama = OllamaProvider(model="qwen3:4b")

# ── Tool registry ─────────────────────────────────────────────────────────────
# Pass gemini provider to summarizer so it can call the LLM internally

from agent.tools.summarizer import SummarizerTool
from agent.tools.web_search import WebSearchTool
from agent.tools.db_query import DBQueryTool
from agent.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register(WebSearchTool())
registry.register(DBQueryTool())
registry.register(SummarizerTool(provider=gemini))   # inject provider

# ── Request models ────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    task: str
    provider: str = "gemini"     # "gemini" or "ollama"

class ChatRequest(BaseModel):
    messages: list[dict]
    provider: str = "gemini"


# ── Helper ────────────────────────────────────────────────────────────────────

def get_provider(name: str):
    if name == "ollama":
        return ollama
    return gemini

def tool_registry_for_token(tools: list[str]) -> ToolRegistry:
    """Build a custom registry based on token permissions."""
    registry = ToolRegistry()
    if "web_search" in tools:
        registry.register(WebSearchTool())
    if "db_query" in tools:
        registry.register(DBQueryTool())
    if "summarizer" in tools:
        registry.register(SummarizerTool(provider=gemini))
    return registry


# -----------------Routes ------------------------------------------------

def validate_token(token: Annotated[str, Header()]) -> dict:
    try:
        decoded_token = jwt.decode(
            token,
            os.getenv("JWT_SECRET"),
            algorithms=["HS256"]
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {
        "providers": decoded_token.get("providers"),
        "tools": decoded_token.get("tools"),
    }

@app.post("/agent/run")
async def run_agent(
    request: AgentRequest,
    token_data: Annotated[dict, Depends(validate_token)],
):
    providers = token_data["providers"]
    tools = token_data["tools"]
    """
    Run an agentic task and stream trace events via SSE.

    Request:
        { "task": "Search for AI news and summarize", "provider": "gemini" }

    Response (SSE stream):
        data: {"type": "thought", "content": "...", "metadata": {}}
        data: {"type": "tool_call", "content": "...", "metadata": {...}}
        data: {"type": "tool_result", "content": "...", "metadata": {...}}
        data: {"type": "final_answer", "content": "...", "metadata": {...}}
        data: {"type": "done"}
    """

    if request.provider not in providers:
        raise HTTPException(status_code=403, detail="Provider access denied")
    provider = get_provider(request.provider)
    
    if tools:
        registry = tool_registry_for_token(tools)
    else:
        registry = default_registry()
    orchestrator = Orchestrator(provider=provider, registry=registry)
    return await stream_agent(orchestrator, task=request.task)


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Simple LLM completion — no agent, no tools.
    Streams raw tokens via SSE.

    Request:
        { "messages": [{"role": "user", "content": "Hello"}] }

    Response (SSE stream):
        data: {"type": "token", "content": "Hello"}
        data: {"type": "token", "content": " there"}
        data: {"type": "done"}
    """
    provider = get_provider(request.provider)
    return await stream_completion(provider, messages=request.messages)


@app.get("/health")
async def health():
    """Check health of all providers."""
    gemini_health = await gemini.is_healthy()
    ollama_health = await ollama.is_healthy()
    return {
        "gemini": {
            "healthy": gemini_health.healthy,
            "latency_ms": gemini_health.latency_ms,
            "error": gemini_health.error,
        },
        "ollama": {
            "healthy": ollama_health.healthy,
            "latency_ms": ollama_health.latency_ms,
            "error": ollama_health.error,
        }
    }


@app.get("/")
async def root():
    return {
        "name": "AgentGateway",
        "version": "0.1.0",
        "endpoints": ["/agent/run", "/chat", "/health"]
    }