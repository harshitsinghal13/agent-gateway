from dataclasses import dataclass, field
from typing import Literal

@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    provider: str

@dataclass
class ToolCall:
    name: str
    arguments: dict
    provider: str

@dataclass
class ToolResult:
    tool_name: str
    output: str
    success: bool = True
    error: str | None = None

@dataclass
class TraceEvent:
    event_type: Literal["thought", "tool_call", "tool_result", "final_answer"]
    content: str
    metadata: dict = field(default_factory=dict)

@dataclass
class ProviderHealth:
    provider: str
    healthy: bool
    model: str
    latency_ms: float | None = None
    error: str | None = None
