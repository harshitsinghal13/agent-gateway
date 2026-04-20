"""
agent/tools/summarizer.py

Summarizes long text using the LLM.
Injected with provider at runtime via SummarizerTool(provider=...).
"""

from agent.tools.base import BaseTool


class SummarizerTool(BaseTool):
    name = "summarizer"
    description = "Summarizes long text into key points. Use this when you have large amounts of text that need to be condensed."

    def __init__(self, provider=None):
        # provider is injected so summarizer can call the LLM
        # if None, falls back to simple truncation
        self.provider = provider

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to summarize"
                        },
                        "focus": {
                            "type": "string",
                            "description": "Optional focus area for the summary e.g. 'key findings' or 'action items'"
                        }
                    },
                    "required": ["text"]
                }
            }
        }

    async def execute(self, text: str, focus: str = "key points") -> str:
        # truncate very long text to avoid token overflow
        if len(text) > 6000:
            text = text[:6000] + "... [truncated]"

        if not self.provider:
            # fallback — return first 500 chars if no provider injected
            return f"Summary (no provider): {text[:500]}..."

        messages = [
            {
                "role": "system",
                "content": "You are a precise summarizer. Return only the summary, no preamble."
            },
            {
                "role": "user",
                "content": f"Summarize the following text focusing on {focus}:\n\n{text}"
            }
        ]

        response = await self.provider.complete(messages)
        return response.content