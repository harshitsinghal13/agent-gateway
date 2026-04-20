"""
Add to .env: SERP_API_KEY=your_key_here
"""

import os
from serpapi import GoogleSearch
from agent.tools.base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web for current information on any topic. "
        "Returns top 3 results with title, URL, and snippet. "
        "Use this for news, facts, recent events, or any information you don't already know."
    )

    def __init__(self):
        self.api_key = os.getenv("SERP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SERP_API_KEY not set. "
                "Get a free key at serpapi.com and add it to .env"
            )

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    async def execute(self, query: str) -> str:
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": 3,              # top 3 results
            "hl": "en",            # language
            "gl": "us",            # country
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        organic = results.get("organic_results", [])

        if not organic:
            return "No results found."

        formatted = []
        for r in organic[:3]:
            formatted.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('link', 'N/A')}\n"
                f"Snippet: {r.get('snippet', 'N/A')}"
            )

        return "\n\n---\n\n".join(formatted)