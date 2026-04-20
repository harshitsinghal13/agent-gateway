import requests
import json

url = "http://localhost:8000/chat"

payload = {
    "messages": [
        {"role": "user", "content": "Explain Agentic AI in simple terms."}
    ],
    "provider": "gemini"
}

headers = {
    "accept": "text/event-stream",   # important for SSE
    "Content-Type": "application/json"
}

with requests.post(url, json=payload, headers=headers, stream=True) as response:
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8")

            # SSE lines usually start with "data:"
            if decoded.startswith("data:"):
                data = decoded.replace("data:", "").strip()

                # sometimes servers send [DONE]
                if data == "[DONE]":
                    break

                try:
                    s = json.loads(data)
                    print(s.get("content", ""))
                except json.JSONDecodeError:
                    print(data)