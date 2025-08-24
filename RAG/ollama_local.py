import requests, os, json
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def call_groq_llm(prompt, model="gemma3:latest"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"stream": True}
        ]
    }
    response = requests.post(
        "http://localhost:11434/api/chat",
        headers=headers,
        json=payload
    )
    response.raise_for_status()

    parts = []
    for line in response.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue

        if "message" in obj and "content" in obj["message"]:
            parts.append(obj["message"]["content"])

        if obj.get("done", False):
            break

    return "".join(parts)


response = call_groq_llm("what's the diff between Capital.com and Interactive brokers")
print(response)