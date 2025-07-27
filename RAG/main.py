import requests, os
from search import get_top_k_chunks
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def call_groq_llm(prompt, model="llama3-70b-8192"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )
    return response.json()["choices"][0]["message"]["content"]



query = "what's the company's name and what service do they render?"
context_chunks = get_top_k_chunks(query)
context = "\n\n".join(context_chunks)

full_prompt = f"""Based on the following document context, answer the user's question.

Context:
{context}

Question: {query}
"""

response = call_groq_llm(full_prompt)
print(response)

# print("\n\n\n",context_chunks)
