import requests
import os
from dotenv import load_dotenv

load_dotenv(".env.local")
key = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "qwen/qwen3.6-plus:free",
    "messages": [
        {"role": "user", "content": "Hello. Analyze this data summary: Columns: ['Date', 'Sales'] | Stats: count 10, mean 500"}
    ]
}

resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
print(f"HTTP STATUS: {resp.status_code}")
if resp.status_code == 200:
    print("SUCCESS: Data received from Qwen.")
    print(resp.json()["choices"][0]["message"]["content"][:100] + "...")
else:
    print(f"FAILURE: {resp.text}")
