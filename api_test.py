import requests
import json
import os
from dotenv import load_dotenv

load_dotenv(".env.local")

API_KEY = os.getenv("OPENROUTER_API_KEY")
print(f"Using key: {API_KEY[:20]}...")

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'http://localhost:8501',
    'X-Title': 'Test'
}

payload = {
    'model': 'qwen/qwen3.6-plus:free',
    'messages': [{'role': 'user', 'content': 'Say hello in one sentence.'}]
}

resp = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, json=payload)
print(f'Status: {resp.status_code}')
print(json.dumps(resp.json(), indent=2))
