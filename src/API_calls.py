import requests

def call_deepseek(question, context, prompt, model, client):
    """Call DeepSeek API to generate an answer."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DeepSeek Error] Request failed: {e}")
        return ""

def call_ollama(question, context, prompt, model, api_url="http://localhost:11434/api/generate"):
    """Call Ollama API to generate an answer."""
    try:
        response = requests.post(api_url, json={
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "stream": False
        }, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[Ollama Error - Answer Generation] {e}")
        return ""