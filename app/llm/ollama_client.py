import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def query_mistral(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]