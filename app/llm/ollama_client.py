import requests
import json
from typing import Optional


OLLAMA_BASE_URL = "http://localhost:11434"


# def generate(
#     prompt: str,
#     model: str = "mistral",
#     system: Optional[str] = None,
#     temperature: float = 0.7,
#     max_tokens: int = 1024,
# ) -> str:
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "stream": False,
#         "options": {
#             "temperature": temperature,
#             "num_predict": max_tokens,
#         },
#     }
#     if system:
#         payload["system"] = system

#     resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
#     resp.raise_for_status()
#     return resp.json()["response"].strip()


def generate(
    prompt: str,
    model: str = "mistral",
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if system:
        payload["system"] = system

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120,
    )

    resp.raise_for_status()

    data = resp.json()

    return {
        "text": data.get("response", "").strip(),
        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "response_tokens": data.get("eval_count", 0),
        "raw": data,
    }


def chat(
    messages: list[dict],
    model: str = "mistral",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def is_ollama_running() -> bool: # Health check up
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]
