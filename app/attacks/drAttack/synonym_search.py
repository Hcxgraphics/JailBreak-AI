from app.llm.ollama_client import query_mistral

def generate_synonym(word):

    prompt = f"""
Give one better suitable contextual synonym for:

{word}

Only return the synonym.
Do not include any explanations or additional text (like "The synonym for {word} is ..."). Just return the synonym itself.
"""

    response = query_mistral(prompt)

    return response.strip()