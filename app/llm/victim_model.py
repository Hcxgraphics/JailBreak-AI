from llm.ollama_client import query_mistral
def run_victim_model(prompt):
    response = query_mistral(prompt)
    return response
