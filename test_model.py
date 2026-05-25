from app.llm.ollama_client import query_mistral

response = query_mistral(
    "What is machine learning?"
)

print(response)