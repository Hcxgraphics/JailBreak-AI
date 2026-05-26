from app.llm.ollama_client import generate

query = "what is AI deployment?"
model = "mistral"

response = generate(prompt=query, model=model, temperature=0.2)

print(response)