import requests
import json

r = requests.post('http://localhost:11434/api/embed', json={
    'model': 'nomic-embed-text', 
    'input': 'test'
})
data = r.json()
embeddings = data.get('embeddings', [[]])
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"Response keys: {data.keys()}")
