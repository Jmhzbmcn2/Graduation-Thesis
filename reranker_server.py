"""
Self-hosted Reranker Server using BGE-Reranker-v2-m3
Compatible with Cohere API format for LightRAG integration
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from sentence_transformers import CrossEncoder
import torch

app = FastAPI(title="BGE Reranker Server")

# Load model on startup
print("Loading BGE Reranker model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    max_length=8192,
    device=device
)
print("Model loaded successfully!")


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    model: Optional[str] = None


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]


@app.post("/rerank", response_model=RerankResponse)
@app.post("/v1/rerank", response_model=RerankResponse)
@app.post("/v2/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """Rerank documents based on query relevance"""
    
    # Create query-document pairs
    pairs = [[request.query, doc] for doc in request.documents]
    
    # Get scores from model
    scores = model.predict(pairs)
    
    # Create results with index and score
    results = [
        {"index": i, "relevance_score": float(score)}
        for i, score in enumerate(scores)
    ]
    
    # Sort by relevance score (descending)
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Apply top_n limit if specified
    if request.top_n and request.top_n > 0:
        results = results[:request.top_n]
    
    return {"results": results}


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "BAAI/bge-reranker-v2-m3", "device": device}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7997)
