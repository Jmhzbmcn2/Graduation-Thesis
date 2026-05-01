"""
Qwen3-Reranker-0.6B Self-hosted Server (Cohere-compatible API)

Endpoint: POST /rerank
Compatible with LightRAG's `RERANK_BINDING=cohere`.

Usage:
    python scripts/rerank_server.py
    # Or via uvicorn:
    uvicorn scripts.rerank_server:app --host 0.0.0.0 --port 7997

Test:
    curl -X POST http://localhost:7997/rerank \\
        -H "Content-Type: application/json" \\
        -d '{"query": "thuốc giảm đau", "documents": ["Paracetamol", "Aspirin", "Vitamin C"], "top_n": 3}'
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Config ────────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B")
MAX_LENGTH = int(os.environ.get("RERANKER_MAX_LENGTH", "8192"))
BATCH_SIZE = int(os.environ.get("RERANKER_BATCH_SIZE", "8"))
DEVICE = os.environ.get("RERANKER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PORT = int(os.environ.get("RERANKER_PORT", "7997"))

DEFAULT_INSTRUCTION = (
    "Given a query, retrieve relevant medical passages that answer the query"
)
PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# ─── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rerank_server")


# ─── Module-level state (loaded on startup) ────────────────────────────────
class ModelState:
    tokenizer = None
    model = None
    token_true_id: int = -1
    token_false_id: int = -1
    prefix_tokens: list = []
    suffix_tokens: list = []


state = ModelState()


def _format_pair(query: str, doc: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


@torch.no_grad()
def _score_batch(query: str, docs: List[str], instruction: str) -> List[float]:
    """Return list of relevance scores in [0, 1] for given (query, docs)."""
    tokenizer = state.tokenizer
    model = state.model
    prefix_tokens = state.prefix_tokens
    suffix_tokens = state.suffix_tokens

    pairs = [_format_pair(query, d, instruction) for d in docs]

    # Tokenize without padding first; truncate body to fit MAX_LENGTH minus prefix+suffix
    body_max_len = MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens)
    enc = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=body_max_len,
    )
    # Wrap with prefix/suffix tokens
    for i in range(len(enc["input_ids"])):
        enc["input_ids"][i] = prefix_tokens + enc["input_ids"][i] + suffix_tokens

    # Pad batch (left-padding because tokenizer.padding_side='left')
    inputs = tokenizer.pad(
        enc, padding=True, return_tensors="pt", max_length=MAX_LENGTH
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Forward → take logits at last position
    logits = model(**inputs).logits[:, -1, :]
    yes_logits = logits[:, state.token_true_id]
    no_logits = logits[:, state.token_false_id]
    pair = torch.stack([no_logits, yes_logits], dim=1)
    probs = torch.nn.functional.log_softmax(pair, dim=1)
    scores = probs[:, 1].exp().tolist()
    return scores


def _load_model() -> None:
    logger.info(f"Loading reranker model: {MODEL_NAME}")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype
    ).to(DEVICE).eval()

    state.tokenizer = tokenizer
    state.model = model
    state.token_true_id = tokenizer.convert_tokens_to_ids("yes")
    state.token_false_id = tokenizer.convert_tokens_to_ids("no")
    state.prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    state.suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Model loaded in {elapsed:.1f}s on {DEVICE} (dtype={dtype}, "
        f"max_length={MAX_LENGTH}, batch_size={BATCH_SIZE}, "
        f"yes_id={state.token_true_id}, no_id={state.token_false_id})"
    )

    if DEVICE.startswith("cuda"):
        free_b, total_b = torch.cuda.mem_get_info()
        used_gb = (total_b - free_b) / 1024**3
        free_gb = free_b / 1024**3
        logger.info(f"GPU memory: used={used_gb:.2f} GB, free={free_gb:.2f} GB")


# ─── FastAPI app with lifespan ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    # Cleanup
    if state.model is not None:
        del state.model
        del state.tokenizer
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()


app = FastAPI(
    title="Qwen3-Reranker Server (Cohere-compatible)",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Pydantic schemas (Cohere format) ──────────────────────────────────────
class RerankRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Ignored; kept for compat")
    query: str
    documents: List[str]
    top_n: Optional[int] = Field(default=None, ge=1)
    instruction: Optional[str] = Field(
        default=None, description="Optional custom task instruction"
    )


class RerankResultItem(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: List[RerankResultItem]
    model: str
    usage: dict


# ─── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "loaded": state.model is not None,
    }


@app.post("/rerank", response_model=RerankResponse)
@app.post("/v2/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    if not req.documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    instruction = req.instruction or DEFAULT_INSTRUCTION
    t0 = time.perf_counter()

    all_scores: List[float] = []
    for i in range(0, len(req.documents), BATCH_SIZE):
        chunk = req.documents[i : i + BATCH_SIZE]
        scores = _score_batch(req.query, chunk, instruction)
        all_scores.extend(scores)

    indexed = [
        RerankResultItem(index=i, relevance_score=s)
        for i, s in enumerate(all_scores)
    ]
    indexed.sort(key=lambda x: x.relevance_score, reverse=True)

    if req.top_n is not None:
        indexed = indexed[: req.top_n]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"Reranked {len(req.documents)} docs in {elapsed_ms:.1f}ms "
        f"(top score={indexed[0].relevance_score:.4f})"
    )

    return RerankResponse(
        results=indexed,
        model=MODEL_NAME,
        usage={
            "num_documents": len(req.documents),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )


# ─── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "scripts.rerank_server:app" if __package__ else app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
