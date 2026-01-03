"""
Medical RAG Query Interface
Hỏi đáp với dữ liệu y tế đã được xử lý (100 files)
"""

import os
import asyncio
import torch
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, setup_logger
from dotenv import load_dotenv

# Setup logging
setup_logger("lightrag", level="INFO")

# ================================
# CONFIGURATION
# ================================
load_dotenv(dotenv_path=r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\.env")

MEGALLM_API_KEY = os.getenv("MEGALLM_API_KEY", "your-megallm-api-key-here")
MEGALLM_BASE_URL = os.getenv("MEGALLM_BASE_URL", "https://ai.megallm.io/v1")
MEGALLM_MODEL = os.getenv("MEGALLM_MODEL", "deepseek-ai/deepseek-v3.1")

# Thư mục chứa dữ liệu đã xử lý
WORKING_DIR = "./medical_rag"

# Hugging Face Embedding
HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_EMBEDDING_DIM = 384

# ================================
# Embedding Function
# ================================
_hf_tokenizer = None
_hf_embed_model = None

def load_hf_embedding_model():
    global _hf_tokenizer, _hf_embed_model
    if _hf_tokenizer is None:
        print(f"Loading embedding model: {HF_EMBEDDING_MODEL}")
        from transformers import AutoTokenizer, AutoModel
        _hf_tokenizer = AutoTokenizer.from_pretrained(HF_EMBEDDING_MODEL)
        _hf_embed_model = AutoModel.from_pretrained(HF_EMBEDDING_MODEL)
        if torch.cuda.is_available():
            _hf_embed_model = _hf_embed_model.cuda()
            print("   Using CUDA GPU")
        _hf_embed_model.eval()
    return _hf_tokenizer, _hf_embed_model

async def hf_embedding_func(texts: list[str]) -> np.ndarray:
    tokenizer, model = load_hf_embedding_model()
    device = next(model.parameters()).device
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    if embeddings.dtype == torch.bfloat16:
        return embeddings.to(torch.float32).cpu().numpy()
    return embeddings.cpu().numpy()

# ================================
# LLM Function
# ================================
async def megallm_complete(prompt: str, system_prompt: str = None, history_messages: list = [], **kwargs) -> str:
    return await openai_complete_if_cache(
        model=MEGALLM_MODEL, prompt=prompt, system_prompt=system_prompt,
        history_messages=history_messages, api_key=MEGALLM_API_KEY, base_url=MEGALLM_BASE_URL, **kwargs
    )

# ================================
# Initialize RAG
# ================================
async def initialize_rag():
    embedding_func = EmbeddingFunc(embedding_dim=HF_EMBEDDING_DIM, max_token_size=512, func=hf_embedding_func)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=megallm_complete,
        llm_model_name=MEGALLM_MODEL,
        embedding_func=embedding_func,
    )
    await rag.initialize_storages()
    return rag

# ================================
# Main
# ================================
async def main():
    print("="*60)
    print("Medical RAG - Query Interface")
    print("="*60)
    print(f"Data: {WORKING_DIR}")
    print(f"LLM: {MEGALLM_MODEL}")
    
    if MEGALLM_API_KEY == "your-megallm-api-key-here":
        print("\n[ERROR] Set MEGALLM_API_KEY in .env file!")
        return
    
    rag = None
    try:
        print("\nLoading RAG system...")
        load_hf_embedding_model()
        rag = await initialize_rag()
        print("RAG loaded successfully!\n")
        
        # Query modes
        print("Query modes: naive, local, global, hybrid, mix (recommended)")
        print("Type 'exit' to quit, 'mode:xxx' to change mode\n")
        
        current_mode = "mix"
        
        while True:
            try:
                query = input(f"[{current_mode}] Your question: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                if not query:
                    continue
                if query.startswith("mode:"):
                    current_mode = query.split(":")[1].strip()
                    print(f"   Mode changed to: {current_mode}")
                    continue
                
                print("\nSearching...")
                result = await rag.aquery(query, param=QueryParam(mode=current_mode))
                print(f"\nAnswer:\n{result}\n")
                print("-"*60)
                
            except KeyboardInterrupt:
                break
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            await rag.finalize_storages()
            print("\nRAG closed.")

if __name__ == "__main__":
    asyncio.run(main())
