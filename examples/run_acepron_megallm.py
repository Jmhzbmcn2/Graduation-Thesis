"""
LightRAG Demo with MegaLLM API (DeepSeek V3.1) + Hugging Face Embedding
Sử dụng với file dữ liệu acepron.txt
"""

import os
import asyncio
import torch
import numpy as np
from functools import partial
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, setup_logger

# Setup logging
setup_logger("lightrag", level="INFO")

# ================================
# CẤU HÌNH - ĐỌC TỪ BIẾN MÔI TRƯỜNG HOẶC .env
# ================================
from dotenv import load_dotenv

# Load .env file từ thư mục gốc dự án
load_dotenv(dotenv_path=r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\.env")

# MegaLLM Configuration
MEGALLM_API_KEY = os.getenv("MEGALLM_API_KEY", "your-megallm-api-key-here")
MEGALLM_BASE_URL = os.getenv("MEGALLM_BASE_URL", "https://ai.megallm.io/v1")
MEGALLM_MODEL = os.getenv("MEGALLM_MODEL", "deepseek-ai/deepseek-v3.1")

# Hugging Face Embedding Model - Đa ngôn ngữ (hỗ trợ tiếng Việt tốt)
HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_EMBEDDING_DIM = 384  # Model này có 384 dimensions

# Thư mục lưu dữ liệu RAG
WORKING_DIR = "./acepron_rag"

# Đường dẫn file dữ liệu
DATA_FILE = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\data\acepron.txt"

# ================================
# Hugging Face Embedding Setup
# ================================
# Global variables cho model (để tránh load lại nhiều lần)
_hf_tokenizer = None
_hf_embed_model = None


def load_hf_embedding_model():
    """Load Hugging Face embedding model (chỉ load 1 lần)"""
    global _hf_tokenizer, _hf_embed_model
    
    if _hf_tokenizer is None or _hf_embed_model is None:
        print(f"📥 Loading Hugging Face embedding model: {HF_EMBEDDING_MODEL}")
        print("   (Lần đầu chạy sẽ download model, có thể mất vài phút...)")
        
        from transformers import AutoTokenizer, AutoModel
        
        _hf_tokenizer = AutoTokenizer.from_pretrained(HF_EMBEDDING_MODEL)
        _hf_embed_model = AutoModel.from_pretrained(HF_EMBEDDING_MODEL)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _hf_embed_model = _hf_embed_model.cuda()
            print("   ✅ Using CUDA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _hf_embed_model = _hf_embed_model.to("mps")
            print("   ✅ Using Apple MPS")
        else:
            print("   ⚠️ Using CPU (slower)")
        
        _hf_embed_model.eval()
        print(f"   ✅ Model loaded successfully!")
    
    return _hf_tokenizer, _hf_embed_model


async def hf_embedding_func(texts: list[str]) -> np.ndarray:
    """
    Hugging Face embedding function
    """
    tokenizer, model = load_hf_embedding_model()
    
    # Determine device
    if torch.cuda.is_available():
        device = next(model.parameters()).device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Tokenize
    encoded = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**encoded)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    # Convert to numpy
    if embeddings.dtype == torch.bfloat16:
        return embeddings.to(torch.float32).cpu().numpy()
    return embeddings.cpu().numpy()


# ================================
# LLM Function - MegaLLM
# ================================
async def megallm_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs
) -> str:
    """
    LLM function sử dụng MegaLLM API (OpenAI-compatible)
    """
    return await openai_complete_if_cache(
        model=MEGALLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=MEGALLM_API_KEY,
        base_url=MEGALLM_BASE_URL,
        **kwargs
    )


# ================================
# Main Functions
# ================================
async def initialize_rag():
    """Khởi tạo LightRAG instance"""
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    
    # Tạo embedding function với Hugging Face
    embedding_func = EmbeddingFunc(
        embedding_dim=HF_EMBEDDING_DIM,
        max_token_size=512,
        func=hf_embedding_func,
    )
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=megallm_complete,
        llm_model_name=MEGALLM_MODEL,
        embedding_func=embedding_func,
        # Cấu hình tùy chọn
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=1,
    )
    
    await rag.initialize_storages()
    return rag


async def main():
    """Main function"""
    print("=" * 60)
    print("🚀 LightRAG with MegaLLM + Hugging Face Embedding")
    print("=" * 60)
    print(f"   LLM: {MEGALLM_MODEL}")
    print(f"   Embedding: {HF_EMBEDDING_MODEL} ({HF_EMBEDDING_DIM}d)")
    
    # Kiểm tra API key
    if MEGALLM_API_KEY == "your-megallm-api-key-here":
        print("\n❌ Error: Vui lòng cập nhật MEGALLM_API_KEY trong file script!")
        print("   Mở file và thay 'your-megallm-api-key-here' bằng API key thật.")
        return
    
    rag = None
    try:
        # Pre-load embedding model
        print("\n⏳ Loading embedding model...")
        load_hf_embedding_model()
        
        # Test embedding
        print("\n🔗 Testing embedding function...")
        test_emb = await hf_embedding_func(["Test embedding"])
        print(f"   ✅ Embedding shape: {test_emb.shape}")
        
        # Initialize RAG
        print("\n⏳ Initializing RAG...")
        rag = await initialize_rag()
        print("✅ RAG initialized successfully!")
        
        # Test LLM connection
        print("\n🔗 Testing LLM connection...")
        test_response = await megallm_complete("Hello, respond with just 'OK'")
        print(f"   ✅ LLM response: {test_response[:100]}...")

        # Đọc và insert dữ liệu
        print(f"\n📄 Loading data from: {DATA_FILE}")
        
        if not os.path.exists(DATA_FILE):
            print(f"❌ File không tồn tại: {DATA_FILE}")
            return
            
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        
        print(f"📝 Document size: {len(content)} characters")
        print("⏳ Inserting document (this may take a while for entity extraction)...")
        
        await rag.ainsert(content)
        print("✅ Document inserted successfully!")

        # Query examples
        print("\n" + "=" * 60)
        print("📋 QUERY EXAMPLES")
        print("=" * 60)
        
        queries = [
            "Acepron là gì?",
            "Công dụng của thuốc này là gì?",
            "Liều dùng như thế nào?",
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n{'─' * 50}")
            print(f"❓ Query {i}: {query}")
            print(f"{'─' * 50}")
            
            # Sử dụng mode "mix" để kết hợp KG và vector search
            result = await rag.aquery(
                query, 
                param=QueryParam(mode="mix")
            )
            print(f"📝 Answer:\n{result}")

        # Interactive mode
        print("\n" + "=" * 60)
        print("💬 INTERACTIVE MODE (type 'exit' to quit)")
        print("=" * 60)
        
        while True:
            try:
                user_query = input("\n❓ Your question: ").strip()
                if user_query.lower() in ['exit', 'quit', 'q']:
                    break
                if not user_query:
                    continue
                    
                result = await rag.aquery(
                    user_query,
                    param=QueryParam(mode="mix")
                )
                print(f"\n📝 Answer:\n{result}")
                
            except KeyboardInterrupt:
                break

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            await rag.finalize_storages()
            print("\n✅ Storage finalized!")


if __name__ == "__main__":
    asyncio.run(main())
    print("\n🎉 Done!")
