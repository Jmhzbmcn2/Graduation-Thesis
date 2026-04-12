"""
LightRAG Batch Processing with Checkpoint System
- Xóa dữ liệu cũ
- Xử lý tất cả files trong folder data/
- Lưu checkpoint để bỏ qua files đã xử lý
"""

import os
import asyncio
import json
import numpy as np
import ollama
from pathlib import Path
from datetime import datetime
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

# MegaLLM Configuration
MEGALLM_API_KEY = os.getenv("MEGALLM_API_KEY") or os.getenv("LLM_BINDING_API_KEY", "")
MEGALLM_BASE_URL = os.getenv("MEGALLM_BASE_URL") or os.getenv("LLM_BINDING_HOST", "https://ai.megallm.io/v1")
MEGALLM_MODEL = os.getenv("MEGALLM_MODEL") or os.getenv("LLM_MODEL", "deepseek-ai/deepseek-v3.1-terminus")

# Paths
DATA_DIR = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\data"
WORKING_DIR = "./medical_rag_ollama"
CHECKPOINT_FILE = "./medical_rag_ollama/checkpoint.json"

# Ollama Embedding Configuration
OLLAMA_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = "embeddinggemma:300m"
OLLAMA_EMBED_DIM = 768

# ================================
# Checkpoint Management
# ================================
def load_checkpoint():
    """Load danh sách files đã xử lý"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed_files": [], "last_update": None}

def save_checkpoint(checkpoint):
    """Lưu checkpoint"""
    checkpoint["last_update"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    print(f"   [Checkpoint] Saved: {len(checkpoint['processed_files'])} files processed")

def is_file_processed(filename, checkpoint):
    """Kiểm tra file đã được xử lý chưa"""
    return filename in checkpoint["processed_files"]

def mark_file_processed(filename, checkpoint):
    """Đánh dấu file đã xử lý"""
    if filename not in checkpoint["processed_files"]:
        checkpoint["processed_files"].append(filename)
        save_checkpoint(checkpoint)

# ================================
# Delete Old Data
# ================================
def delete_old_data():
    """Xóa toàn bộ dữ liệu cũ"""
    import shutil
    
    if os.path.exists(WORKING_DIR):
        print(f"\n[DELETE] Removing old data: {WORKING_DIR}")
        shutil.rmtree(WORKING_DIR)
        print("   [OK] Old data deleted successfully!")
    
    # Tạo lại thư mục
    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f"   [OK] Created fresh directory: {WORKING_DIR}")

# ================================
# Ollama Embedding
# ================================
async def ollama_embedding_func(texts: list[str]) -> np.ndarray:
    """Ollama embedding function using embeddinggemma:300m (768d)"""
    client = ollama.AsyncClient(host=OLLAMA_HOST)
    try:
        response = await client.embed(model=OLLAMA_EMBED_MODEL, input=texts)
        return np.array(response["embeddings"])
    finally:
        try:
            await client._client.aclose()
        except:
            pass

def check_ollama_ready():
    """Check if Ollama is running and model is available"""
    try:
        response = ollama.list()
        models = [m.get('name', m.get('model', '')) for m in response.get('models', [])]
        if not any(OLLAMA_EMBED_MODEL in m for m in models):
            print(f"[!] Model {OLLAMA_EMBED_MODEL} not found. Pulling...")
            ollama.pull(OLLAMA_EMBED_MODEL)
        return True
    except Exception as e:
        print(f"[ERROR] Ollama not running: {e}")
        return False

# ================================
# LLM Function
# ================================
async def megallm_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs
) -> str:
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
# Main Processing
# ================================
async def initialize_rag():
    """Initialize LightRAG"""
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    embedding_func = EmbeddingFunc(
        embedding_dim=OLLAMA_EMBED_DIM,
        max_token_size=8192,
        func=ollama_embedding_func,
    )
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=megallm_complete,
        llm_model_name=MEGALLM_MODEL,
        embedding_func=embedding_func,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=1,
    )
    
    await rag.initialize_storages()
    return rag

async def process_files(rag, delete_old=False):
    """Xử lý tất cả files trong DATA_DIR"""
    
    # Xóa dữ liệu cũ nếu được yêu cầu
    if delete_old:
        delete_old_data()
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    
    # Lấy danh sách files
    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.txt')])
    total_files = len(all_files)
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING: {total_files} files")
    print(f"Already processed: {len(checkpoint['processed_files'])} files")
    print(f"{'='*60}")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx, filename in enumerate(all_files, 1):
        file_path = os.path.join(DATA_DIR, filename)
        
        # Kiểm tra checkpoint
        if is_file_processed(filename, checkpoint):
            skipped_count += 1
            print(f"[{idx}/{total_files}] SKIP (already processed): {filename}")
            continue
        
        print(f"\n[{idx}/{total_files}] Processing: {filename}")
        
        try:
            # Đọc file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"   Size: {len(content)} chars")
            
            # Insert vào RAG
            await rag.ainsert(content, file_paths=[filename])
            
            # Đánh dấu đã xử lý
            mark_file_processed(filename, checkpoint)
            processed_count += 1
            
            print(f"   [OK] Processed successfully!")
            
        except Exception as e:
            error_count += 1
            print(f"   [ERROR] {str(e)[:100]}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"   Total files: {total_files}")
    print(f"   Processed: {processed_count}")
    print(f"   Skipped: {skipped_count}")
    print(f"   Errors: {error_count}")
    
    return processed_count

async def main():
    """Main function"""
    print("="*60)
    print("LightRAG Batch Processing with Checkpoint")
    print("="*60)
    print(f"LLM: {MEGALLM_MODEL}")
    print(f"Embedding: {OLLAMA_EMBED_MODEL} via Ollama ({OLLAMA_EMBED_DIM}d)")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Working Dir: {WORKING_DIR}")
    
    # Kiểm tra API key
    if not MEGALLM_API_KEY:
        print("\n[ERROR] Please set MEGALLM_API_KEY or LLM_BINDING_API_KEY in .env file!")
        return
    
    # Kiểm tra Ollama
    print("\n[INIT] Checking Ollama...")
    if not check_ollama_ready():
        print("[ERROR] Ollama not available. Please start Ollama first.")
        return
    print("[OK] Ollama ready!")
    
    rag = None
    try:
        # Hỏi người dùng có muốn xóa dữ liệu cũ không
        print("\n[?] Delete old data and start fresh? (y/n): ", end="")
        user_input = input().strip().lower()
        delete_old = user_input in ['y', 'yes']
        
        if delete_old:
            delete_old_data()
        
        # Initialize RAG
        print("\n[INIT] Initializing RAG...")
        
        # Initialize RAG
        print("[INIT] Initializing RAG...")
        rag = await initialize_rag()
        print("[OK] RAG initialized!")
        
        # Process files
        await process_files(rag, delete_old=False)
        
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user. Progress saved to checkpoint.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            await rag.finalize_storages()
            print("\n[OK] Storage finalized!")

if __name__ == "__main__":
    asyncio.run(main())
    print("\n[DONE]")
