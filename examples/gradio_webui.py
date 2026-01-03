"""
Medical RAG - Gradio Web UI
Chat với dữ liệu y tế đã xử lý
"""

import os
import asyncio
import torch
import numpy as np
import gradio as gr
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, setup_logger

# Setup
setup_logger("lightrag", level="INFO")

# Load .env manually
env_path = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\.env"
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Configuration
MEGALLM_API_KEY = os.getenv("MEGALLM_API_KEY") or os.getenv("LLM_BINDING_API_KEY", "")
MEGALLM_BASE_URL = os.getenv("MEGALLM_BASE_URL") or os.getenv("LLM_BINDING_HOST", "https://ai.megallm.io/v1")
MEGALLM_MODEL = os.getenv("MEGALLM_MODEL") or os.getenv("LLM_MODEL", "deepseek-ai/deepseek-v3.1")
WORKING_DIR = "./medical_rag"
HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_EMBEDDING_DIM = 384

print(f"API Key loaded: {'Yes (' + MEGALLM_API_KEY[:10] + '...)' if MEGALLM_API_KEY else 'No'}")
print(f"API Base URL: {MEGALLM_BASE_URL}")
print(f"Model: {MEGALLM_MODEL}")

# Global variables
_hf_tokenizer = None
_hf_embed_model = None
_rag = None
_loop = None

def load_hf_embedding_model():
    global _hf_tokenizer, _hf_embed_model
    if _hf_tokenizer is None:
        print("Loading HuggingFace embedding model...")
        from transformers import AutoTokenizer, AutoModel
        _hf_tokenizer = AutoTokenizer.from_pretrained(HF_EMBEDDING_MODEL)
        _hf_embed_model = AutoModel.from_pretrained(HF_EMBEDDING_MODEL)
        if torch.cuda.is_available():
            _hf_embed_model = _hf_embed_model.cuda()
            print("Using CUDA GPU")
        _hf_embed_model.eval()
        print("Embedding model loaded!")
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

async def megallm_complete(prompt: str, system_prompt: str = None, history_messages: list = [], **kwargs) -> str:
    return await openai_complete_if_cache(
        model=MEGALLM_MODEL, prompt=prompt, system_prompt=system_prompt,
        history_messages=history_messages, api_key=MEGALLM_API_KEY, base_url=MEGALLM_BASE_URL, **kwargs
    )

async def async_initialize_rag():
    global _rag
    if _rag is None:
        print("Initializing RAG...")
        embedding_func = EmbeddingFunc(embedding_dim=HF_EMBEDDING_DIM, max_token_size=512, func=hf_embedding_func)
        _rag = LightRAG(working_dir=WORKING_DIR, llm_model_func=megallm_complete, llm_model_name=MEGALLM_MODEL, embedding_func=embedding_func)
        await _rag.initialize_storages()
        print("RAG initialized!")
    return _rag

def get_event_loop():
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop

def query_rag(question: str, mode: str):
    """Query RAG và trả về kết quả"""
    if not question.strip():
        return ""
    
    if not MEGALLM_API_KEY:
        return "⚠️ Chưa cấu hình API Key trong file .env"
    
    async def _query():
        rag = await async_initialize_rag()
        result = await rag.aquery(question, param=QueryParam(mode=mode))
        return result
    
    try:
        loop = get_event_loop()
        answer = loop.run_until_complete(_query())
        return str(answer)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Lỗi: {str(e)}"

# Simple Gradio UI (not chatbot style)
with gr.Blocks(title="Medical RAG") as demo:
    gr.Markdown("# 🏥 Medical RAG - Hỏi đáp Y tế\nChat với 100 tài liệu y tế (3,252 entities, 4,661 relations)")
    
    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(label="Câu hỏi", placeholder="Nhập câu hỏi về thuốc, bệnh...", lines=2)
            mode = gr.Radio(["mix", "hybrid", "local", "global", "naive"], value="mix", label="Query Mode", info="mix: KG + Vector (khuyên dùng)")
            submit = gr.Button("🔍 Tìm kiếm", variant="primary")
            
        with gr.Column(scale=4):
            answer = gr.Textbox(label="Kết quả", lines=20, max_lines=30)
    
    submit.click(query_rag, [question, mode], answer)
    question.submit(query_rag, [question, mode], answer)

if __name__ == "__main__":
    print("="*50)
    print("Medical RAG - Gradio WebUI")
    print("="*50)
    load_hf_embedding_model()
    print("Pre-initializing RAG...")
    loop = get_event_loop()
    loop.run_until_complete(async_initialize_rag())
    print("\nStarting Gradio UI at http://localhost:7860\n")
    demo.launch(server_name="0.0.0.0", server_port=7860)
