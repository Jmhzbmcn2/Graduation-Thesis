# 🏥 Medical RAG + Knowledge Graph

Hệ thống Hỏi đáp Y tế Tiếng Việt sử dụng [LightRAG](https://github.com/HKUDS/LightRAG) — kết hợp Knowledge Graph với RAG.  
**Khóa luận tốt nghiệp** — Vũ Duy Linh.

## Yêu cầu

- Python ≥ 3.10
- [Ollama](https://ollama.com/) (embedding local)
- [Bun](https://bun.sh/) (WebUI, tùy chọn)
- OpenRouter API Key — lấy tại [OpenRouter](https://openrouter.ai/keys)
  - Model: `qwen/qwen3-30b-a3b-instruct-2507`

## Cài đặt & Chạy

```bash
# 1. Clone & cài đặt
git clone https://github.com/VuDuyLinh150804/Graduation-Thesis.git
cd Graduation-Thesis
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[api]"

# 2. Pull embedding model
ollama pull embeddinggemma:300m

# 3. Tạo file .env (copy từ env.example, điền API key)
cp env.example .env
# Sửa OPENROUTER_API_KEY=<your-openrouter-api-key>
# Sửa LLM_BINDING=openai
# Sửa LLM_BINDING_HOST=https://openrouter.ai/api/v1

# 4. Chạy
ollama serve                                        # Terminal 1: Embedding server
lightrag-server --working-dir ./medical_rag/medical_rag_v2  # Terminal 2: API server (port 9621)
```

API docs: `http://localhost:9621/docs`

### WebUI (tùy chọn)

```bash
cd lightrag_webui && bun install && bun run dev   # http://localhost:5173
```

## Truy vấn

```bash
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Paracetamol có tác dụng phụ gì?", "mode": "hybrid"}'
```

**Các mode:** `naive` (vector search) · `hybrid` (KG + vector) · `local` · `global` · `mix`

## Công nghệ

LightRAG · OpenRouter (qwen3-30b-a3b-instruct) · Ollama (embeddinggemma:300m) · NetworkX · FastAPI · React + Vite