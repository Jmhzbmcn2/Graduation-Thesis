#!/bin/bash

SESSION_NAME="lightrag_services"
OLLAMA_EMBED_MODEL="google/embeddinggemma:300m"

echo "Đang dọn dẹp các tiến trình cũ (nếu có)..."
pkill -f "vllm serve"
pkill -f "ollama serve"
pkill -f "rerank_server.py"

# Đợi một chút để cổng được giải phóng
sleep 3

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
  tmux kill-session -t $SESSION_NAME
fi

# Cửa sổ 0: Ollama Embedding
tmux new-session -d -s $SESSION_NAME -n 'Ollama'
tmux send-keys -t $SESSION_NAME:0 'ollama serve' C-m

# Cửa sổ 1: vLLM — LLM Judge
# [ACTIVE] Qwen2.5-14B-Instruct-AWQ (chat, 32K context, ~10GB VRAM)
tmux new-window -t $SESSION_NAME -n 'vLLM'
tmux send-keys -t $SESSION_NAME:1 'source $HOME/miniconda/bin/activate lightrag' C-m
tmux send-keys -t $SESSION_NAME:1 'vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --dtype auto --max-model-len 32768 --port 8000 --gpu-memory-utilization 0.75' C-m
# [BACKUP] Qwen2.5-32B-Instruct-AWQ (chat, 20K context, ~22GB — cần Ollama CPU)
# tmux send-keys -t $SESSION_NAME:1 'vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --dtype auto --max-model-len 20480 --port 8000 --gpu-memory-utilization 0.94' C-m
# (Nếu dùng 32B: đổi ollama serve → CUDA_VISIBLE_DEVICES="" ollama serve)

# Cửa sổ 2: LightRAG API Server
tmux new-window -t $SESSION_NAME -n 'LightRAG_Server'
tmux send-keys -t $SESSION_NAME:2 'source $HOME/miniconda/bin/activate lightrag' C-m
tmux send-keys -t $SESSION_NAME:2 'cd /home/Graduation-Thesis && ENABLE_LLM_CACHE=false lightrag-server --workspace /home/Graduation-Thesis/medical_rag_v6' C-m

# Cửa sổ 3: Reranker (GPU) — Qwen3-Reranker-0.6B
tmux new-window -t $SESSION_NAME -n 'Reranker'
tmux send-keys -t $SESSION_NAME:3 'source $HOME/miniconda/bin/activate lightrag' C-m
tmux send-keys -t $SESSION_NAME:3 'cd /home/Graduation-Thesis && RERANKER_BATCH_SIZE=4 python scripts/rerank_server.py' C-m

# WebUI — đã build production, được serve trực tiếp từ LightRAG API tại /webui
# Không cần Vite dev server nữa. Nếu cần rebuild WebUI:
#   cd /home/Graduation-Thesis/lightrag_webui && bun run build
# Rồi restart cửa sổ [2] LightRAG_Server

echo ""
echo "✅ Đã khởi động toàn bộ dịch vụ trong tmux session '$SESSION_NAME':"
echo "   [0] Ollama Embedding ($OLLAMA_EMBED_MODEL) -> http://localhost:11434"
echo "   [1] vLLM Qwen2.5-14B-Instruct-AWQ         -> http://localhost:8000  (32K context, 80% GPU)"
echo "   [2] LightRAG API + WebUI                   -> http://localhost:9621/webui/"
echo "   [3] Reranker Qwen3-Reranker-0.6B           -> http://localhost:7997  (batch=4)"
echo ""
echo "👉 Để kiểm tra log, chạy: tmux attach -t $SESSION_NAME"
echo "   Ctrl+B rồi số [0/1/2/3] để chuyển cửa sổ"
echo "   Ctrl+B rồi D để thoát ra ngoài (server vẫn chạy ngầm)"
