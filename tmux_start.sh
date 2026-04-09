#!/bin/bash

SESSION_NAME="lightrag_services"

echo "Đang dọn dẹp các tiến trình cũ (nếu có)..."
pkill -f "vllm serve"
pkill -f "ollama serve"

# Đợi một chút để cổng được giải phóng
sleep 3

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
  tmux kill-session -t $SESSION_NAME
fi

# Cửa sổ 0: Ollama Embedding
tmux new-session -d -s $SESSION_NAME -n 'Ollama'
tmux send-keys -t $SESSION_NAME:0 'ollama serve' C-m

# Cửa sổ 1: vLLM (Qwen2.5-14B-Instruct-AWQ)
tmux new-window -t $SESSION_NAME -n 'vLLM'
tmux send-keys -t $SESSION_NAME:1 'source $HOME/miniconda/bin/activate lightrag' C-m
tmux send-keys -t $SESSION_NAME:1 'vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --dtype auto --max-model-len 8192 --port 8000' C-m

# Cửa sổ 2: LightRAG API Server
tmux new-window -t $SESSION_NAME -n 'LightRAG_Server'
tmux send-keys -t $SESSION_NAME:2 'source $HOME/miniconda/bin/activate lightrag' C-m
tmux send-keys -t $SESSION_NAME:2 'cd /home/linhvd/Graduation-Thesis && lightrag-server --workspace ./medical_rag_ollama' C-m

# Cửa sổ 3: Web UI (Bun + Vite)
tmux new-window -t $SESSION_NAME -n 'WebUI'
tmux send-keys -t $SESSION_NAME:3 'export PATH=$HOME/.bun/bin:$PATH' C-m
tmux send-keys -t $SESSION_NAME:3 'cd /home/linhvd/Graduation-Thesis/lightrag_webui && bun install && bun run dev --host --port 5173' C-m

echo ""
echo "✅ Đã khởi động toàn bộ dịch vụ trong tmux session '$SESSION_NAME':"
echo "   [0] Ollama Embedding    -> http://localhost:11434"
echo "   [1] vLLM Qwen2.5-14B   -> http://localhost:8000"
echo "   [2] LightRAG API Server -> http://localhost:9621"
echo "   [3] Web UI              -> http://localhost:5173/webui/"
echo ""
echo "👉 Để kiểm tra log, chạy: tmux attach -t $SESSION_NAME"
echo "   Ctrl+B rồi số [0/1/2/3] để chuyển cửa sổ"
echo "   Ctrl+B rồi D để thoát ra ngoài (server vẫn chạy ngầm)"
