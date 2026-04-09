#!/bin/bash

SESSION_NAME="lightrag_services"

echo "🛑 Đang dừng toàn bộ dịch vụ LightRAG..."

# Kill tmux session (sẽ kill toàn bộ process bên trong)
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux kill-session -t $SESSION_NAME
    echo "   ✅ tmux session '$SESSION_NAME' đã bị kill"
else
    echo "   ℹ️  Không tìm thấy tmux session '$SESSION_NAME'"
fi

# Kill các process còn sót lại
echo ""
echo "🔍 Kiểm tra và kill các process sót lại..."

for pattern in "vllm serve" "ollama serve" "lightrag-server" "vite --host"; do
    pids=$(pgrep -f "$pattern" 2>/dev/null)
    if [ -n "$pids" ]; then
        pkill -f "$pattern" 2>/dev/null
        echo "   ✅ Đã kill: $pattern (PIDs: $pids)"
    fi
done

sleep 2
echo ""
echo "✅ Tất cả dịch vụ đã được dừng."
echo "   Chạy './tmux_start.sh' để khởi động lại."
