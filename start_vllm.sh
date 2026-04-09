#!/bin/bash
source $HOME/miniconda/bin/activate lightrag
echo "Đang khởi động vLLM server với model Qwen/Qwen2.5-14B-Instruct-AWQ..."
echo "Quá trình tải model (nếu chưa tải) có thể mất vài phút."
nohup vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --dtype auto --max-model-len 8192 --port 8000 > vllm.log 2>&1 &
echo "vLLM server đã được chạy ngầm! Bạn có thể xem tiến trình bằng lệnh: tail -f vllm.log"
