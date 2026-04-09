#!/bin/bash
###############################################################################
#  setup_server.sh — Cài đặt toàn bộ môi trường cho LightRAG Graduation Thesis
#
#  Script này sẽ cài đặt:
#    1. Git
#    2. Miniconda (Python 3.10)
#    3. Conda environment "lightrag" với các thư viện cần thiết
#    4. vLLM (serving Qwen2.5-14B-Instruct-AWQ)
#    5. Ollama (serving nomic-embed-text embedding)
#    6. LightRAG + API dependencies
#    7. sentence-transformers (cho reranker server)
#    8. Bun (cho WebUI)
#    9. tmux (quản lý services)
#
#  Cách dùng:   chmod +x setup_server.sh && ./setup_server.sh
#  Hệ thống:    Ubuntu 20.04+ với NVIDIA GPU (RTX 3090)
###############################################################################

set -euo pipefail

# ─── Màu sắc cho output ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ─── Helper functions ────────────────────────────────────────────────────────
info()    { echo -e "${BLUE}ℹ️  $*${NC}"; }
success() { echo -e "${GREEN}✅ $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠️  $*${NC}"; }
error()   { echo -e "${RED}❌ $*${NC}"; exit 1; }
step()    { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; \
            echo -e "${CYAN}  $*${NC}"; \
            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ─── Biến cấu hình ──────────────────────────────────────────────────────────
MINICONDA_DIR="$HOME/miniconda"
CONDA_ENV_NAME="lightrag"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

VLLM_MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ"
VLLM_PORT=8000
OLLAMA_EMBED_MODEL="nomic-embed-text"

# ─── Kiểm tra quyền root ────────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

###############################################################################
#  BƯỚC 0: Kiểm tra GPU
###############################################################################
step "BƯỚC 0: Kiểm tra GPU NVIDIA"

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_INFO" ]; then
        success "Phát hiện GPU: $GPU_INFO"
    else
        warn "nvidia-smi tìm thấy nhưng không thể query GPU. Tiếp tục..."
    fi
else
    warn "Không tìm thấy nvidia-smi. Đảm bảo bạn đã cài NVIDIA driver!"
    warn "Nếu chưa cài driver, hãy cài trước rồi chạy lại script này."
    read -p "Bạn có muốn tiếp tục không? (y/n): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

###############################################################################
#  BƯỚC 1: Cài đặt Git & các system packages cần thiết
###############################################################################
step "BƯỚC 1: Cài đặt Git & system packages"

$SUDO apt-get update -qq
$SUDO apt-get install -y -qq git curl wget tmux build-essential > /dev/null 2>&1

if command -v git &> /dev/null; then
    success "Git đã được cài: $(git --version)"
else
    error "Cài Git thất bại!"
fi

if command -v tmux &> /dev/null; then
    success "tmux đã được cài: $(tmux -V)"
else
    error "Cài tmux thất bại!"
fi

###############################################################################
#  BƯỚC 2: Cài đặt Miniconda
###############################################################################
step "BƯỚC 2: Cài đặt Miniconda"

if [ -d "$MINICONDA_DIR" ] && [ -f "$MINICONDA_DIR/bin/conda" ]; then
    success "Miniconda đã tồn tại tại $MINICONDA_DIR"
    info "Phiên bản: $($MINICONDA_DIR/bin/conda --version)"
else
    info "Đang tải Miniconda..."
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
         -O "$MINICONDA_INSTALLER"

    info "Đang cài đặt Miniconda vào $MINICONDA_DIR..."
    bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_DIR"
    rm -f "$MINICONDA_INSTALLER"

    success "Miniconda đã cài thành công!"
fi

# Thêm conda vào PATH cho session hiện tại
export PATH="$MINICONDA_DIR/bin:$PATH"
eval "$($MINICONDA_DIR/bin/conda shell.bash hook)"

# Khởi tạo conda cho bash (để các session sau tự nhận)
$MINICONDA_DIR/bin/conda init bash > /dev/null 2>&1 || true
success "Conda version: $(conda --version)"

###############################################################################
#  BƯỚC 3: Tạo Conda environment
###############################################################################
step "BƯỚC 3: Tạo Conda environment '$CONDA_ENV_NAME' (Python $PYTHON_VERSION)"

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    success "Environment '$CONDA_ENV_NAME' đã tồn tại"
else
    info "Đang tạo environment '$CONDA_ENV_NAME' với Python $PYTHON_VERSION..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y -q
    success "Đã tạo environment '$CONDA_ENV_NAME'"
fi

# Kích hoạt environment
conda activate "$CONDA_ENV_NAME"
info "Python đang dùng: $(python --version) tại $(which python)"

###############################################################################
#  BƯỚC 4: Cài đặt PyTorch + CUDA
###############################################################################
step "BƯỚC 4: Cài đặt PyTorch (CUDA)"

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    success "PyTorch $TORCH_VER đã cài với CUDA support"
else
    info "Đang cài đặt PyTorch với CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
    success "PyTorch đã cài thành công!"
fi

###############################################################################
#  BƯỚC 5: Cài đặt vLLM
###############################################################################
step "BƯỚC 5: Cài đặt vLLM"

if python -c "import vllm; print(vllm.__version__)" 2>/dev/null; then
    VLLM_VER=$(python -c "import vllm; print(vllm.__version__)")
    success "vLLM $VLLM_VER đã được cài"
else
    info "Đang cài đặt vLLM (có thể mất vài phút)..."
    pip install vllm -q
    VLLM_VER=$(python -c "import vllm; print(vllm.__version__)")
    success "vLLM $VLLM_VER đã cài thành công!"
fi

###############################################################################
#  BƯỚC 6: Cài đặt Ollama
###############################################################################
step "BƯỚC 6: Cài đặt Ollama"

if command -v ollama &> /dev/null; then
    success "Ollama đã được cài: $(ollama --version)"
else
    info "Đang cài đặt Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    success "Ollama đã cài thành công: $(ollama --version)"
fi

# Pull embedding model
info "Đang pull model embedding '$OLLAMA_EMBED_MODEL' (nếu chưa có)..."
ollama pull "$OLLAMA_EMBED_MODEL" 2>/dev/null || {
    warn "Không thể pull model ngay. Ollama server có thể chưa chạy."
    warn "Hãy chạy 'ollama serve' rồi 'ollama pull $OLLAMA_EMBED_MODEL' sau."
}

###############################################################################
#  BƯỚC 7: Cài đặt LightRAG + dependencies
###############################################################################
step "BƯỚC 7: Cài đặt LightRAG và các thư viện liên quan"

cd "$PROJECT_DIR"

info "Đang cài đặt LightRAG ở chế độ editable với API extras..."
pip install -e ".[api]" -q

info "Đang cài đặt thêm các thư viện LLM provider..."
pip install openai anthropic ollama -q

info "Đang cài đặt sentence-transformers (cho Reranker server)..."
pip install sentence-transformers -q

success "Đã cài đặt xong LightRAG và các dependencies!"

# Kiểm tra LightRAG
LIGHTRAG_VER=$(python -c "import lightrag; print(lightrag.__version__)" 2>/dev/null || echo "unknown")
info "LightRAG version: $LIGHTRAG_VER"

###############################################################################
#  BƯỚC 8: Cài đặt Bun (cho WebUI)
###############################################################################
step "BƯỚC 8: Cài đặt Bun (cho WebUI)"

export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"

if command -v bun &> /dev/null; then
    success "Bun đã được cài: $(bun --version)"
else
    info "Đang cài đặt Bun..."
    curl -fsSL https://bun.sh/install | bash
    export PATH="$BUN_INSTALL/bin:$PATH"
    success "Bun đã cài thành công: $(bun --version)"
fi

# Cài dependencies cho WebUI
if [ -d "$PROJECT_DIR/lightrag_webui" ]; then
    info "Đang cài dependencies cho WebUI..."
    cd "$PROJECT_DIR/lightrag_webui"
    bun install --silent 2>/dev/null || bun install
    cd "$PROJECT_DIR"
    success "WebUI dependencies đã được cài!"
else
    warn "Không tìm thấy thư mục lightrag_webui/"
fi

###############################################################################
#  BƯỚC 9: Tải trước model vLLM (tùy chọn)
###############################################################################
step "BƯỚC 9: Tải trước model vLLM"

info "Model vLLM: $VLLM_MODEL"
info "Model sẽ được tự tải khi khởi động vLLM lần đầu tiên."
info "Nếu muốn tải trước, chạy:"
echo ""
echo "    source \$HOME/miniconda/bin/activate lightrag"
echo "    python -c \"from huggingface_hub import snapshot_download; snapshot_download('$VLLM_MODEL')\""
echo ""

###############################################################################
#  BƯỚC 10: Kiểm tra cấu hình .env
###############################################################################
step "BƯỚC 10: Kiểm tra file .env"

if [ -f "$PROJECT_DIR/.env" ]; then
    success "File .env đã tồn tại"
    info "Các cấu hình chính:"
    echo "    LLM_MODEL=$(grep '^LLM_MODEL=' "$PROJECT_DIR/.env" | cut -d= -f2 || echo 'N/A')"
    echo "    LLM_BINDING_HOST=$(grep '^LLM_BINDING_HOST=' "$PROJECT_DIR/.env" | cut -d= -f2 || echo 'N/A')"
    echo "    EMBEDDING_MODEL=$(grep '^EMBEDDING_MODEL=' "$PROJECT_DIR/.env" | cut -d= -f2 || echo 'N/A')"
    echo "    EMBEDDING_BINDING_HOST=$(grep '^EMBEDDING_BINDING_HOST=' "$PROJECT_DIR/.env" | cut -d= -f2 || echo 'N/A')"
else
    warn "File .env chưa tồn tại!"
    if [ -f "$PROJECT_DIR/env.example" ]; then
        info "Đang tạo .env từ env.example..."
        cp "$PROJECT_DIR/env.example" "$PROJECT_DIR/.env"
        warn "⚠️  Hãy chỉnh sửa file .env cho phù hợp với cấu hình của bạn!"
    fi
fi

###############################################################################
#  TỔNG KẾT
###############################################################################
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  🎉 CÀI ĐẶT HOÀN TẤT!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}📦 Các thành phần đã cài:${NC}"
echo "    ✅ Git:                  $(git --version 2>/dev/null || echo 'N/A')"
echo "    ✅ Miniconda:            $(conda --version 2>/dev/null || echo 'N/A')"
echo "    ✅ Conda env:            $CONDA_ENV_NAME (Python $PYTHON_VERSION)"
echo "    ✅ PyTorch:              $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "    ✅ CUDA available:       $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "    ✅ vLLM:                 $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'N/A')"
echo "    ✅ Ollama:               $(ollama --version 2>/dev/null || echo 'N/A')"
echo "    ✅ LightRAG:             $LIGHTRAG_VER"
echo "    ✅ Bun:                  $(bun --version 2>/dev/null || echo 'N/A')"
echo "    ✅ tmux:                 $(tmux -V 2>/dev/null || echo 'N/A')"
echo ""
echo -e "${CYAN}🚀 Để khởi động toàn bộ services:${NC}"
echo "    cd $PROJECT_DIR"
echo "    ./tmux_start.sh"
echo ""
echo -e "${CYAN}📋 Services sẽ chạy ở:${NC}"
echo "    [0] Ollama Embedding    → http://localhost:11434"
echo "    [1] vLLM (Qwen2.5-14B) → http://localhost:$VLLM_PORT"
echo "    [2] LightRAG API       → http://localhost:9621"
echo "    [3] Web UI              → http://localhost:5173/webui/"
echo ""
echo -e "${CYAN}🧪 Để test models:${NC}"
echo "    source \$HOME/miniconda/bin/activate lightrag"
echo "    python test_local_models.py"
echo ""
echo -e "${YELLOW}💡 Lưu ý: Nếu đây là lần đầu chạy vLLM, model '$VLLM_MODEL'${NC}"
echo -e "${YELLOW}   sẽ được tải tự động (~8GB). Quá trình này có thể mất vài phút.${NC}"
echo ""
