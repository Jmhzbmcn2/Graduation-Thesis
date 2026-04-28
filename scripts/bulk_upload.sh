#!/bin/bash
# bulk_upload.sh — Upload toàn bộ .txt files trong DATA_DIR lên LightRAG server
#
# Dùng: bash scripts/bulk_upload.sh /path/to/data_dir
# Ví dụ: bash scripts/bulk_upload.sh /home/Graduation-Thesis/inputs/vietmed_crawled

set -euo pipefail

DATA_DIR="${1:-}"
SERVER="${LIGHTRAG_SERVER:-http://localhost:9621}"
DELAY="${UPLOAD_DELAY:-0.3}"   # seconds between uploads (tránh overload)

if [[ -z "$DATA_DIR" ]]; then
    echo "Usage: $0 <data_directory>"
    echo "Example: $0 /home/Graduation-Thesis/inputs/vietmed_crawled"
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Directory not found: $DATA_DIR"
    exit 1
fi

# Đếm tổng số file
TOTAL=$(find "$DATA_DIR" -name "*.txt" | wc -l)
if [[ "$TOTAL" -eq 0 ]]; then
    echo "No .txt files found in $DATA_DIR"
    exit 0
fi

echo "=========================================="
echo " LightRAG Bulk Upload"
echo "=========================================="
echo " Server   : $SERVER"
echo " Data dir : $DATA_DIR"
echo " Files    : $TOTAL .txt files"
echo "=========================================="
echo ""

# Kiểm tra server có online không
if ! curl -sf "$SERVER/health" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach server at $SERVER"
    echo "Hãy đảm bảo server đang chạy (tmux_start.sh)"
    exit 1
fi

echo "Server OK. Bắt đầu upload..."
echo ""

COUNT=0
FAILED=0
SKIPPED=0

while IFS= read -r -d '' FILE; do
    COUNT=$((COUNT + 1))
    FILENAME=$(basename "$FILE")

    printf "[%d/%d] Uploading: %s ... " "$COUNT" "$TOTAL" "$FILENAME"

    HTTP_CODE=$(curl -s -o /tmp/upload_response.json -w "%{http_code}" \
        -X POST "$SERVER/documents/upload" \
        -F "file=@${FILE};type=text/plain" \
        2>/dev/null)

    if [[ "$HTTP_CODE" == "200" ]]; then
        echo "✅ OK"
    elif [[ "$HTTP_CODE" == "409" ]]; then
        echo "⏭️  Skipped (already exists)"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "❌ FAILED (HTTP $HTTP_CODE)"
        cat /tmp/upload_response.json 2>/dev/null || true
        FAILED=$((FAILED + 1))
    fi

    sleep "$DELAY"

done < <(find "$DATA_DIR" -name "*.txt" -print0 | sort -z)

echo ""
echo "=========================================="
echo " Upload hoàn tất"
echo "=========================================="
echo " Uploaded : $((COUNT - FAILED - SKIPPED))"
echo " Skipped  : $SKIPPED (already in system)"
echo " Failed   : $FAILED"
echo "=========================================="
echo ""
echo "Server sẽ tự xử lý queue trong nền."
echo "Theo dõi progress: tmux attach -t lightrag_services"
