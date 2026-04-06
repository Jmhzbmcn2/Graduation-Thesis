# Tổng hợp các file code & cấu hình của KLTN

> Khóa luận tốt nghiệp: Xây dựng Knowledge Graph trên dữ liệu y tế tiếng Việt với LightRAG
> Sinh viên: Vũ Duy Linh

---

## 🔴 NHÓM 1: File cấu hình — Nơi thiết lập toàn bộ hệ thống

### `.env`
> **Mục đích:** File cấu hình trung tâm, nơi định nghĩa toàn bộ kiến trúc hệ thống.

**Các chỉnh sửa chính:**
- `WORKING_DIR=./medical_rag` — Thư mục chứa dữ liệu KG y tế
- `LLM_BINDING=cloudflare_worker` — Chọn LLM binding tự viết (Cloudflare Worker chạy Llama 3.3 70B)
- `EMBEDDING_BINDING=ollama` / `EMBEDDING_MODEL=nomic-embed-text` — Dùng Ollama local cho embedding
- `SUMMARY_LANGUAGE=Vietnamese` — Buộc LLM xuất kết quả bằng tiếng Việt
- `ENTITY_TYPES=["Disease","Symptom","Drug","Chemical compound","Protein","Anatomy","Biological process","Exposure","Diagnostic test","Treatment method"]` — **Bộ 10 entity types y tế tự thiết kế**
- `MAX_ASYNC=2`, `MAX_PARALLEL_INSERT=1` — Giới hạn concurrency để tránh API rate limit
- `MAX_GLEANING=0` — Tắt gleaning (bóc lại entity) để tiết kiệm API call

---

## 🟡 NHÓM 2: File source code viết MỚI (không có trong repo gốc LightRAG)

### `lightrag/llm/cloudflare_worker.py` `[NEW]`
> **Mục đích:** LLM binding tự viết để kết nối LightRAG với Cloudflare Workers AI (chạy Llama 3.3 70B miễn phí).

**Chỉnh sửa:** Viết hoàn toàn mới 140 dòng. Gồm:
- Hàm `cloudflare_worker_complete()` gọi API qua `aiohttp`
- Retry logic với `tenacity` (3 lần, exponential backoff)
- Xử lý lỗi riêng: `CloudflareWorkerError`, `CloudflareWorkerRetryError`
- Format request theo chuẩn custom Worker (`prompt`, `systemPrompt`, `history`)

### `evaluate_kg_quality.py` `[NEW]`
> **Mục đích:** Script đánh giá chất lượng Knowledge Graph (511 dòng).

**Chỉnh sửa:** Viết hoàn toàn mới. Gồm 3 module đánh giá:
- **fCorrectness (Schema Compliance Rate)** — Tỷ lệ triple tuân thủ schema (Seo et al. 2022)
- **Average Clustering Coefficient** — Mức độ cụm hóa đồ thị (Newman 2010)
- **Graph Topology Metrics** — Density, Degree, Connected Components, Shannon Entropy
- Định nghĩa `PREDEFINED_SCHEMA_TYPES` (10 types y tế)
- Mapping `VIETNAMESE_TYPE_MAPPING` để phát hiện LLM hallucination tiếng Việt

### `testcase/eval_ragas.py` `[NEW]`
> **Mục đích:** Script đánh giá RAGAS so sánh Naive vs Hybrid (50 câu hỏi).

**Chỉnh sửa:** Viết hoàn toàn mới 241 dòng. Gồm:
- Query LightRAG API lấy answer + full context
- Chạy RAGAS 4 metrics (Faithfulness, Answer Relevancy, Context Recall, Context Precision)
- Dùng Gemini làm LLM Judge + Ollama nomic-embed-text làm embedding
- So sánh side-by-side Naive vs Hybrid, xuất kết quả ra Excel

### `testcase/eval_ragas_naive_mix.py` `[NEW]`
> **Mục đích:** Script RAGAS mở rộng — so sánh Naive vs Mix (50 câu từ 130 testcase).

**Chỉnh sửa:** Viết hoàn toàn mới 282 dòng. Cải tiến so với `eval_ragas.py`:
- Xuất 3 sheets Excel (Naive, Mix, Summary)
- Tính thêm Mean/Median/Std/Min/Max cho từng metric
- Tự động xác định Winner cho từng metric

### `testcase/eval_lightrag_paper.py` `[NEW]`
> **Mục đích:** Evaluation theo format paper gốc LightRAG.

### `testcase/eval_text_similarity.py` `[NEW]`
> **Mục đích:** Đánh giá bằng text similarity metrics.

### `testcase/run_testcase.py` `[NEW]`
> **Mục đích:** Script chạy batch test cases tự động.

### `testcase/prepare_ragas_test.py` `[NEW]`
> **Mục đích:** Chuẩn bị dữ liệu test cho RAGAS.

### `testcase/test_ragas_1question.py` `[NEW]`
> **Mục đích:** Test nhanh RAGAS trên 1 câu hỏi để debug.

---

## 🟢 NHÓM 3: File source code SỬA trong repo gốc LightRAG

### `lightrag/api/lightrag_server.py` `[MODIFIED]`
> **Mục đích:** Thêm hỗ trợ Cloudflare Worker binding vào API server.

**Chỉnh sửa:**
- Thêm `"cloudflare_worker"` vào danh sách các LLM binding hợp lệ (dòng 311)
- Viết hàm `create_cloudflare_worker_llm_func()` (dòng 601-626)
- Thêm nhánh `elif binding == "cloudflare_worker"` vào factory function (dòng 651-652)

### `lightrag/api/config.py` `[MODIFIED]`
> **Mục đích:** Đăng ký Cloudflare Worker vào danh sách binding config.

**Chỉnh sửa:**
- Thêm `"cloudflare_worker"` vào list các binding hợp lệ (dòng 235)

---

## 🔵 NHÓM 4: File hỗ trợ phân tích & tiện ích (thư mục `tmp/`)

| File | Mục đích |
|---|---|
| `tmp/check_entity_types.py` | Thống kê phân bố entity types trong KG |
| `tmp/analyze_other.py` | Phân tích nhóm "other" (lần 1) |
| `tmp/analyze_other2.py` | Phân tích nhóm "other" (lần 2, chi tiết hơn) |
| `tmp/analyze_other_entities.py` | Phân tích "other" kèm phân cụm keyword |
| `tmp/calc_kg_time.py` / `calc_kg_time2.py` | Tính thời gian xây dựng KG |
| `tmp/check_env.py` | Kiểm tra biến môi trường |
| `tmp/check_json.py` / `repair_json.py` | Kiểm tra và sửa file JSON bị lỗi |
| `tmp/test_vertex_ai.py` | Test kết nối Google Vertex AI |
| `tmp/test_chunking.py` | Test chunking strategy |

---

## 🟣 NHÓM 5: File fine-tuning embedding (thư mục `finetune/`)

| File | Mục đích |
|---|---|
| `finetune/generate_training_data.py` | Sinh dữ liệu huấn luyện từ KG y tế |
| `finetune/finetune_nomic_embed.py` | Fine-tune model nomic-embed-text |
| `finetune/training_data.json` | Dữ liệu pair (query, passage) |
| `finetune/training_data_triplet.json` | Dữ liệu triplet (anchor, positive, negative) |

---

## 🟠 NHÓM 6: Nguồn dữ liệu đầu vào & Kết quả

### Nguồn dữ liệu y tế (Input)
- **Đường dẫn gốc:** `C:\Users\VUDUYLINH\PycharmProjects\KLTN\Processing_Data\VietMed_Crawl_Data\vietmed_crawled\`
- **Tổng số file crawl được:** 1932 file `.txt` (bài báo y tế tiếng Việt)
- **Số file sử dụng cho KG:** 1500 file đầu tiên
- **Danh sách file đã dùng:** `medical_data_filelist.json` (1500 tên file kèm thứ tự)
- **Nguồn thu thập:** Crawl từ các trang y tế Việt Nam (VietMed)

### Kết quả đầu ra

| File | Mục đích |
|---|---|
| `testcase/50 testcase.xlsx` | 50 câu hỏi y tế test |
| `testcase/130_testcase.xlsx` | 130 câu hỏi y tế mở rộng |
| `testcase/ragas_naive_vs_hybrid_*.xlsx` | Kết quả RAGAS Naive vs Hybrid |
| `testcase/eval_ragas_naive_mix_2603.xlsx` | Kết quả RAGAS Naive vs Mix |
| `medical_rag_ollama/` | Thư mục chứa toàn bộ KG đã build (GraphML + VDB + KV) |

---

## 📋 TÓM TẮT

| Loại | Số file | Ghi chú |
|---|---|---|
| File cấu hình (.env) | 1 | Toàn bộ kiến trúc hệ thống |
| File viết MỚI hoàn toàn | ~12 | Cloudflare binding, KG evaluation, RAGAS scripts |
| File SỬA trong repo gốc | 2 | lightrag_server.py, config.py |
| File tiện ích phân tích | ~10 | Thư mục tmp/ |
| File fine-tuning | 4 | Thư mục finetune/ |
| Dữ liệu & kết quả | ~8 | Thư mục testcase/, medical_rag_ollama/ |
