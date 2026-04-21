# 🚀 Kế hoạch triển khai BM25 (Hybrid Search) cho Mode Beam — KLTN LightRAG

## 🎯 1. Mục tiêu
Tích hợp thuật toán Lexical Search (BM25) chạy song song với Dense Vector Search, **CHỈ ÁP DỤNG cho mode `beam`**. Các mode khác (`hybrid`, `mix`, `local`, `global`) giữ nguyên 100% logic gốc của LightRAG.

> [!IMPORTANT]
> **Lý do giới hạn phạm vi:**
> Mode `hybrid`, `mix` là các mode chính thức đã được LightRAG công bố (Published Baseline). Giữ nguyên chúng để tạo ra bài so sánh thực nghiệm công bằng trong KLTN:
> - **Cột trái:** Hybrid, Mix — Mode gốc LightRAG (không can thiệp)
> - **Cột phải:** Beam Search — Thuật toán do sinh viên phát triển (có BM25 Hybrid)
>
> Nếu Beam vượt mặt Hybrid/Mix → chứng minh giá trị đóng góp của đồ án.

---

## 📍 2. Vị trí Áp dụng & Thuật toán Kết hợp

BM25 chỉ can thiệp vào hàm `_perform_kg_search()` (beam mode) trong `operate.py`.
Sử dụng **Weighted Sum (Tổng có trọng số)** + **Min-Max Normalization**.

### Vị trí 1: Tìm điểm Neo xuất phát (Anchor Entities) — ⭐ TÁC ĐỘNG LỚN NHẤT
- **Hiện tại (dòng 3554-3555 operate.py):** 100% dựa vào `entities_vdb.query()` (Dense Vector).
- **Sau khi có BM25:** Quét song song qua `entities_vdb` và `BM25_entities_index`.
- **Công thức:**
  `Anchor_Score = (anchor_alpha × Normalized_Semantic) + ((1 - anchor_alpha) × Normalized_BM25)`

### Vị trí 2: Tìm văn bản hỗ trợ (Text Chunks) — 📎 TÁC ĐỘNG BỔ TRỢ
- **Cơ chế:** Quét song song qua `chunks_vdb` và `BM25_chunks_index`.
- **Công thức:**
  `Chunk_Score = (chunk_alpha × Normalized_Semantic) + ((1 - chunk_alpha) × Normalized_BM25)`

> [!NOTE]
> **Tại sao tách riêng `anchor_alpha` và `chunk_alpha`?**
> Entities (tên ngắn, dễ match từ khóa) và Chunks (đoạn văn dài, giàu ngữ nghĩa) có đặc tính dữ liệu rất khác nhau.

> [!WARNING]
> **KHÔNG dùng BM25 khi đang duyệt/hop trong Beam Search!**
> Các node lân cận mang tính suy luận khái niệm (ví dụ: `[Aspirin]` → `[Đau dạ dày]`). Node `[Đau dạ dày]` không chứa chữ "Aspirin" nên BM25 sẽ cho điểm 0 → bị cắt tỉa sai.

---

## 🛠️ 3. Các bước triển khai

### Bước 1: Cài đặt thư viện
```bash
pip install rank-bm25
```
- **Tokenizer:** Dùng Whitespace Tokenizer (split bằng khoảng trắng + lowercase). Phù hợp cho tên thuốc Latin/English trong dữ liệu y khoa.

### Bước 2: Viết module `lightrag/bm25_storage.py`
Class `BM25Storage` sẽ đảm nhiệm:
1. **Load Data:** Đọc text (tên + mô tả) của Entities và Chunks từ KVStorage.
2. **Build Index:** Tokenize và đưa vào `BM25Okapi`.
3. **Save/Load:** Lưu file `.pkl` để tái sử dụng (không build lại mỗi lần khởi động server).
4. **Query & Normalize:** Trả về Top N ID + điểm đã chuẩn hóa về [0, 1].

> [!CAUTION]
> **Edge Case: BM25 Score = 0 cho tất cả ứng viên!**
> Khi câu hỏi mang tính ngữ nghĩa thuần túy (ví dụ: "Thuốc nào dùng cho trẻ em?"), BM25 trả điểm = 0 cho toàn bộ.
> `max_bm25_score = 0` → **chia cho 0** khi normalize!
> **Xử lý:** Nếu `max_bm25_score == 0`, fallback về Vector-only mode (bỏ qua BM25 hoàn toàn).

### Bước 3: Script `build_bm25_index.py` (Chạy 1 lần duy nhất)
- Đọc Entities và Chunks từ working-dir.
- Build và lưu `bm25_entities.pkl` + `bm25_chunks.pkl`.
- **Thời gian chạy:** < 5 giây.
- Không phá vỡ logic Insert/Upsert của LightRAG.

### Bước 4: Sửa `operate.py` — CHỈ hàm `_perform_kg_search()`
1. Khi khởi tạo beam search, load BM25 index.
2. **Phase 1 (Tìm Anchor):**
   - Gọi `BM25Storage.query_entities()` song song với `entities_vdb.query()`.
   - Hợp nhất bằng Weighted Sum → lấy `beam_max_anchor` nodes.
3. **Phase 3 (Tìm Chunks):**
   - Gọi `BM25Storage.query_chunks()` song song với `chunks_vdb.query()`.
   - Hợp nhất bằng Weighted Sum → lấy `chunk_top_k` chunks.
4. **KHÔNG đụng chạm** bất kỳ hàm nào khác (`_find_most_related_*`, `hybrid`, `mix`, `local`, `global`).

### Bước 5: Thêm API Params
- Thêm `anchor_alpha` (default=0.5) và `chunk_alpha` (default=0.7) vào `QueryParam` và `QueryRequest`.
- Cập nhật `beam_tuning.py` và `eval_ragas_hybrid_mix_beam.py`.

---

## 📊 4. Ước tính Cải thiện (Beam Search: Vector-only → Hybrid BM25)

| Metric | Hiện tại | Ước tính Hybrid | Phân tích |
|---|---|---|---|
| **Context Recall** | `0.65 - 0.70` | 🚀 **`0.85 - 0.95`** | Bắt chính xác tên thuốc → đúng anchor → đúng context |
| **Faithfulness** | `0.45 - 0.50` | 🚀 **`0.65 - 0.75`** | Context chuẩn xác → LLM bớt hallucinate |
| **Context Precision** | `0.90 - 1.00` | ➖ **`0.90 - 1.00`** | Beam vẫn lọc rác tốt |
| **Answer Relevancy** | `0.80 - 0.82` | 📈 **`0.83 - 0.88`** | Trả lời đúng trọng tâm hơn |
| **RAGAS Score** | `~0.70` | 🏆 **`0.80 - 0.88`** | Cạnh tranh trực tiếp với Mix/Hybrid |
| **Input Tokens** | `~13,500` | `~14,000 - 15,000` | Vẫn tiết kiệm hơn Mix (~18,000) |
| **Retrieval Latency** | `~5,100 ms` | `~5,200 ms` | +50-100ms (BM25 chỉ dùng CPU) |

---

## 📝 5. Khung So sánh Thực nghiệm cho KLTN

Sau khi triển khai xong, bảng so sánh cuối cùng trong báo cáo sẽ có dạng:

| Metric | Hybrid (LightRAG gốc) | Mix (LightRAG gốc) | **Beam (Đề xuất)** | Winner |
|---|---|---|---|---|
| Context Recall | ? | ? | ? | ? |
| Faithfulness | ? | ? | ? | ? |
| RAGAS Score | ? | ? | ? | ? |
| Input Tokens | ? | ? | ? | ? |
| Retrieval Latency | ? | ? | ? | ? |

> Hybrid và Mix giữ nguyên code gốc LightRAG → **Published Baseline**.
> Beam là mode do sinh viên phát triển → **Proposed Method**.
> So sánh này đảm bảo tính công bằng và khách quan cho hội đồng.

---

## ⏳ 6. Ước tính thời gian
| Bước | Nội dung | Thời gian |
|---|---|---|
| 1 | Cài `rank-bm25` | 1 phút |
| 2 | Viết `bm25_storage.py` | ~2 tiếng |
| 3 | Script Build Index | ~30 phút |
| 4 | Sửa `operate.py` (chỉ hàm beam) | ~3-4 tiếng |
| 5 | Cập nhật API params + scripts | ~30 phút |
| | **Tổng cộng** | **~1 ngày** |

---

## 🔑 7. Sơ đồ Kiến trúc (Mode Beam — Hybrid Search)

```
Câu hỏi → LLM Keyword Extraction → ll_keywords / hl_keywords
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
            entities_vdb            BM25_entities         relationships_vdb
            (Dense Vector)          (Sparse Lexical)       (Dense Vector)
                    │                     │                     │
                    └──────┬──────────────┘                     │
                           ▼                                    │
                  Weighted Sum + Normalize                      │
                  anchor_alpha tunable                          │
                           │                                    │
                           ▼                                    │
                  TOP K Anchor Entities ◄───────────────────────┘
                           │
                           ▼
              ╔════════════════════════╗
              ║  BEAM SEARCH (hop)     ║  ← Không dùng BM25 ở đây
              ║  Score = Semantic      ║
              ║       + Parent Path    ║
              ║       + Edge Weight    ║
              ║       - Length Penalty  ║
              ╠════════════════════════╣
              ║  Adaptive Pruning      ║
              ╚════════════════════════╝
                           │
                           ▼
                  Collected Entities & Relations
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              chunks_vdb    BM25_chunks
              (Dense)       (Sparse)
                    │             │
                    └──────┬──────┘
                           ▼
                  Weighted Sum + Normalize
                  chunk_alpha tunable
                           │
                           ▼
                  TOP K Text Chunks
                           │
                           ▼
                  Final Context → LLM → Answer
```

*Lưu ý: Các mode hybrid, mix, local, global hoàn toàn không đi qua luồng này. Chúng giữ nguyên code gốc LightRAG.*
