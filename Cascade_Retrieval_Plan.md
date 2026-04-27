# Kế hoạch Triển khai Kiến trúc Dual-VectorDB & Cascade Retrieval (Dành riêng cho Focused Mode) - BẢN FINAL

Bản kế hoạch này mô tả quy trình nâng cấp cơ chế truy xuất điểm neo (anchor node retrieval) của LightRAG bằng việc kết hợp **Kiến trúc DB Vector Kép (Dual-VectorDB)** và **Lọc nối tiếp (Cascade Filter: BM25 -> Semantic Name-based)**. Mục tiêu cốt lõi là khắc phục triệt để hiện tượng *Semantic Dilution* (Pha loãng ngữ nghĩa) đối với các keyword chuyên ngành y tế mà không làm ảnh hưởng đến hiệu năng hay cấu trúc gốc của đồ thị tri thức.

## 1. Phân tích Hiện trạng & Mục tiêu thiết kế
*   **Hiện trạng (Semantic Dilution):** Khi thực hiện Focused Search mặc định, hệ thống gộp `Entity Name + Description` để nạp vào DB. Khi người dùng truy vấn một từ khóa ngắn (VD: "Trạch tả"), việc lấy Vector của từ khóa này so sánh với Vector của một đoạn văn bản dài (chứa nhiều từ nhiễu) làm điểm số Cosine bị giảm mạnh, khiến Hit rate chỉ đạt ~38%.
*   **Mục tiêu (Dual-DB & Cascade):**
    *   **Dual-VectorDB:** Xây dựng một DB phụ chỉ chứa Vector của `Entity Name` để phục vụ riêng cho việc so sánh ngữ nghĩa cực chuẩn xác. DB chính (`medical_rag_v2`) vẫn giữ nguyên không xâm phạm.
    *   **Cascade Retrieval:** Lớp 1 dùng BM25 để lọc thô (bảo toàn tính Exact Match). Lớp 2 dùng Vector từ DB phụ để tinh chỉnh (Re-ranking) lại thứ hạng.

---

## 2. Quy trình Thực thi (Bảo toàn 100% Database gốc)

Quy trình được chia làm 2 giai đoạn: Giai đoạn Offline (Chuẩn bị Data) và Giai đoạn Online (Truy xuất thực tế).

### Phase 1: Xây dựng DB Phụ (Offline Data Preparation - Thực hiện 1 lần)
*   **Mục tiêu:** Tạo ra một không gian Vector riêng biệt (VD: `vdb_entities_name_only.json`) độc lập hoàn toàn với Graph gốc.
*   **Hành động:** Viết một đoạn script Python chạy ngầm. Script này sẽ lặp qua toàn bộ 54.000 thực thể trong Knowledge Graph hiện tại, trích xuất `entity_name`, gọi API tính Vector và lưu vào file `vdb_entities_name_only.json`.
*   **Chi phí:** Thời gian chạy script tốn khoảng 5 - 15 phút. Dung lượng lưu trữ tăng thêm ~50MB.

### Phase 2: Thay đổi luồng Truy xuất (Online Dynamic Retrieval)
**Lưu ý cực kỳ quan trọng:** Tuyệt đối không chạm vào các hàm dùng chung như `_get_node_data`. Chỉ viết đè (override) logic bên trong hàm `_get_anchor_nodes` (nơi phục vụ riêng cho mode `focused` tại `lightrag/operate.py`).

*   **Bước 1 - Lexical Filter (Lọc thô):** Chạy `bm25.get_scores(low_level_keywords)` để lấy ra Top `N` chuỗi `entity_name` tiềm năng (Ví dụ $N = 30$).
*   **Bước 2 - Semantic Re-ranking (Lọc tinh - CHỈ SO SÁNH NAME):** 
    *   Tính Vector cho `low_level_keywords` (gọi API 1 lần).
    *   Trích xuất Vector của 30 `entity_name` vừa lọc trực tiếp từ file DB phụ `vdb_entities_name_only.json` (Gần như 0ms latency vì không phải gọi API tính lại).
    *   Tính Cosine Similarity giữa **Vector(Keyword) và Vector(Name)** bằng `numpy` trên RAM. Sắp xếp để chốt hạ danh sách `Top K` (Ví dụ $K = 5$) `entity_name` chuẩn xác nhất.
*   **Bước 3 - Bàn giao lại cho luồng Focused Mode gốc:**
    *   Sau khi có danh sách `Top K` cái tên và điểm Cosine của chúng, hàm `_get_anchor_nodes` kết thúc và **trả lại kết quả** (dưới dạng tuple `node_datas, anchor_scores`).
    *   Từ đây, logic đặc trưng của mode `focused` (hàm `_focused_1hop_edge_scoring`) sẽ **tiếp quản hoàn toàn**. Hệ thống sẽ tự động chui vào Graph gốc kéo `Name + Description` của các điểm neo này, tiếp tục nội suy mở rộng 1-hop (1-hop edge expansion) để thu thập các mối quan hệ (Relationships) xung quanh, sau đó áp dụng quota/threshold để chắt lọc trước khi nhét vào Prompt cho LLM. Tóm lại: Đặc sản của mode Focused được giữ nguyên 100%!

---

## 3. Quản lý Hyperparameters (Tham số hệ thống)
Cần bổ sung các tham số tinh chỉnh (Tuning) để chạy RAGAS evaluation:
*   `CASCADE_BM25_RECALL_SIZE` (hay $N$): Số lượng node Lớp 1 giữ lại. Đề xuất: $N = 30$ hoặc $50$.
*   `TOP_K_ENTITIES` (hay $K$): Số lượng điểm neo cuối cùng giữ lại ở Lớp 2. Đề xuất: $K = 5$.

---

## 4. Ưu điểm Kiến trúc & Khắc phục Rủi ro
1.  **Hiệu năng vô địch (Blazing Fast):** Nhờ cơ chế Dual-DB, việc Re-ranking 50 hay 500 node cũng chỉ diễn ra trong micro-giây do toàn bộ Vector của Name đã được tính sẵn (Pre-computed). Không còn nỗi lo "nghẽn cổ chai" API khi gặp High Concurrency.
2.  **Sạch sẽ & An toàn (Zero-touch Primary DB):** Việc phân tách rạch ròi giữa quá trình *Tìm kiếm (Chỉ dùng Name)* và quá trình *Sinh câu trả lời (Dùng Name + Description)* giúp Database Y khoa gốc không bị xâm phạm.
3.  **Khắc phục nhược điểm của BM25:** BM25 có thể vô tình nhặt những thực thể có chứa từ khóa nhưng sai ngữ cảnh. Lớp lọc Semantic Re-ranking số 2 đóng vai trò như một màng chắn cuối cùng, dọn dẹp các "rác từ vựng" để đảm bảo Top 5 nạp vào LLM là chuẩn y khoa 100%.

---

## 5. Tiêu chí Nghiệm thu (Acceptance Criteria)
Hệ thống được đánh giá là thành công nếu:
*   **Hit rate @ K=5** của luồng Focused Search tăng từ 38% lên trên 80%.
*   **Average Latency** (Độ trễ) gần như bằng phương pháp nguyên thủy do chi phí đã được trả trước (One-time Offline Cost).
*   **RAGAS Context Precision & Answer Relevancy** tăng tối thiểu 15-20%.
