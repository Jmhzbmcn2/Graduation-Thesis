"""
RAGAS Evaluation — So sánh Naive vs Hybrid vs Mix (500 test cases từ 500_cases.csv)
Khóa Luận Tốt Nghiệp

**Phiên bản CONTINUE**: Hỗ trợ chạy tiếp từ kết quả đã lưu.
  - Đọc file output Excel đã có, xác định câu nào đã chạy rồi
  - Chỉ query + evaluate các câu CHƯA chạy
  - Ghép kết quả cũ + mới, tính lại Summary trên TOÀN BỘ

Đánh giá 4 metrics RAGAS:
  - Faithfulness: Câu trả lời có đúng theo context không?
  - Answer Relevancy: Câu trả lời có liên quan đến câu hỏi không?
  - Context Recall: Context truy xuất có đầy đủ không?
  - Context Precision: Context có chính xác không?

Cấu hình local server:
  - LLM Judge  : vLLM (Qwen2.5-14B-Instruct-AWQ) tại http://localhost:8000
  - Embedding  : Ollama (embeddinggemma:300m) tại http://localhost:11434
  - LightRAG   : http://localhost:9621

Output: eval_ragas_naive_mix_500cases.xlsx (mỗi mode 1 sheet + Summary)

Cách dùng:
  python testcase/eval_ragas_naive_mix_continue.py
"""
import os
import re
import time
import warnings
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=False)

# Suppress warnings
warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*token usage.*", category=UserWarning)

# ======================== CẤU HÌNH ========================
LIGHTRAG_URL = "http://localhost:9621"

# Đường dẫn file (server)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(_SCRIPT_DIR, "500_cases.csv")
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "eval_ragas_naive_mix_500cases.xlsx")

# LLM Judge — vLLM local (OpenAI-compatible)
# Có thể override qua environment variables
EVAL_LLM_MODEL   = os.getenv("EVAL_LLM_MODEL",    "Qwen/Qwen2.5-14B-Instruct-AWQ")
EVAL_LLM_API_KEY = os.getenv("EVAL_LLM_API_KEY",  "EMPTY")          # vLLM không cần key thật
EVAL_LLM_BASE_URL = os.getenv("EVAL_LLM_BASE_URL", "http://localhost:8000/v1")

# Embedding — Ollama local
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",        "embeddinggemma:300m")
EMBEDDING_HOST  = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")

# Số test case (None = dùng tất cả)
TEST_LIMIT = 300

# Giới hạn độ dài context (ký tự) để tránh vượt token limit của LLM Judge
# Qwen2.5-14B-Instruct-AWQ có context window 8192 tokens
# ~4 ký tự ≈ 1 token → 4000 ký tự ≈ 1000 tokens
MAX_CONTEXT_CHARS = None

# Modes cần đánh giá (phải là mode mà LightRAG API chấp nhận: naive, local, global, hybrid, mix, …)
MODES = ["naive", "hybrid", "mix"]

# Batch size cho RAGAS evaluation (tránh OOM khi đánh giá quá nhiều câu 1 lúc)
EVAL_BATCH_SIZE = 100
# ===========================================================


def clean_answer(answer: str) -> str:
    """Loại bỏ phần ### References khỏi answer"""
    return re.split(r"\n*###\s*References", answer, maxsplit=1)[0].strip()


def extract_chunks_from_context(raw_context: str) -> str:
    """
    Trích xuất context từ LightRAG cho RAGAS evaluation.

    Chiến lược: CHUNKS TRƯỚC, GRAPH SAU
      1. Document Chunks (text gốc)     ← đặt TRƯỚC → Context Precision cao
      2. Entity descriptions (verbalize) ← đặt SAU   → bổ sung Faithfulness
      3. Relation descriptions (verbalize)← đặt SAU   → bổ sung Faithfulness

    Lý do: LLM sinh answer từ cả Graph + Chunks. Nếu RAGAS chỉ thấy Chunks
    thì sẽ đánh tụt Faithfulness cho phần answer lấy từ Entity/Relation.
    """
    import json as _json

    chunk_start = raw_context.find("Document Chunks")
    if chunk_start == -1:
        return raw_context  # naive mode → trả nguyên gốc

    # --- 1. Trích xuất Document Chunks ---
    ref_start = raw_context.find("Reference Document List", chunk_start)
    chunk_section = raw_context[chunk_start:ref_start] if ref_start != -1 else raw_context[chunk_start:]

    chunk_texts = []
    json_match = re.search(r'```json\s*\n(.*?)```', chunk_section, re.DOTALL)
    if json_match:
        for line in json_match.group(1).strip().split('\n'):
            line = line.strip()
            if line:
                try:
                    obj = _json.loads(line)
                    content = obj.get("content", "")
                    if content:
                        chunk_texts.append(content.strip())
                except _json.JSONDecodeError:
                    pass

    # --- 2. Verbalize Entity descriptions ---
    graph_texts = []
    entity_start = raw_context.find("Knowledge Graph Data (Entity):")
    rel_start = raw_context.find("Knowledge Graph Data (Relationship):")

    if entity_start != -1:
        end_idx = rel_start if rel_start != -1 else chunk_start
        entity_section = raw_context[entity_start:end_idx]
        json_match = re.search(r'```json\s*\n(.*?)```', entity_section, re.DOTALL)
        if json_match:
            for line in json_match.group(1).strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        obj = _json.loads(line)
                        name = obj.get("entity", "")
                        desc = obj.get("description", "")
                        if name and desc:
                            graph_texts.append(f"{name}: {desc}")
                    except _json.JSONDecodeError:
                        pass

    # --- 3. Verbalize Relation descriptions ---
    if rel_start != -1:
        rel_section = raw_context[rel_start:chunk_start]
        json_match = re.search(r'```json\s*\n(.*?)```', rel_section, re.DOTALL)
        if json_match:
            for line in json_match.group(1).strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        obj = _json.loads(line)
                        e1 = obj.get("entity1", "")
                        e2 = obj.get("entity2", "")
                        desc = obj.get("description", "")
                        if e1 and e2 and desc:
                            graph_texts.append(f"{e1} - {e2}: {desc}")
                    except _json.JSONDecodeError:
                        pass

    # --- Ghép: Chunks trước, Graph sau ---
    result_parts = []
    if chunk_texts:
        result_parts.append("\n\n".join(chunk_texts))
    if graph_texts:
        result_parts.append("\n".join(graph_texts))

    if result_parts:
        return "\n\n".join(result_parts)

    return chunk_section.strip()


def truncate_context(text: str, max_chars: int = None) -> str:
    """Truncate context để không vượt token limit của LLM Judge"""
    if max_chars is None:
        max_chars = MAX_CONTEXT_CHARS

    if max_chars is None:
        return text

    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def query_lightrag(question: str, mode: str, retries: int = 3):
    """
    Query LightRAG: lấy answer + full context.
    - Call 1: normal query → answer
    - Call 2: only_need_context=True → full context (entities + relations + chunks)
    Với mode mix/local/global: chỉ trích xuất phần Document Chunks cho RAGAS.
    """
    base = {"query": question, "mode": mode, "stream": False, "top_k": 10}

    for attempt in range(retries):
        try:
            # Call 1: Lấy answer
            resp1 = requests.post(
                f"{LIGHTRAG_URL}/query",
                json={**base, "only_need_context": False},
                timeout=180,
            )
            resp1.raise_for_status()
            answer = clean_answer(resp1.json().get("response", ""))

            # Call 2: Lấy full context
            resp2 = requests.post(
                f"{LIGHTRAG_URL}/query",
                json={**base, "only_need_context": True},
                timeout=180,
            )
            resp2.raise_for_status()
            full_context = resp2.json().get("response", "")

            # Trích xuất CHỈ phần Document Chunks (bỏ Entity/Relation)
            if full_context:
                full_context = extract_chunks_from_context(full_context)
                full_context = truncate_context(full_context)
            contexts = [full_context] if full_context else ["No context retrieved"]
            return answer, contexts

        except Exception as e:
            print(f"    ⚠️ Lỗi (lần {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(3)

    return "Error: Không lấy được response", ["No context"]


def run_ragas_evaluation(questions, answers, contexts_list, ground_truths):
    """Chạy RAGAS evaluation trên dataset"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # LLM Judge — Qwen3-8B-AWQ (vLLM local)
    # Thinking mode TẮT: tránh LLMDidNotFinishException do think tokens chiếm hết output budget.
    # Qwen3 không thinking vẫn mạnh hơn Qwen2.5 về reasoning.
    llm = LangchainLLMWrapper(
        langchain_llm=ChatOpenAI(
            model=EVAL_LLM_MODEL,
            api_key=EVAL_LLM_API_KEY,
            base_url=EVAL_LLM_BASE_URL,
            temperature=0.0,
            max_tokens=2048,
            model_kwargs={
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            },
        ),
        bypass_n=True,
    )

    # Embedding — Ollama local qua OpenAI-compatible endpoint
    # Ollama expose /v1/embeddings tương thích OpenAI ở cổng 11434
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key="ollama",               # Ollama không cần key thật
            base_url=f"{EMBEDDING_HOST}/v1",
            check_embedding_ctx_length=False,
        )
    )

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()],
        llm=llm,
        embeddings=emb,
    )
    return results.to_pandas()


def create_summary_sheet(all_mode_results):
    """Tạo sheet Summary: thống kê theo từng mode trong MODES; Winner = mode có mean cao nhất."""
    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]
    metric_labels = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision", "RAGAS Score"]

    rows = []
    for label, col in zip(metric_labels, metric_cols):
        row = {"Metric": label}
        means = {}
        for mode in MODES:
            df = all_mode_results[mode]
            means[mode] = float(df[col].mean())
            row[f"{mode.capitalize()}_Mean"] = round(means[mode], 4)
            row[f"{mode.capitalize()}_Median"] = round(df[col].median(), 4)
            row[f"{mode.capitalize()}_Std"] = round(df[col].std(), 4)
            row[f"{mode.capitalize()}_Min"] = round(df[col].min(), 4)
            row[f"{mode.capitalize()}_Max"] = round(df[col].max(), 4)

        best = max(means, key=means.get)
        worst = min(means, key=means.get)
        row["Winner"] = best.capitalize()
        row["Spread_max_min"] = round(means[best] - means[worst], 4)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df


def load_existing_results() -> dict[str, pd.DataFrame]:
    """
    Đọc kết quả đã chạy từ file output Excel.
    Trả về dict {mode: DataFrame} cho mỗi mode đã có dữ liệu.
    Nếu file chưa tồn tại → trả về dict rỗng.
    """
    existing = {}
    if not os.path.exists(OUTPUT_FILE):
        return existing

    try:
        xls = pd.ExcelFile(OUTPUT_FILE, engine="openpyxl")
        for mode in MODES:
            sheet_name = mode.capitalize()[:31]
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                if "question_text" in df.columns and len(df) > 0:
                    existing[mode] = df
        xls.close()
    except Exception as e:
        print(f"⚠️ Không đọc được file output cũ: {e}")
        return {}

    return existing


def get_pending_questions(all_questions: list[str], existing_df: pd.DataFrame | None) -> tuple[list[int], list[str]]:
    """
    Xác định các câu hỏi CHƯA được đánh giá.
    Trả về (indices, questions) — indices tương ứng vị trí trong all_questions.
    """
    if existing_df is None or len(existing_df) == 0:
        return list(range(len(all_questions))), all_questions

    done_questions = set(existing_df["question_text"].astype(str).tolist())
    pending_indices = []
    pending_questions = []
    for i, q in enumerate(all_questions):
        if q not in done_questions:
            pending_indices.append(i)
            pending_questions.append(q)

    return pending_indices, pending_questions


def save_intermediate_results(all_mode_results: dict[str, pd.DataFrame]):
    """
    Lưu kết quả trung gian sau mỗi batch để tránh mất dữ liệu nếu bị crash.
    """
    try:
        summary_df = create_summary_sheet(all_mode_results)
        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            for mode in MODES:
                if mode in all_mode_results:
                    sheet = mode.capitalize()[:31]
                    all_mode_results[mode].to_excel(writer, sheet_name=sheet, index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
    except Exception as e:
        print(f"    ⚠️ Lưu trung gian thất bại: {e}")


def main():
    print("=" * 65)
    print(f"📊 RAGAS Evaluation (CONTINUE) — modes: {', '.join(MODES)}")
    print("   Khóa Luận Tốt Nghiệp")
    print("=" * 65)

    # 1. Đọc test cases
    print(f"\n📂 Input: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Tìm thấy {len(df)} test cases")

    # Giới hạn số câu nếu cần
    if TEST_LIMIT is not None:
        df = df.head(TEST_LIMIT)
        print(f"   Sử dụng {TEST_LIMIT} câu đầu tiên")
    else:
        print(f"   Sử dụng toàn bộ {len(df)} câu")

    # Chuẩn bị danh sách câu hỏi + ground truth
    all_questions = []
    all_ground_truths = []
    for _, row in df.iterrows():
        question = str(row.get("question", row.get("Question", "")))
        ground_truth = str(row.get("answer", row.get("Answer", "")))
        if question and question != "nan":
            all_questions.append(question)
            all_ground_truths.append(ground_truth)

    total = len(all_questions)
    print(f"   Tổng số câu hợp lệ: {total}")

    # 2. Đọc kết quả đã chạy (nếu có)
    print(f"\n📦 Kiểm tra kết quả đã lưu: {OUTPUT_FILE}")
    existing_results = load_existing_results()
    if existing_results:
        for mode, edf in existing_results.items():
            print(f"   ✅ Mode {mode.upper()}: đã có {len(edf)} câu")
    else:
        print("   ℹ️ Chưa có kết quả nào — chạy từ đầu")

    all_mode_results = {}

    for mode in MODES:
        print(f"\n{'=' * 65}")
        print(f"🔍 ĐÁNH GIÁ MODE: {mode.upper()} ({total} câu hỏi)")
        print(f"{'=' * 65}")

        # Xác định câu nào chưa chạy
        existing_df = existing_results.get(mode, None)
        pending_indices, pending_questions = get_pending_questions(all_questions, existing_df)
        done_count = total - len(pending_questions)

        if done_count > 0:
            print(f"\n   ⏩ Đã chạy: {done_count}/{total} câu — bỏ qua")

        if len(pending_questions) == 0:
            print(f"   ✅ Mode {mode.upper()} đã hoàn thành — không cần chạy thêm")
            all_mode_results[mode] = existing_df
            continue

        print(f"   🆕 Cần chạy thêm: {len(pending_questions)} câu")

        # Query LightRAG cho các câu chưa chạy
        new_questions = []
        new_answers = []
        new_contexts_list = []
        new_ground_truths = []

        print(f"\n🔄 Querying LightRAG (mode: {mode})...")
        query_start = time.time()

        for i, idx in enumerate(pending_indices):
            question = all_questions[idx]
            ground_truth = all_ground_truths[idx]

            print(f"  [{done_count + i + 1}/{total}] {question[:65]}...")

            answer, contexts = query_lightrag(question, mode)

            new_questions.append(question)
            new_answers.append(answer)
            new_contexts_list.append(contexts)
            new_ground_truths.append(ground_truth)

        query_time = time.time() - query_start
        print(f"\n✅ Query xong {len(new_questions)} câu mới trong {query_time:.1f}s")

        # Chạy RAGAS evaluation theo batch
        print(f"\n🔬 Chạy RAGAS evaluation ({mode}) — {len(new_questions)} câu mới...")
        print(f"   LLM Judge: {EVAL_LLM_MODEL}")
        print(f"   Embedding: {EMBEDDING_MODEL} (Ollama)")

        eval_start = time.time()
        new_result_dfs = []

        # Chia thành batch để tránh OOM và lưu trung gian
        for batch_start in range(0, len(new_questions), EVAL_BATCH_SIZE):
            batch_end = min(batch_start + EVAL_BATCH_SIZE, len(new_questions))
            batch_num = batch_start // EVAL_BATCH_SIZE + 1
            total_batches = (len(new_questions) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE

            print(f"\n   📦 Batch {batch_num}/{total_batches} (câu {batch_start+1}–{batch_end})...")

            batch_q = new_questions[batch_start:batch_end]
            batch_a = new_answers[batch_start:batch_end]
            batch_c = new_contexts_list[batch_start:batch_end]
            batch_g = new_ground_truths[batch_start:batch_end]

            try:
                batch_df = run_ragas_evaluation(batch_q, batch_a, batch_c, batch_g)

                # Thêm cột phụ
                batch_df.insert(0, "question_text", batch_q)
                batch_df["mode"] = mode

                # Tính RAGAS score
                metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
                valid_cols = [c for c in metric_cols if c in batch_df.columns]
                batch_df["ragas_score"] = batch_df[valid_cols].mean(axis=1)

                new_result_dfs.append(batch_df)

                # Lưu trung gian: ghép kết quả cũ + mới (đến thời điểm hiện tại)
                partial_new = pd.concat(new_result_dfs, ignore_index=True) if new_result_dfs else pd.DataFrame()
                if existing_df is not None and len(existing_df) > 0:
                    combined = pd.concat([existing_df, partial_new], ignore_index=True)
                else:
                    combined = partial_new
                all_mode_results[mode] = combined

                # Lưu file trung gian (bảo vệ dữ liệu)
                save_intermediate_results(all_mode_results)
                print(f"   💾 Đã lưu trung gian ({len(combined)} câu tổng cộng cho {mode})")

            except Exception as e:
                print(f"   ❌ Lỗi RAGAS batch {batch_num}: {e}")
                print(f"   ⚠️ Bỏ qua batch này, tiếp tục...")
                continue

        eval_time = time.time() - eval_start

        # Ghép toàn bộ kết quả: cũ + mới
        if new_result_dfs:
            new_results_df = pd.concat(new_result_dfs, ignore_index=True)
        else:
            new_results_df = pd.DataFrame()

        if existing_df is not None and len(existing_df) > 0:
            combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        else:
            combined_df = new_results_df

        all_mode_results[mode] = combined_df

        # In kết quả mode này (trên TOÀN BỘ dữ liệu cũ + mới)
        metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        print(f"\n📈 Kết quả {mode.upper()} ({len(combined_df)} câu tổng cộng):")
        for col in metric_cols:
            if col in combined_df.columns:
                print(f"   {col:<25s}: {combined_df[col].mean():.4f}")
        if "ragas_score" in combined_df.columns:
            print(f"   {'RAGAS Score (avg)':<25s}: {combined_df['ragas_score'].mean():.4f}")
        if len(new_results_df) > 0:
            print(f"   ⏱️ Query: {query_time:.1f}s | Eval: {eval_time:.1f}s (cho {len(new_results_df)} câu mới)")

    # ==================== SO SÁNH ====================
    print(f"\n{'=' * 65}")
    print(f"📊 SO SÁNH CÁC MODE: {', '.join(m.upper() for m in MODES)}")
    print(f"{'=' * 65}")

    # Kiểm tra tất cả modes đã có dữ liệu
    missing_modes = [m for m in MODES if m not in all_mode_results or len(all_mode_results[m]) == 0]
    if missing_modes:
        print(f"⚠️ Các mode chưa có dữ liệu: {', '.join(missing_modes)}")
        print("   Không thể so sánh. Hãy chạy lại để hoàn thành.")
        return

    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]
    col_w = max(10, max(len(m) for m in MODES) + 2)
    header = f"  {'Metric':<25s}" + "".join(f"{m.upper():>{col_w}s}" for m in MODES) + f"  {'Winner':>10s}"
    print(header)
    print(f"  {'─' * (25 + col_w * len(MODES) + 12)}")

    for col in metric_cols:
        means = {m: all_mode_results[m][col].mean() for m in MODES}
        best_m = max(means, key=means.get)
        winner = best_m.upper() if len(set(means.values())) > 1 else "TIE"
        label = col.replace("_", " ").title()
        if col == "ragas_score":
            print(f"  {'─' * (25 + col_w * len(MODES) + 12)}")
            label = "RAGAS Score (avg)"
        line = f"  {label:<25s}" + "".join(f"{means[m]:>{col_w}.4f}" for m in MODES) + f"  {winner:>10s}"
        print(line)

    print(f"{'=' * 65}")

    # In số câu mỗi mode
    for mode in MODES:
        print(f"   📊 {mode.upper()}: {len(all_mode_results[mode])} câu")

    # ==================== LƯU KẾT QUẢ CUỐI CÙNG ====================
    # Tạo summary sheet
    summary_df = create_summary_sheet(all_mode_results)

    # Excel: 1 sheet / mode (tên sheet ≤ 31 ký tự) + Summary
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for mode in MODES:
            sheet = mode.capitalize()[:31]
            all_mode_results[mode].to_excel(writer, sheet_name=sheet, index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"\n💾 Kết quả đã lưu: {OUTPUT_FILE}")
    for mode in MODES:
        print(f"   📄 Sheet '{mode.capitalize()[:31]}' — {len(all_mode_results[mode])} câu (mode {mode})")
    print("   📄 Sheet 'Summary' — tổng hợp so sánh các mode")


if __name__ == "__main__":
    main()
