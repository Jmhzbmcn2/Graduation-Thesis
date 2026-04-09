"""
RAGAS Evaluation — So sánh Naive vs Mix (500 test cases từ 500_cases.csv)
Khóa Luận Tốt Nghiệp

Đánh giá 4 metrics RAGAS:
  - Faithfulness: Câu trả lời có đúng theo context không?
  - Answer Relevancy: Câu trả lời có liên quan đến câu hỏi không?
  - Context Recall: Context truy xuất có đầy đủ không?
  - Context Precision: Context có chính xác không?

Cấu hình local server:
  - LLM Judge  : vLLM (Qwen2.5-14B-Instruct-AWQ) tại http://localhost:8000
  - Embedding  : Ollama (nomic-embed-text) tại http://localhost:11434
  - LightRAG   : http://localhost:9621

Output: eval_ragas_naive_mix_500cases.xlsx (3 sheets: Naive, Mix, Summary)

Cách dùng:
  python testcase/eval_ragas_naive_mix.py
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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",        "nomic-embed-text")
EMBEDDING_HOST  = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")

# Số test case (None = dùng tất cả)
TEST_LIMIT = 5   # Đặt số nguyên (ví dụ 50) để giới hạn số câu hỏi

# Giới hạn độ dài context (ký tự) để tránh vượt token limit của LLM Judge
# Qwen2.5-14B-Instruct-AWQ có context window 8192 tokens
# ~4 ký tự ≈ 1 token → 4000 ký tự ≈ 1000 tokens
MAX_CONTEXT_CHARS = None

# Modes cần đánh giá
MODES = ["naive", "mix"]
# ===========================================================


def clean_answer(answer: str) -> str:
    """Loại bỏ phần ### References khỏi answer"""
    return re.split(r"\n*###\s*References", answer, maxsplit=1)[0].strip()


def extract_chunks_from_context(raw_context: str) -> str:
    """
    Trích xuất CHỈ phần Document Chunks từ context trả về của LightRAG.

    Context mix mode có cấu trúc:
      1. Knowledge Graph Data (Entity)      ← entity JSON → BỎ QUA
      2. Knowledge Graph Data (Relationship) ← relation JSON → BỎ QUA
      3. Document Chunks                     ← text chunks → LẤY CÁI NÀY
      4. Reference Document List             ← file list → BỎ QUA

    Lý do: RAGAS đánh giá trên văn bản xuôi. Entity/Relationship dạng JSON
    gây nhiễu → RAGAS chấm Faithfulness/Recall sai lệch nghiêm trọng.
    """
    # Tìm phần Document Chunks
    chunk_start = raw_context.find("Document Chunks")
    if chunk_start == -1:
        # Không tìm thấy cấu trúc → trả về nguyên gốc (naive mode)
        return raw_context

    # Tìm phần Reference Document List (phần sau Document Chunks)
    ref_start = raw_context.find("Reference Document List", chunk_start)

    if ref_start != -1:
        chunk_section = raw_context[chunk_start:ref_start]
    else:
        chunk_section = raw_context[chunk_start:]

    # Parse JSON content entries và lấy field "content" ra thành text xuôi
    extracted_texts = []
    try:
        import json as _json
        # Tìm các JSON object trong block ```json ... ```
        json_match = re.search(r'```json\s*\n(.*?)```', chunk_section, re.DOTALL)
        if json_match:
            json_lines = json_match.group(1).strip().split('\n')
            for line in json_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                    content = obj.get("content", "")
                    if content:
                        extracted_texts.append(content.strip())
                except _json.JSONDecodeError:
                    continue
    except Exception:
        pass

    if extracted_texts:
        return "\n\n".join(extracted_texts)

    # Fallback: trả về nguyên phần chunk section nếu không parse được JSON
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
    base = {"query": question, "mode": mode, "stream": False, "top_k": 5}

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

    # LLM Judge — vLLM local (OpenAI-compatible)
    llm = LangchainLLMWrapper(
        langchain_llm=ChatOpenAI(
            model=EVAL_LLM_MODEL,
            api_key=EVAL_LLM_API_KEY,
            base_url=EVAL_LLM_BASE_URL,
            temperature=0,
            max_tokens=1024,
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
    """Tạo sheet Summary tổng hợp kết quả 2 mode"""
    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]
    metric_labels = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision", "RAGAS Score"]

    rows = []
    for label, col in zip(metric_labels, metric_cols):
        row = {"Metric": label}
        for mode in MODES:
            df = all_mode_results[mode]
            row[f"{mode.capitalize()}_Mean"] = round(df[col].mean(), 4)
            row[f"{mode.capitalize()}_Median"] = round(df[col].median(), 4)
            row[f"{mode.capitalize()}_Std"] = round(df[col].std(), 4)
            row[f"{mode.capitalize()}_Min"] = round(df[col].min(), 4)
            row[f"{mode.capitalize()}_Max"] = round(df[col].max(), 4)

        # Determine winner
        naive_mean = all_mode_results["naive"][col].mean()
        mix_mean = all_mode_results["mix"][col].mean()
        if mix_mean > naive_mean:
            row["Winner"] = "Mix"
        elif naive_mean > mix_mean:
            row["Winner"] = "Naive"
        else:
            row["Winner"] = "Tie"

        row["Difference"] = round(mix_mean - naive_mean, 4)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df


def main():
    print("=" * 65)
    print("📊 RAGAS Evaluation — Naive vs Mix")
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

    total = len(df)
    all_mode_results = {}

    for mode in MODES:
        print(f"\n{'=' * 65}")
        print(f"🔍 ĐÁNH GIÁ MODE: {mode.upper()} ({total} câu hỏi)")
        print(f"{'=' * 65}")

        questions = []
        answers = []
        contexts_list = []
        ground_truths = []

        # Query LightRAG
        print(f"\n🔄 Querying LightRAG (mode: {mode})...")
        query_start = time.time()

        for idx, row in df.iterrows():
            question = str(row.get("question", row.get("Question", "")))
            ground_truth = str(row.get("answer", row.get("Answer", "")))

            if not question or question == "nan":
                continue

            print(f"  [{idx+1}/{total}] {question[:65]}...")

            answer, contexts = query_lightrag(question, mode)

            questions.append(question)
            answers.append(answer)
            contexts_list.append(contexts)
            ground_truths.append(ground_truth)

        query_time = time.time() - query_start
        print(f"\n✅ Query xong {len(questions)} câu trong {query_time:.1f}s")

        # Chạy RAGAS
        print(f"\n🔬 Chạy RAGAS evaluation ({mode})...")
        print(f"   LLM Judge: {EVAL_LLM_MODEL}")
        print(f"   Embedding: {EMBEDDING_MODEL} (Ollama)")

        eval_start = time.time()
        results_df = run_ragas_evaluation(questions, answers, contexts_list, ground_truths)
        eval_time = time.time() - eval_start

        # Thêm cột phụ
        results_df.insert(0, "question_text", questions)
        results_df["mode"] = mode

        # Tính RAGAS score
        metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        valid_cols = [c for c in metric_cols if c in results_df.columns]
        results_df["ragas_score"] = results_df[valid_cols].mean(axis=1)

        all_mode_results[mode] = results_df

        # In kết quả mode này
        print(f"\n📈 Kết quả {mode.upper()}:")
        for col in metric_cols:
            if col in results_df.columns:
                print(f"   {col:<25s}: {results_df[col].mean():.4f}")
        print(f"   {'RAGAS Score (avg)':<25s}: {results_df['ragas_score'].mean():.4f}")
        print(f"   ⏱️ Query: {query_time:.1f}s | Eval: {eval_time:.1f}s")

    # ==================== SO SÁNH ====================
    print(f"\n{'=' * 65}")
    print("📊 SO SÁNH NAIVE vs MIX")
    print(f"{'=' * 65}")

    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]
    print(f"  {'Metric':<25s} {'Naive':>8s}  {'Mix':>8s}  {'Winner':>8s}")
    print(f"  {'─' * 55}")

    for col in metric_cols:
        n = all_mode_results["naive"][col].mean()
        m = all_mode_results["mix"][col].mean()
        winner = "MIX" if m > n else ("NAIVE" if n > m else "TIE")
        label = col.replace("_", " ").title()
        if col == "ragas_score":
            print(f"  {'─' * 55}")
            label = "RAGAS Score (avg)"
        print(f"  {label:<25s} {n:>8.4f}  {m:>8.4f}  {winner:>8s}")

    print(f"{'=' * 65}")

    # ==================== LƯU KẾT QUẢ ====================
    # Tạo summary sheet
    summary_df = create_summary_sheet(all_mode_results)

    # Lưu vào file Excel với 3 sheets
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        all_mode_results["naive"].to_excel(writer, sheet_name="Naive", index=False)
        all_mode_results["mix"].to_excel(writer, sheet_name="Mix", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"\n💾 Kết quả đã lưu: {OUTPUT_FILE}")
    print("   📄 Sheet 'Naive'   — Chi tiết từng câu hỏi (mode naive)")
    print("   📄 Sheet 'Mix'     — Chi tiết từng câu hỏi (mode mix)")
    print("   📄 Sheet 'Summary' — Tổng hợp so sánh 2 mode")


if __name__ == "__main__":
    main()
