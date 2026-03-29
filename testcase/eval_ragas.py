"""
RAGAS Evaluation — So sánh Naive vs Hybrid (50 câu hỏi)
Khóa Luận Tốt Nghiệp

Đánh giá 4 metrics RAGAS:
  - Faithfulness: Câu trả lời có đúng theo context không?
  - Answer Relevancy: Câu trả lời có liên quan đến câu hỏi không?
  - Context Recall: Context truy xuất có đầy đủ không?
  - Context Precision: Context có chính xác không?

Cách dùng:
  python testcase/eval_ragas.py
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
INPUT_FILE = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\50 testcase.xlsx"
OUTPUT_DIR = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\testcase"

# LLM Judge (Gemini via OpenAI-compatible endpoint)
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "gemini-2.0-flash")
EVAL_LLM_API_KEY = os.getenv("EVAL_LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
EVAL_LLM_BASE_URL = os.getenv("EVAL_LLM_BINDING_HOST")

# Embedding (Ollama local)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")

# Số test case chạy (None = chạy tất cả)
TEST_LIMIT = None

# Modes cần đánh giá
MODES = ["naive", "hybrid"]
# ===========================================================


def clean_answer(answer: str) -> str:
    """Loại bỏ phần ### References khỏi answer"""
    return re.split(r"\n*###\s*References", answer, maxsplit=1)[0].strip()


def query_lightrag(question: str, mode: str, retries: int = 3):
    """
    Query LightRAG: lấy answer + full context.
    - Call 1: normal query → answer
    - Call 2: only_need_context=True → full context (entities + relations + chunks)
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
    from langchain_openai import ChatOpenAI
    from langchain_ollama import OllamaEmbeddings
    from ragas.llms import LangchainLLMWrapper

    # LLM Judge
    llm_kwargs = {"model": EVAL_LLM_MODEL, "api_key": EVAL_LLM_API_KEY}
    if EVAL_LLM_BASE_URL:
        llm_kwargs["base_url"] = EVAL_LLM_BASE_URL
    llm = LangchainLLMWrapper(langchain_llm=ChatOpenAI(**llm_kwargs), bypass_n=True)

    # Embedding
    emb = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=EMBEDDING_HOST)

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


def main():
    print("=" * 65)
    print("📊 RAGAS Evaluation — Naive vs Hybrid")
    print("   Khóa Luận Tốt Nghiệp")
    print("=" * 65)

    # 1. Đọc test cases
    print(f"\n📂 Input: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)
    print(f"   Tìm thấy {len(df)} test cases")

    if TEST_LIMIT:
        df = df.head(TEST_LIMIT)
        print(f"   Giới hạn: {TEST_LIMIT} câu đầu")

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
    print("📊 SO SÁNH NAIVE vs HYBRID")
    print(f"{'=' * 65}")

    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]
    print(f"  {'Metric':<25s} {'Naive':>8s}  {'Hybrid':>8s}  {'Winner':>8s}")
    print(f"  {'─' * 55}")

    for col in metric_cols:
        n = all_mode_results["naive"][col].mean()
        h = all_mode_results["hybrid"][col].mean()
        winner = "HYBRID" if h > n else ("NAIVE" if n > h else "TIE")
        label = col.replace("_", " ").title()
        if col == "ragas_score":
            print(f"  {'─' * 55}")
            label = "RAGAS Score (avg)"
        print(f"  {label:<25s} {n:>8.4f}  {h:>8.4f}  {winner:>8s}")

    print(f"{'=' * 65}")

    # ==================== LƯU KẾT QUẢ ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Gộp kết quả 2 mode vào 1 file
    combined = pd.concat(all_mode_results.values(), ignore_index=True)
    output_file = os.path.join(OUTPUT_DIR, f"ragas_naive_vs_hybrid_{timestamp}.xlsx")
    combined.to_excel(output_file, index=False)
    print(f"\n💾 Kết quả đã lưu: {output_file}")


if __name__ == "__main__":
    main()
