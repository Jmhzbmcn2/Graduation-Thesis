"""
RAGAS Evaluation — So sánh Focused vs IVF_Focused
Khóa Luận Tốt Nghiệp

So sánh các mode: focused vs ivf_focused trên các tiêu chí:
  1. Số phép tính (gián tiếp qua graph_search_ms và log server)
  2. Latency retrieval (thời gian truy xuất)
  3. Latency generation (thời gian sinh)
  4. RAGAS: Faithfulness, Answer Relevancy, Context Recall, Context Precision

Cấu hình:
  - LLM: Qwen2.5-14B-Instruct-AWQ (vLLM local)
  - Embedding: embeddinggemma:300m (Ollama local)
  - Input: 300_case_random.csv
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import re
import time
import warnings
import pandas as pd
import requests

from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=False)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Unexpected type for token usage.*", category=UserWarning)

# ======================== CẤU HÌNH ========================
LIGHTRAG_URL = "http://localhost:9621"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(_SCRIPT_DIR, "300_case_random.csv")
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "eval_ragas_ivf_vs_focused.xlsx")

EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "qwen/qwen3-30b-a3b-instruct-2507")
EVAL_LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", os.getenv("LLM_BINDING_API_KEY", "EMPTY"))
EVAL_LLM_HOST = os.getenv("EVAL_LLM_BINDING_HOST", "https://openrouter.ai/api/v1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
EMBEDDING_HOST  = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")

TEST_LIMIT = 3  # Đổi thành None để chạy toàn bộ
MODES = ["focused", "ivf_focused"]

# ===========================================================


def estimate_tokens(text: str) -> int:
    if not text: return 0
    return max(1, len(text) // 4)

def clean_answer(answer: str) -> str:
    return re.split(r"\n*###\s*References", answer, maxsplit=1)[0].strip()

def extract_chunks_from_context(raw_context: str) -> str:
    # Rút gọn context extraction để tránh max context size errors
    return raw_context[:4000] if raw_context else ""

def query_lightrag_with_timing(question: str, mode: str, retries: int = 3):
    base = {
        "query": question,
        "mode": mode,
        "stream": False,
        "top_k": 10,
        "include_context": True,
    }

    for attempt in range(retries):
        try:
            client_start = time.perf_counter()
            resp = requests.post(f"{LIGHTRAG_URL}/query", json=base, timeout=180)
            client_elapsed_ms = (time.perf_counter() - client_start) * 1000
            resp.raise_for_status()
            resp_json = resp.json()

            answer = clean_answer(resp_json.get("response", ""))
            
            timing = resp_json.get("timing") or {}
            graph_search_ms = timing.get("graph_search_ms", 0)
            retrieval_latency_ms = timing.get("retrieval_ms", 0)
            generation_latency_ms = timing.get("generation_ms", 0)
            total_latency_ms = timing.get("total_ms", 0)
            vector_comparisons = timing.get("vector_comparisons", 0)
            total_vectors = timing.get("total_vectors", 0)

            full_context = resp_json.get("context", "")
            contexts = [extract_chunks_from_context(full_context)] if full_context else ["No context retrieved"]

            token_counts = resp_json.get("token_counts") or {}
            input_tokens = token_counts.get("input_tokens", estimate_tokens(question + "\n" + (full_context or "")))

            metrics = {
                "graph_search_ms": round(graph_search_ms, 2),
                "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                "generation_latency_ms": round(generation_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
                "vector_comparisons": vector_comparisons,
                "total_vectors": total_vectors,
                "input_tokens": input_tokens,
            }
            return answer, contexts, metrics

        except Exception as e:
            print(f"    Loi (lan {attempt+1}/{retries}): {e}")
            time.sleep(3)
            
    return "Error", ["No context"], {"graph_search_ms":0, "retrieval_latency_ms":0, "generation_latency_ms":0, "total_latency_ms":0, "input_tokens":0}


def run_ragas_evaluation(questions, answers, contexts_list, ground_truths):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings

    llm = LangchainLLMWrapper(
        langchain_llm=ChatOpenAI(
            model=EVAL_LLM_MODEL,
            api_key=EVAL_LLM_API_KEY,
            base_url=EVAL_LLM_HOST if EVAL_LLM_HOST else None,
            temperature=0.0,
            max_tokens=4096,
        ), bypass_n=True
    )
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key="ollama",
            base_url=f"{EMBEDDING_HOST}/v1" if EMBEDDING_HOST else None,
            check_embedding_ctx_length=False,
        )
    )
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })
    
    from ragas.run_config import RunConfig
    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()],
        llm=llm, embeddings=emb,
        run_config=RunConfig(max_workers=4),
    )
    return results.to_pandas()


def main():
    print("=" * 70)
    print(f"Danh gia IVF Clustering: Focused vs IVF_Focused")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)
    if TEST_LIMIT: df = df.head(TEST_LIMIT)

    all_questions = df["question"].tolist() if "question" in df.columns else df["Question"].tolist()
    all_ground_truths = df["answer"].tolist() if "answer" in df.columns else df["Answer"].tolist()

    all_mode_results = {}

    for mode in MODES:
        print(f"\n Running Mode: {mode.upper()} ({len(all_questions)} queries)...")
        m_answers, m_contexts, m_metrics = [], [], []

        for i, (q, gt) in enumerate(zip(all_questions, all_ground_truths)):
            ans, ctx, metrics = query_lightrag_with_timing(q, mode)
            m_answers.append(ans)
            m_contexts.append(ctx)
            m_metrics.append(metrics)
            print(f"  [{i+1}/{len(all_questions)}] {q[:50]}... | Graph: {metrics['graph_search_ms']}ms | Total: {metrics['total_latency_ms']}ms")

        print(f" Running RAGAS for {mode.upper()}...")
        ragas_df = run_ragas_evaluation(all_questions, m_answers, m_contexts, all_ground_truths)
        
        # Merge metrics
        timing_df = pd.DataFrame(m_metrics)
        for col in timing_df.columns:
            ragas_df[col] = timing_df[col]
            
        ragas_df.insert(0, "question", all_questions)
        all_mode_results[mode] = ragas_df

    # Save to Excel
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for mode, df_res in all_mode_results.items():
            df_res.to_excel(writer, sheet_name=mode.upper(), index=False)

    # Print summary
    print("\n" + "="*70)
    print(" TONG KET SO SANH: FOCUSED vs IVF_FOCUSED")
    print("="*70)
    
    metrics = ["vector_comparisons", "total_vectors", "graph_search_ms", "retrieval_latency_ms", "total_latency_ms", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    
    print(f"{'Metric':<25} | {'Focused':<12} | {'IVF_Focused':<12} | {'Difference'}")
    print("-" * 70)
    
    for metric in metrics:
        v_foc = all_mode_results["focused"][metric].mean()
        v_ivf = all_mode_results["ivf_focused"][metric].mean()
        
        if "ms" in metric:
            diff = f"{(v_ivf - v_foc) / max(v_foc, 1) * 100:+.1f}%"
        elif "vectors" in metric or "comparisons" in metric:
            diff = f"{(v_ivf - v_foc) / max(v_foc, 1) * 100:+.1f}%"
        else:
            diff = f"{v_ivf - v_foc:+.4f}"
            
        print(f"{metric:<25} | {v_foc:<12.2f} | {v_ivf:<12.2f} | {diff}")

if __name__ == "__main__":
    main()
