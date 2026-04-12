"""
Test RAGAS trên 1 câu hỏi — So sánh Naive vs Hybrid
"""
import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv(override=False)

LIGHTRAG_URL = "http://localhost:9621"

# Câu hỏi test
QUESTION = "Béo phì làm tăng nguy cơ mắc bệnh gout ở độ tuổi nào?"
GROUND_TRUTH = "Béo phì làm tăng nguy cơ mắc bệnh gout ở người cao tuổi."


def clean_answer(answer: str) -> str:
    """Loại bỏ phần ### References khỏi answer"""
    return re.split(r"\n*###\s*References", answer, maxsplit=1)[0].strip()


def query_lightrag(question: str, mode: str):
    """Query LightRAG: lấy answer + full context"""
    base = {"query": question, "mode": mode, "stream": False, "top_k": 10}

    # Call 1: Lấy answer
    resp1 = requests.post(f"{LIGHTRAG_URL}/query", json={**base, "only_need_context": False}, timeout=180)
    raw_answer = resp1.json().get("response", "")
    answer = clean_answer(raw_answer)

    # Call 2: Lấy full context
    resp2 = requests.post(f"{LIGHTRAG_URL}/query", json={**base, "only_need_context": True}, timeout=180)
    context = resp2.json().get("response", "")

    return answer, [context] if context else ["No context"]


def run_ragas(question, answer, contexts, ground_truth):
    """Chạy RAGAS evaluation trên 1 câu"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
    from langchain_openai import ChatOpenAI
    from langchain_ollama import OllamaEmbeddings
    from ragas.llms import LangchainLLMWrapper

    llm_kwargs = {
        "model": os.getenv("EVAL_LLM_MODEL", "gemini-2.0-flash"),
        "api_key": os.getenv("EVAL_LLM_BINDING_API_KEY"),
    }
    base_url = os.getenv("EVAL_LLM_BINDING_HOST")
    if base_url:
        llm_kwargs["base_url"] = base_url

    llm = LangchainLLMWrapper(langchain_llm=ChatOpenAI(**llm_kwargs), bypass_n=True)
    emb = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m"),
        base_url=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
    )

    dataset = Dataset.from_dict({
        "question": [question], "answer": [answer],
        "contexts": [contexts], "ground_truth": [ground_truth],
    })

    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()],
        llm=llm, embeddings=emb,
    )
    return results.to_pandas().iloc[0]


def main():
    print("=" * 65)
    print("📊 RAGAS 1-Question Test — Naive vs Hybrid")
    print("=" * 65)
    print(f"📝 Question:     {QUESTION}")
    print(f"✅ Ground Truth: {GROUND_TRUTH}")
    print("=" * 65)

    all_results = {}

    for mode in ["naive", "hybrid"]:
        print(f"\n{'─' * 65}")
        print(f"🔍 Mode: {mode.upper()}")
        print(f"{'─' * 65}")

        # Query
        print(f"  🔹 Querying LightRAG ({mode})...")
        t0 = time.time()
        answer, contexts = query_lightrag(QUESTION, mode)
        query_time = time.time() - t0
        print(f"     Answer: {answer[:120]}...")
        print(f"     Context: {len(contexts[0])} chars")
        print(f"     Time: {query_time:.1f}s")

        # RAGAS
        print(f"  🔹 Running RAGAS evaluation...")
        t1 = time.time()
        scores = run_ragas(QUESTION, answer, contexts, GROUND_TRUTH)
        eval_time = time.time() - t1

        all_results[mode] = {
            "faith": scores["faithfulness"],
            "relevancy": scores["answer_relevancy"],
            "recall": scores["context_recall"],
            "precision": scores["context_precision"],
            "answer": answer[:100],
            "eval_time": eval_time,
        }

        print(f"     Faithfulness:      {scores['faithfulness']:.4f}")
        print(f"     Answer Relevancy:  {scores['answer_relevancy']:.4f}")
        print(f"     Context Recall:    {scores['context_recall']:.4f}")
        print(f"     Context Precision: {scores['context_precision']:.4f}")
        print(f"     RAGAS eval time:   {eval_time:.1f}s")

    # Bảng so sánh
    print(f"\n{'=' * 65}")
    print("📈 SO SÁNH NAIVE vs HYBRID")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<25s} {'Naive':>8s}  {'Hybrid':>8s}  {'Winner':>8s}")
    print(f"  {'─' * 55}")

    metrics = [
        ("Faithfulness", "faith"),
        ("Answer Relevancy", "relevancy"),
        ("Context Recall", "recall"),
        ("Context Precision", "precision"),
    ]

    naive_total, hybrid_total = 0, 0
    for name, key in metrics:
        n = all_results["naive"][key]
        h = all_results["hybrid"][key]
        naive_total += n
        hybrid_total += h
        winner = "HYBRID" if h > n else ("NAIVE" if n > h else "TIE")
        print(f"  {name:<25s} {n:>8.4f}  {h:>8.4f}  {winner:>8s}")

    n_avg = naive_total / 4
    h_avg = hybrid_total / 4
    winner = "HYBRID" if h_avg > n_avg else ("NAIVE" if n_avg > h_avg else "TIE")
    print(f"  {'─' * 55}")
    print(f"  {'RAGAS Score (avg)':<25s} {n_avg:>8.4f}  {h_avg:>8.4f}  {winner:>8s}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
