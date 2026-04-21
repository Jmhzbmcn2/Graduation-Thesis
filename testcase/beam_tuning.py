"""
Beam Search Tuning Script — Chạy 1 câu hỏi với nhiều bộ tham số beam
Khóa Luận Tốt Nghiệp

Mục đích: Nhanh chóng thử nghiệm các tổ hợp tham số beam search
          và so sánh kết quả RAGAS + timing trên cùng 1 câu hỏi.

Cách dùng:
  1. Sửa QUESTION_INDEX hoặc CUSTOM_QUESTION ở phần CẤU HÌNH bên dưới
  2. Sửa CONFIGS — mỗi dict là 1 bộ tham số cần thử
  3. Chạy: python testcase/beam_tuning.py
  4. Kết quả sẽ in ra bảng so sánh + lưu file Excel

Lưu ý:
  - pruning_threshold và BEAM_MAX_ANCHOR_K là tham số server-side,
    phải sửa trong base.py / operate.py rồi restart server.
  - Script này chỉ tune các tham số API-level: beam_width, beam_max_depth,
    top_k, chunk_top_k.
"""
import os
import re
import time
import json
import warnings
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=False)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper.*")
warnings.filterwarnings("ignore", message=".*token usage.*")

# ======================== CẤU HÌNH ========================
LIGHTRAG_URL = "http://localhost:9621"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(_SCRIPT_DIR, "500_cases_part2.csv")
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "beam_tuning_results.xlsx")

# RAGAS LLM Judge — OpenRouter
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "qwen/qwen3-30b-a3b-instruct-2507")
EVAL_LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", os.getenv("EVAL_LLM_BINDING_API_KEY", ""))
EVAL_LLM_HOST = os.getenv("EVAL_LLM_BINDING_HOST", "https://openrouter.ai/api/v1")

# RAGAS Embedding — Ollama local
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
EMBEDDING_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")

# ======================== CÂU HỎI ========================
# Chọn 1 mảng các câu hỏi từ CSV (0-indexed)
QUESTION_INDICES = [5,6,11,12,13]  # Có thể truyền nhiều câu: [6, 7, 8]
CUSTOM_QUESTION = None  # Đặt string ở đây để dùng câu hỏi tự nhập
CUSTOM_GROUND_TRUTH = None  # Ground truth cho câu hỏi tự nhập

# ======================== CÁC BỘ THAM SỐ CẦN THỬ ========================
# Mỗi dict = 1 lần chạy beam. Sửa thoải mái, thêm/bớt tuỳ ý.
CONFIGS = [
    {
        "name": "baseline_vdb_only",
        "beam_width": 10,
        "beam_max_depth": 2,
        "top_k": 10,
        "chunk_top_k": 5,
        "pruning_threshold": 0.3,
        "beam_max_anchor": 10,
        "anchor_alpha": 1.0,   # 1.0 = VDB-only (no BM25)
        "chunk_alpha": 1.0,    # 1.0 = VDB-only (no BM25)
    },
    {
        "name": "bm25_hybrid_balanced",
        "beam_width": 10,
        "beam_max_depth": 1,
        "top_k": 10,
        "chunk_top_k": 5,
        "pruning_threshold": 0.35,
        "beam_max_anchor": 10,
        "anchor_alpha": 0.5,   # 50/50 Dense + BM25
        "chunk_alpha": 0.7,    # 70% Dense, 30% BM25
    },
]
# ===========================================================


def clean_answer(answer: str) -> str:
    return re.split(r"\n*###\s*References", answer, maxsplit=1)[0].strip()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def extract_chunks_from_context(raw_context: str) -> str:
    """Trích xuất context từ LightRAG cho RAGAS evaluation."""
    chunk_start = raw_context.find("Document Chunks")
    if chunk_start == -1:
        return raw_context

    ref_start = raw_context.find("Reference Document List", chunk_start)
    chunk_section = raw_context[chunk_start:ref_start] if ref_start != -1 else raw_context[chunk_start:]

    chunk_texts = []
    json_match = re.search(r'```json\s*\n(.*?)```', chunk_section, re.DOTALL)
    if json_match:
        for line in json_match.group(1).strip().split('\n'):
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    content = obj.get("content", "")
                    if content:
                        chunk_texts.append(content.strip())
                except json.JSONDecodeError:
                    pass

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
                        obj = json.loads(line)
                        name = obj.get("entity", "")
                        desc = obj.get("description", "")
                        if name and desc:
                            graph_texts.append(f"Thực thể [{name}] có thông tin: {desc}")
                    except json.JSONDecodeError:
                        pass

    if rel_start != -1:
        rel_section = raw_context[rel_start:chunk_start]
        json_match = re.search(r'```json\s*\n(.*?)```', rel_section, re.DOTALL)
        if json_match:
            for line in json_match.group(1).strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        e1 = obj.get("entity1", "")
                        e2 = obj.get("entity2", "")
                        desc = obj.get("description", "")
                        if e1 and e2 and desc:
                            graph_texts.append(f"Mối quan hệ giữa [{e1}] và [{e2}] là: {desc}")
                    except json.JSONDecodeError:
                        pass

    result_parts = []
    if chunk_texts:
        result_parts.append("\n\n".join(chunk_texts))
    if graph_texts:
        result_parts.append("\n".join(graph_texts))

    if result_parts:
        return "\n\n".join(result_parts)
    return chunk_section.strip()


def query_beam(question: str, config: dict) -> dict:
    """Query LightRAG beam mode với tham số cụ thể."""
    payload = {
        "query": question,
        "mode": "beam",
        "stream": False,
        "top_k": config["top_k"],
        "beam_width": config["beam_width"],
        "beam_max_depth": config["beam_max_depth"],
        "chunk_top_k": config["chunk_top_k"],
        "pruning_threshold": config["pruning_threshold"],
        "beam_max_anchor": config["beam_max_anchor"],
        "anchor_alpha": config.get("anchor_alpha", 0.5),
        "chunk_alpha": config.get("chunk_alpha", 0.7),
        "include_context": True,
    }

    try:
        resp = requests.post(f"{LIGHTRAG_URL}/query", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        answer = clean_answer(data.get("response", ""))

        timing = data.get("timing") or {}
        token_counts = data.get("token_counts") or {}

        full_context = data.get("context", "")
        if full_context:
            full_context = extract_chunks_from_context(full_context)
        contexts = [full_context] if full_context else ["No context retrieved"]

        input_tokens = token_counts.get("input_tokens", 0)
        output_tokens = token_counts.get("output_tokens", 0)
        if input_tokens == 0:
            input_tokens = estimate_tokens(question + "\n" + (full_context or ""))
        if output_tokens == 0:
            output_tokens = estimate_tokens(answer)

        return {
            "answer": answer,
            "contexts": contexts,
            "keyword_extraction_ms": timing.get("keyword_extraction_ms", 0),
            "graph_search_ms": timing.get("graph_search_ms", 0),
            "retrieval_ms": timing.get("retrieval_ms", 0),
            "generation_ms": timing.get("generation_ms", 0),
            "total_ms": timing.get("total_ms", 0),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": None,
        }
    except Exception as e:
        return {
            "answer": f"Error: {e}",
            "contexts": ["No context"],
            "keyword_extraction_ms": 0,
            "graph_search_ms": 0,
            "retrieval_ms": 0,
            "generation_ms": 0,
            "total_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": str(e),
        }


def run_ragas_single(question: str, answer: str, contexts: list, ground_truth: str) -> dict:
    """Chạy RAGAS evaluation trên 1 câu hỏi."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.run_config import RunConfig

    llm = LangchainLLMWrapper(
        langchain_llm=ChatOpenAI(
            model=EVAL_LLM_MODEL,
            api_key=EVAL_LLM_API_KEY,
            base_url=EVAL_LLM_HOST,
            temperature=0.0,
            max_tokens=8192,
            max_retries=3,
            timeout=180.0,
        ),
        bypass_n=True,
    )

    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key="ollama",
            base_url=f"{EMBEDDING_HOST}/v1",
            check_embedding_ctx_length=False,
            max_retries=5,
            timeout=60.0,
        )
    )

    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth],
    })

    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()],
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=4, max_retries=10),
    )

    df = results.to_pandas()
    return {
        "faithfulness": float(df["faithfulness"].iloc[0]),
        "answer_relevancy": float(df["answer_relevancy"].iloc[0]),
        "context_recall": float(df["context_recall"].iloc[0]),
        "context_precision": float(df["context_precision"].iloc[0]),
    }


def main():
    print("=" * 70)
    print("🔧 Beam Search Tuning — KLTN (Multi-Questions)")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)
    # Lấy chính xác cột 'question' và 'answer'/'ground_truth'
    q_col = [c for c in df.columns if c.strip().lower() in ["question", "query"]]
    gt_col = [c for c in df.columns if c.strip().lower() in ["ground_truth", "answer", "truth"]]
    
    questions_data = []
    if CUSTOM_QUESTION:
        questions_data.append({"q": CUSTOM_QUESTION, "gt": CUSTOM_GROUND_TRUTH or ""})
        print(f"\n📝 Sử dụng 1 câu hỏi tự nhập.")
    else:
        for idx in QUESTION_INDICES:
            row = df.iloc[idx]
            q = str(row[q_col[0]]) if q_col else str(row.iloc[0])
            gt = str(row[gt_col[0]]) if gt_col else ""
            questions_data.append({"q": q, "gt": gt, "idx": idx})
        print(f"\n📝 Đang test trên {len(questions_data)} câu hỏi: {QUESTION_INDICES}")

    print(f"\n📝 Questions: {questions_data}")
    # ======================== TIẾN HÀNH ========================
    print(f"\n🔬 Số bộ tham số cần thử: {len(CONFIGS)}")

    all_results = []       # Lưu chi tiết từng câu
    config_summary = []    # Lưu trung bình của từng config

    for i, config in enumerate(CONFIGS):
        name = config["name"]
        print(f"\n{'─' * 70}")
        print(f"▶ [{i+1}/{len(CONFIGS)}] Config: {name}")
        print(f"  beam_width={config['beam_width']}, max_depth={config['beam_max_depth']}, top_k={config['top_k']}, "
              f"chunk_top_k={config['chunk_top_k']}, pruning={config['pruning_threshold']}, max_anchor={config['beam_max_anchor']}")

        config_metrics = []

        for q_idx, q_data in enumerate(questions_data):
            question = q_data["q"]
            ground_truth = q_data["gt"]
            print(f"\n  ➤ Câu {q_idx+1}/{len(questions_data)}: {question[:60]}...")
            
            # Query
            result = query_beam(question, config)
            if result["error"]:
                print(f"    ❌ Error: {result['error']}")
                continue

            print(f"    ✅ Answer: {result['answer'][:60]}...")
            
            # RAGAS
            try:
                ragas = run_ragas_single(question, result["answer"], result["contexts"], ground_truth)
                ragas_score = sum(ragas.values()) / 4
            except Exception as e:
                print(f"    ❌ RAGAS error: {e}")
                ragas = {"faithfulness": 0, "answer_relevancy": 0, "context_recall": 0, "context_precision": 0}
                ragas_score = 0

            # Lưu lại metric của câu này
            metrics = {
                **ragas,
                "ragas_score": ragas_score,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "keyword_extraction_ms": result["keyword_extraction_ms"],
                "graph_search_ms": result["graph_search_ms"],
                "retrieval_ms": result["retrieval_ms"],
                "generation_ms": result["generation_ms"],
                "total_ms": result["total_ms"],
            }
            config_metrics.append(metrics)
            
            # Lưu detail
            all_results.append({
                "config": name,
                "question_idx": q_data.get("idx", "custom"),
                "question": question,
                **metrics,
                "answer": result["answer"]
            })

        # Tính trung bình cho config này
        if config_metrics:
            avg_metrics = {k: sum(m[k] for m in config_metrics) / len(config_metrics) for k in config_metrics[0].keys()}
            config_summary.append({"config": name, **avg_metrics})

    # ======================== BẢNG SO SÁNH (TRUNG BÌNH) ========================
    print(f"\n{'=' * 70}")
    print(f"📊 BẢNG SO SÁNH TRUNG BÌNH ({len(questions_data)} câu hỏi)")
    print(f"{'=' * 70}")

    if not config_summary:
        print("  Không có kết quả nào!")
        return

    col_w = 14
    header = f"  {'Metric':<25s}" + "".join(f"{r['config']:>{col_w}s}" for r in config_summary)
    print(header)
    print(f"  {'─' * (25 + col_w * len(config_summary))}")

    metrics_list = [
        ("Faithfulness", "faithfulness"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Context Recall", "context_recall"),
        ("Context Precision", "context_precision"),
        ("RAGAS Score", "ragas_score"),
        ("Input Tokens", "input_tokens"),
        ("Output Tokens", "output_tokens"),
        ("KW Extract (ms)", "keyword_extraction_ms"),
        ("Graph Search (ms)", "graph_search_ms"),
        ("Retrieval (ms)", "retrieval_ms"),
        ("Generation (ms)", "generation_ms"),
        ("Total (ms)", "total_ms"),
    ]

    quality_metrics = {"faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"}

    for label, key in metrics_list:
        if key == "ragas_score":
            print(f"  {'─' * (25 + col_w * len(config_summary))}")

        vals = []
        for r in config_summary:
            v = r.get(key)
            if key in quality_metrics:
                vals.append(f"{v:.4f}")
            else:
                vals.append(f"{v:.1f}")

        row = f"  {label:<25s}" + "".join(f"{v:>{col_w}s}" for v in vals)
        print(row)

    # Lưu Excel
    try:
        with pd.ExcelWriter(OUTPUT_FILE) as writer:
            pd.DataFrame(config_summary).to_excel(writer, index=False, sheet_name="Summary (Average)")
            pd.DataFrame(all_results).to_excel(writer, index=False, sheet_name="Detailed Results")
        print(f"\n💾 Saved: {OUTPUT_FILE}")
    except Exception as e:
        print(f"\n⚠️ Cannot save Excel: {e}")

    print(f"\n✅ Done! Evaluated {len(CONFIGS)} configs across {len(questions_data)} questions.")


if __name__ == "__main__":
    main()
