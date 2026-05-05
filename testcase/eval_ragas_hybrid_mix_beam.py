"""
RAGAS Evaluation — So sánh Hybrid vs Mix vs Beam vs Focused
Khóa Luận Tốt Nghiệp

So sánh các mode: hybrid, mix, beam, focused trên các tiêu chí:
  1. Số token truyền vào (input tokens - ước lượng từ question + context)
  2. Latency retrieval (thời gian truy xuất context từ LightRAG)
  3. Latency generation (thời gian sinh câu trả lời)
  4. Độ đo RAGAS: Faithfulness, Answer Relevancy, Context Recall, Context Precision

Cấu hình:
  - LLM Generator  : OpenRouter (qwen/qwen3-30b-a3b-instruct-2507)
  - Embedding      : Ollama (embeddinggemma:300m) tại http://localhost:11434
  - LightRAG API   : http://localhost:9621
  - RAGAS LLM Judge: OpenRouter (qwen/qwen3-30b-a3b-instruct-2507)
  - RAGAS Embedding: Ollama (embeddinggemma:300m)

Output: eval_ragas_beam_phase2.xlsx (1 sheet/mode + Summary)

Cách dùng:
  python testcase/eval_ragas_hybrid_mix_beam.py
"""
import os
import re
import time
import warnings
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=False)

warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*token usage.*", category=UserWarning)

# ======================== CẤU HÌNH ========================
LIGHTRAG_URL = "http://localhost:9621"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(_SCRIPT_DIR, "500-random-sample-of-test-data.csv")
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "eval_focused_chunk_only.xlsx")

# RAGAS LLM Judge — Local vLLM (OpenAI-compatible)
# Ưu tiên EVAL_LLM_MODEL/EVAL_LLM_BINDING_HOST từ .env; fallback về LLM_MODEL/LLM_BINDING_HOST
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ"))
EVAL_LLM_API_KEY = os.getenv("LLM_BINDING_API_KEY", "EMPTY")
EVAL_LLM_HOST = os.getenv("EVAL_LLM_BINDING_HOST", os.getenv("LLM_BINDING_HOST", "http://localhost:8000/v1"))

# RAGAS Embedding — Ollama local
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
EMBEDDING_HOST  = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")

# Số test case (None = tất cả, 3 = 3 câu đầu tiên)
TEST_LIMIT = 300

# Giới hạn độ dài context để tránh vượt token limit của LLM Judge
MAX_CONTEXT_CHARS = None

# Giới hạn riêng cho BEAM mode
# QUAN TRỌNG: KHÔNG cắt ngắn context beam, vì RAGAS cần thấy toàn bộ context
# mà LLM đã dùng để sinh answer. Cắt ngắn → Faithfulness = 0 giả.
BEAM_MAX_CONTEXT_CHARS = None  # None = không giới hạn (giống các mode khác)

# Modes cần đánh giá
# MODES = ["naive", "hybrid", "mix"]
# MODES = ["hybrid","mix","focused"]  # chạy riêng 1 mode
MODES = ["focused"]
# MODES = ["focused"]

# Batch size cho RAGAS
EVAL_BATCH_SIZE = 500

# ======================== CẤU HÌNH MODE-SPECIFIC ========================
# Beam search tối ưu: Narrow + Deep + Rerank
# • beam_width nhỏ (5) → chỉ theo nhánh tốt nhất, giảm context pollution
# • max_depth=2 → bắt entity 2-hop, bù cho width nhỏ
# • pruning cao (0.4) → cắt entity kem sớm hơn để giữ token budget ~15k
# • enable_rerank=True → lọc lại context sau beam bằng Vietnamese Reranker
BEAM_BEAM_WIDTH = 6          # từ 10 → narrow để tránh noise
BEAM_MAX_DEPTH = 2           # từ 1 → bắt entity 2-hop
BEAM_CHUNK_TOP_K = 15         # từ 15 → ít chunk rác hơn
BEAM_PRUNING_THRESHOLD = 0.2 # từ 0.25 → cắt noise sớm hơn
BEAM_ANCHOR_ALPHA = 0.7      # từ 0.7 → cân bằng BM25/dense
BEAM_CHUNK_ALPHA = 0.5       # từ 0.7 → tương tự
RELATED_CHUNK_NUMBER = 10     # từ 5 → giảm chunk kèm entity
# Rerank: dùng Vietnamese Reranker (AITeamVN/Vietnamese_Reranker) — bật khi server có RERANK_BINDING
BEAM_ENABLE_RERANK = True    # Bật rerank sau beam search (chunk-level)
BEAM_ENABLE_ANCHOR_RERANK = True  # Bật rerank ở bước anchor entity selection

# Focused search: per-anchor quota + semantic threshold + global search
# • focused_edge_quota: mỗi anchor chỉ đóng góp tối đa m cạnh liên quan nhất
# • focused_edge_threshold: ngưỡng cosine tối thiểu giữa query và edge
# • focused_alpha/beta: trọng số joint scoring Score(e) = α·Sim(Q,A) + β·Sim(Q,e)
# • focused_max_edges: tổng số cạnh tối đa sau khi gộp pool
# • focused_anchor_top_k: K per-branch per-keyword (True Hybrid Anchor)
# • focused_anchor_semantic_threshold: ngưỡng cosine Branch 2 (Semantic)
# • focused_chunk_top_k: số chunks giữ lại sau rerank (override chunk_top_k)
FOCUSED_TOP_K = 15                          # Số anchor nodes
FOCUSED_EDGE_QUOTA = 10                     # Max edges per anchor
FOCUSED_EDGE_THRESHOLD = 0.3               # Min semantic score
FOCUSED_ALPHA = 0.3                        # Weight anchor score
FOCUSED_BETA = 0.7                         # Weight edge semantic score
FOCUSED_MAX_EDGES = 50                     # Global cap edges
FOCUSED_CHUNK_TOP_K = 10                   # chunk_top_k cho vector retrieval
FOCUSED_ANCHOR_TOP_K = 10                   # K per-branch per-keyword (BM25 & Semantic)
FOCUSED_ANCHOR_SEMANTIC_THRESHOLD = 0.6    # Ngưỡng cosine Branch 2 (Semantic)
FOCUSED_BOTH_BONUS = 0.1                   # Bonus for entities found by BOTH BM25 + semantic
FOCUSED_CHUNK_TOP_K_RERANK = 10             # Top-K chunks sau rerank (precision mode)

# Các mode khác dùng top_k mặc định
DEFAULT_TOP_K = 10
# ===========================================================


def load_existing_results() -> dict:
    """Đọc kết quả đã có từ file Excel (nếu tồn tại).
    Trả về dict: {mode: DataFrame} cho các mode đã có sheet."""
    existing = {}
    if not os.path.exists(OUTPUT_FILE):
        return existing
    try:
        xls = pd.ExcelFile(OUTPUT_FILE, engine="openpyxl")
        for mode in MODES:
            sheet = mode.capitalize()[:31]
            if sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                if "question_text" in df.columns and len(df) > 0:
                    existing[mode] = df
                    print(f"   ♻️  Đã tải {len(df)} kết quả cũ cho mode {mode.upper()}")
        xls.close()
    except Exception as e:
        print(f"   ⚠️ Không đọc được file kết quả cũ: {e}")
    return existing


def load_baseline_results() -> dict:
    """Đọc kết quả baseline (hybrid, mix) từ file Excel để so sánh % cải thiện.
    Luôn tải hybrid và mix bất kể MODES hiện tại là gì."""
    baselines = {}
    if not os.path.exists(OUTPUT_FILE):
        return baselines
    try:
        xls = pd.ExcelFile(OUTPUT_FILE, engine="openpyxl")
        for mode in ["hybrid", "mix"]:
            sheet = mode.capitalize()[:31]
            if sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                if "question_text" in df.columns and len(df) > 0:
                    baselines[mode] = df
        xls.close()
    except Exception:
        pass
    return baselines


def save_results_incremental(all_mode_results: dict, baselines: dict = None):
    """Lưu kết quả hiện tại ra Excel (ghi đè). Gọi sau mỗi mode để tránh mất data."""
    try:
        summary_df = create_summary_sheet(all_mode_results, baselines)
        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            for mode in all_mode_results:
                sheet = mode.capitalize()[:31]
                all_mode_results[mode].to_excel(writer, sheet_name=sheet, index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
        print(f"   💾 Đã lưu checkpoint → {OUTPUT_FILE}")
    except PermissionError:
        print(f"   ⚠️ Không lưu được checkpoint — file đang mở. Đóng Excel và thử lại.")
    except Exception as e:
        print(f"   ⚠️ Lỗi lưu checkpoint: {e}")


def clean_answer(answer: str) -> str:
    """Loại bỏ phần ### References khỏi answer"""
    return re.split(r"\n*###\s*References", answer, maxsplit=1)[0].strip()


def estimate_tokens(text: str) -> int:
    """Ước lượng số tokens từ text (approximate: ~4 chars/token)"""
    if not text:
        return 0
    return max(1, len(text) // 4)


def extract_chunks_from_context(raw_context: str) -> str:
    """
    Trích xuất context từ LightRAG cho RAGAS evaluation.

    Chiến lược: CHUNKS TRƯỚC, GRAPH SAU
      1. Document Chunks (text gốc)     ← đặt TRƯỚC → Context Precision cao
      2. Entity descriptions (verbalize) ← đặt SAU   → bổ sung Faithfulness
      3. Relation descriptions (verbalize)← đặt SAU   → bổ sung Faithfulness
    """
    import json as _json

    chunk_start = raw_context.find("Document Chunks")
    if chunk_start == -1:
        return raw_context  # naive mode → trả nguyên gốc

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
                            graph_texts.append(f"Thực thể [{name}] có thông tin: {desc}")
                    except _json.JSONDecodeError:
                        pass

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
                            graph_texts.append(f"Mối quan hệ giữa [{e1}] và [{e2}] là: {desc}")
                    except _json.JSONDecodeError:
                        pass

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


def query_lightrag_with_timing(question: str, mode: str, retries: int = 3):
    """
    Query LightRAG — DUY NHẤT 1 API CALL.

    Gửi include_context=True để server trả về luôn context trong response:
      {
        "response": "...",
        "context": "...(retrieval context cho RAGAS)...",
        "timing": {"retrieval_ms": ..., "generation_ms": ..., "total_ms": ...}
      }

    Không cần gọi thêm call nào khác.
    """
    # Build mode-specific request params
    if mode == "beam":
        base = {
            "query": question,
            "mode": mode,
            "stream": False,
            "top_k": DEFAULT_TOP_K,
            "beam_width": BEAM_BEAM_WIDTH,
            "beam_max_depth": BEAM_MAX_DEPTH,
            "chunk_top_k": BEAM_CHUNK_TOP_K,
            "pruning_threshold": BEAM_PRUNING_THRESHOLD,
            "anchor_alpha": BEAM_ANCHOR_ALPHA,
            "chunk_alpha": BEAM_CHUNK_ALPHA,
            "related_chunk_number": RELATED_CHUNK_NUMBER,  # Ép riêng cho Beam
            "enable_rerank": BEAM_ENABLE_RERANK,  # Vietnamese Reranker lọc context (chunk-level)
            "enable_anchor_rerank": BEAM_ENABLE_ANCHOR_RERANK,  # Rerank anchor entities trước beam
            "include_context": True,  # Server trả context luôn
        }
    elif mode == "focused":
        base = {
            "query": question,
            "mode": mode,
            "stream": False,
            "top_k": FOCUSED_TOP_K,
            "chunk_top_k": FOCUSED_CHUNK_TOP_K,
            "focused_edge_quota": FOCUSED_EDGE_QUOTA,
            "focused_edge_threshold": FOCUSED_EDGE_THRESHOLD,
            "focused_alpha": FOCUSED_ALPHA,
            "focused_beta": FOCUSED_BETA,
            "focused_max_edges": FOCUSED_MAX_EDGES,
            # True Hybrid Anchor (KLTN) — BM25 ∪ Semantic per-keyword
            "focused_anchor_top_k": FOCUSED_ANCHOR_TOP_K,
            "focused_anchor_semantic_threshold": FOCUSED_ANCHOR_SEMANTIC_THRESHOLD,
            "focused_both_bonus": FOCUSED_BOTH_BONUS,
            # Rerank top-K chunks (KLTN Phần 2): override chunk_top_k sau rerank
            "focused_chunk_top_k": FOCUSED_CHUNK_TOP_K_RERANK,
            "enable_rerank": True,   # Bật rerank để focused_chunk_top_k có hiệu lực
            "include_context": True,  # Server trả context luôn
        }
    else:
        base = {
            "query": question,
            "mode": mode,
            "stream": False,
            "top_k": DEFAULT_TOP_K,
            "enable_rerank": False,
            "include_context": True,  # Server trả context luôn
        }

    for attempt in range(retries):
        try:
            # Đo wall-clock time phía client để bắt rerank latency
            # (server không có field rerank_ms riêng, gộp trong retrieval_ms)
            client_start = time.perf_counter()

            # --- DUY NHẤT 1 CALL: answer + timing + context ---
            resp = requests.post(
                f"{LIGHTRAG_URL}/query",
                json=base,
                timeout=180,
            )
            client_elapsed_ms = (time.perf_counter() - client_start) * 1000

            resp.raise_for_status()
            resp_json = resp.json()

            answer = clean_answer(resp_json.get("response", ""))

            # Server-side timing (đo chính xác tại server)
            timing = resp_json.get("timing") or {}
            keyword_extraction_ms = timing.get("keyword_extraction_ms", 0)
            graph_search_ms = timing.get("graph_search_ms", 0)
            rerank_ms = timing.get("rerank_ms", 0)           # ← Tách riêng từ server
            retrieval_latency_ms = timing.get("retrieval_ms", 0)
            generation_latency_ms = timing.get("generation_ms", 0)
            total_latency_ms = timing.get("total_ms", 0)

            # ước tính rerank_latency:
            # Client wall-time - server total = network overhead + rerank (nếu rerank chạy ngoài server timer)
            # Thực tế: rerank đã được gộp trong retrieval_ms của server
            # → Lưu client_wall_ms để theo dõi toàn bộ round-trip (bao gồm rerank)
            rerank_latency_ms = round(client_elapsed_ms - total_latency_ms, 2) if total_latency_ms > 0 else 0

            # Context từ cùng response (không cần call thêm)
            full_context = resp_json.get("context", "")
            if full_context:
                max_chars = BEAM_MAX_CONTEXT_CHARS if mode == "beam" else MAX_CONTEXT_CHARS
                full_context = truncate_context(full_context, max_chars)
            contexts = [full_context] if full_context else ["No context retrieved"]

            # Real token counts từ server tokenizer (chính xác, không phải ước lượng)
            token_counts = resp_json.get("token_counts") or {}
            input_tokens = token_counts.get("input_tokens", 0)
            output_tokens = token_counts.get("output_tokens", 0)
            # Fallback nếu server không trả token counts (cache hit, streaming, etc.)
            if input_tokens == 0:
                input_tokens = estimate_tokens(question + "\n" + (full_context or ""))
            if output_tokens == 0:
                output_tokens = estimate_tokens(answer)

            metrics = {
                "keyword_extraction_ms": round(keyword_extraction_ms, 2),
                "graph_search_ms": round(graph_search_ms, 2),        # Graph thuần (không rerank)
                "rerank_ms": round(rerank_ms, 2),                     # Rerank riêng (từ server)
                "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                "rerank_latency_ms": rerank_latency_ms,               # client overhead
                "generation_latency_ms": round(generation_latency_ms, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_latency_ms": round(total_latency_ms, 2),
                "client_wall_ms": round(client_elapsed_ms, 2),
            }

            return answer, contexts, metrics

        except Exception as e:
            print(f"    ⚠️ Lỗi (lần {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(3)

    return "Error: Không lấy được response", ["No context"], {
        "keyword_extraction_ms": 0,
        "graph_search_ms": 0,
        "rerank_ms": 0,
        "retrieval_latency_ms": 0,
        "rerank_latency_ms": 0,
        "generation_latency_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_latency_ms": 0,
        "client_wall_ms": 0,
    }




def run_ragas_evaluation(questions, answers, contexts_list, ground_truths):
    """Chạy RAGAS evaluation trên dataset — LLM Judge dùng local vLLM"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings

    # LLM Judge — Local vLLM qua OpenAI-compatible endpoint
    llm = LangchainLLMWrapper(
        langchain_llm=ChatOpenAI(
            model=EVAL_LLM_MODEL,
            api_key=EVAL_LLM_API_KEY,
            base_url=EVAL_LLM_HOST,
            temperature=0.0,
            max_tokens=4096,   # local model: 4096 đủ, tiết kiệm VRAM
            max_retries=3,
            timeout=300.0,     # local inference chậm hơn cloud, tăng timeout
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
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    from ragas.run_config import RunConfig

    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()],
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=4, max_retries=10),
    )
    return results.to_pandas()


def create_summary_sheet(all_mode_results: dict, baselines: dict = None) -> pd.DataFrame:
    """Tạo sheet Summary: thống kê theo từng mode + % cải thiện so với baseline"""
    metric_cols = [
        "faithfulness", "answer_relevancy", "context_recall", "context_precision",
        "ragas_score", "input_tokens", "output_tokens",
        "keyword_extraction_ms", "graph_search_ms",
        "retrieval_latency_ms", "generation_latency_ms", "total_latency_ms"
    ]
    metric_labels = [
        "Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision",
        "RAGAS Score", "Input Tokens", "Output Tokens",
        "Keyword Extraction (ms)", "Graph Search (ms)",
        "Retrieval Latency (ms)", "Generation Latency (ms)", "Total Latency (ms)"
    ]
    # Quality metrics: higher is better; Cost/latency metrics: lower is better
    higher_is_better = {"faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"}

    if baselines is None:
        baselines = {}

    rows = []
    for label, col in zip(metric_labels, metric_cols):
        row = {"Metric": label}
        means = {}
        for mode in MODES:
            df = all_mode_results[mode]
            if col in df.columns:
                means[mode] = float(df[col].mean())
                row[f"{mode.capitalize()}_Mean"] = round(means[mode], 4)
            else:
                means[mode] = 0.0
                row[f"{mode.capitalize()}_Mean"] = "-"

        if col in higher_is_better:
            best = max(means, key=means.get)
            worst = min(means, key=means.get)
            row["Winner"] = best.capitalize()
            row["Spread"] = round(means[best] - means[worst], 4)
        else:
            best = min(means, key=means.get)
            worst = max(means, key=means.get)
            row["Winner"] = best.capitalize()
            row["Spread"] = round(means[worst] - means[best], 4)

        # --- % cải thiện so với baseline (hybrid, mix) ---
        # Tính % cải thiện cho tất cả các mode không phải baseline
        non_baseline_modes = [m for m in MODES if m not in ["hybrid", "mix"]]
        for cmp_mode in non_baseline_modes:
            cmp_val = means.get(cmp_mode)
            if cmp_val is not None:
                for baseline_mode in ["hybrid", "mix"]:
                    baseline_df = baselines.get(baseline_mode)
                    if baseline_df is None:
                        baseline_df = all_mode_results.get(baseline_mode)
                    if baseline_df is not None and col in baseline_df.columns:
                        baseline_val = float(baseline_df[col].mean())
                        if baseline_val != 0:
                            pct = ((cmp_val - baseline_val) / abs(baseline_val)) * 100
                            row[f"{cmp_mode.capitalize()}_vs_{baseline_mode.capitalize()}_%"] = f"{pct:+.1f}%"
                        else:
                            row[f"{cmp_mode.capitalize()}_vs_{baseline_mode.capitalize()}_%"] = "N/A"
                    else:
                        row[f"{cmp_mode.capitalize()}_vs_{baseline_mode.capitalize()}_%"] = "-"

        rows.append(row)

    return pd.DataFrame(rows)


def print_comparison_table(all_mode_results: dict, baselines: dict = None):
    """In bảng so sánh ra console, bao gồm cột % cải thiện so với Hybrid/Mix"""
    if baselines is None:
        baselines = {}

    metric_cols = [
        "faithfulness", "answer_relevancy", "context_recall", "context_precision",
        "ragas_score", "input_tokens", "output_tokens",
        "keyword_extraction_ms", "graph_search_ms",
        "retrieval_latency_ms", "generation_latency_ms", "total_latency_ms"
    ]
    metric_labels = [
        "Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision",
        "RAGAS Score", "Input Tokens", "Output Tokens",
        "Keyword Extraction (ms)", "Graph Search (ms)",
        "Retrieval Latency (ms)", "Generation Latency (ms)", "Total Latency (ms)"
    ]
    # Quality metrics: higher is better; Cost/latency metrics: lower is better
    higher_is_better = {"faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"}

    # Detect which baselines are available
    available_baselines = []
    for bm in ["hybrid", "mix"]:
        if bm in all_mode_results or bm in baselines:
            available_baselines.append(bm)

    col_w = max(10, max(len(m) for m in MODES) + 2)
    pct_w = 12  # width for % columns

    # Build header
    header = f"  {'Metric':<28s}"
    header += "".join(f"{m.upper():>{col_w}s}" for m in MODES)
    for bm in available_baselines:
        header += f"  {'vs ' + bm.capitalize():>{pct_w}s}"
    header += f"  {'Winner':>10s}"
    print(header)

    sep_len = 28 + col_w * len(MODES) + pct_w * len(available_baselines) + 2 * len(available_baselines) + 12
    print(f"  {'─' * sep_len}")

    for label, col in zip(metric_labels, metric_cols):
        means = {}
        for mode in MODES:
            df = all_mode_results[mode]
            if col in df.columns:
                means[mode] = df[col].mean()
            else:
                means[mode] = None

        if col in ["ragas_score"]:
            print(f"  {'─' * sep_len}")

        # Determine winner
        valid_means = {m: v for m, v in means.items() if v is not None and v != 0}
        if not valid_means:
            continue

        if col in higher_is_better:
            best_m = max(valid_means, key=valid_means.get)
            if len(set(round(v, 4) for v in valid_means.values())) == 1:
                winner = "TIE"
            else:
                winner = best_m.upper()
        else:
            best_m = min(valid_means, key=valid_means.get)
            if len(set(round(v, 4) for v in valid_means.values())) == 1:
                winner = "TIE"
            else:
                winner = best_m.upper()

        # Format mode values
        vals_str = ""
        for mode in MODES:
            v = means[mode]
            if v is None or v == 0:
                vals_str += f"{'N/A':>{col_w}s}"
            elif col in higher_is_better:
                vals_str += f"{v:>{col_w}.4f}"
            else:
                vals_str += f"{v:>{col_w}.1f}"

        # Format % vs baseline columns (for all non-baseline modes)
        non_baseline_modes = [m for m in MODES if m not in ["hybrid", "mix"]]
        for cmp_mode in non_baseline_modes:
            cmp_val = means.get(cmp_mode)
            for bm in available_baselines:
                baseline_df = baselines.get(bm)
                if baseline_df is None:
                    baseline_df = all_mode_results.get(bm)
                if baseline_df is not None and col in baseline_df.columns and cmp_val is not None:
                    baseline_val = float(baseline_df[col].mean())
                    if baseline_val != 0:
                        pct = ((cmp_val - baseline_val) / abs(baseline_val)) * 100
                        # For quality metrics: positive = better; for latency: negative = better
                        vals_str += f"  {pct:>+{pct_w}.1f}%"
                    else:
                        vals_str += f"  {'N/A':>{pct_w}s}"
                else:
                    vals_str += f"  {'-':>{pct_w}s}"

        line = f"  {label:<28s}{vals_str}  {winner:>10s}"
        print(line)

    # Print legend
    if available_baselines:
        print(f"\n  📝 Ghi chú: Cột 'vs Hybrid/Mix' = % thay đổi của Beam so với baseline")
        print(f"     Chất lượng (Faithfulness, RAGAS...): + = tốt hơn, - = kém hơn")
        print(f"     Tốc độ/Token (Latency, Tokens...):  - = nhanh/ít hơn (tốt), + = chậm/nhiều hơn (xấu)")


def main():
    print("=" * 70)
    print(f"📊 RAGAS Evaluation — So sánh {' vs '.join(m.capitalize() for m in MODES)}")
    print("   Khóa Luận Tốt Nghiệp")
    print("=" * 70)

    # 1. Đọc test cases
    print(f"\n📂 Input: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Tìm thấy {len(df)} test cases")

    if TEST_LIMIT is not None:
        df = df.head(TEST_LIMIT)
        print(f"   Sử dụng {TEST_LIMIT} câu đầu tiên")

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
    print(f"   Modes: {', '.join(m.upper() for m in MODES)}")

    # 2. Tải kết quả cũ (nếu có) để skip câu đã đánh giá
    existing_results = load_existing_results()
    all_mode_results = {}

    for mode in MODES:
        print(f"\n{'=' * 70}")
        print(f"🔍 ĐÁNH GIÁ MODE: {mode.upper()} ({total} câu hỏi)")
        print(f"{'=' * 70}")

        # Xác định câu đã đánh giá rồi
        existing_df = existing_results.get(mode)
        if existing_df is not None:
            done_questions = set(existing_df["question_text"].tolist())
        else:
            done_questions = set()

        # Lọc ra câu cần chạy mới
        new_indices = [i for i, q in enumerate(all_questions) if q not in done_questions]
        skipped = total - len(new_indices)
        if skipped > 0:
            print(f"   ♻️  Bỏ qua {skipped} câu đã có kết quả, chỉ chạy {len(new_indices)} câu mới")

        if len(new_indices) == 0:
            print(f"   ✅ Tất cả {total} câu đã có kết quả — bỏ qua mode {mode.upper()}")
            all_mode_results[mode] = existing_df
            continue

        # --- Query LightRAG cho các câu MỚI ---
        print(f"\n🔄 Querying LightRAG (mode: {mode}) — {len(new_indices)} câu mới...")
        mode_questions = []
        mode_answers = []
        mode_contexts_list = []
        mode_ground_truths = []
        mode_metrics = []

        overall_query_start = time.time()

        for idx_in_new, i in enumerate(new_indices):
            question = all_questions[i]
            ground_truth = all_ground_truths[i]
            print(f"\n  [{idx_in_new+1}/{len(new_indices)}] (câu {i+1}/{total}) {question[:60]}...")
            print(f"      Mode: {mode.upper()}")

            answer, contexts, metrics = query_lightrag_with_timing(question, mode)
            if mode == "beam":
                print(f"      BEAM params: beam_width={BEAM_BEAM_WIDTH}, max_depth={BEAM_MAX_DEPTH}, chunk_top_k={BEAM_CHUNK_TOP_K}")
            elif mode == "focused":
                print(f"      FOCUSED params: top_k={FOCUSED_TOP_K}, quota={FOCUSED_EDGE_QUOTA}, threshold={FOCUSED_EDGE_THRESHOLD}, α={FOCUSED_ALPHA}, β={FOCUSED_BETA}, max_edges={FOCUSED_MAX_EDGES}")
                print(f"      ANCHOR  params: anchor_top_k={FOCUSED_ANCHOR_TOP_K}, sem_threshold={FOCUSED_ANCHOR_SEMANTIC_THRESHOLD}, both_bonus={FOCUSED_BOTH_BONUS}")
                print(f"      CHUNK   params: chunk_top_k={FOCUSED_CHUNK_TOP_K}, rerank_top_k={FOCUSED_CHUNK_TOP_K_RERANK}, enable_rerank=True")

            print(f"      Answer: {answer[:80]}...")
            print(f"      Input tokens: {metrics['input_tokens']}")
            print(f"      Output tokens: {metrics['output_tokens']}")
            print(f"      Keyword extraction: {metrics['keyword_extraction_ms']:.1f}ms")
            print(f"      Graph search: {metrics['graph_search_ms']:.1f}ms")
            print(f"      Retrieval latency: {metrics['retrieval_latency_ms']:.1f}ms")
            print(f"      Generation latency: {metrics['generation_latency_ms']:.1f}ms")

            mode_questions.append(question)
            mode_answers.append(answer)
            mode_contexts_list.append(contexts)
            mode_ground_truths.append(ground_truth)
            mode_metrics.append(metrics)

        overall_query_time = time.time() - overall_query_start
        print(f"\n✅ Query xong {len(mode_questions)} câu mới trong {overall_query_time:.1f}s")

        # Tạo DataFrame cho timing metrics
        timing_df = pd.DataFrame(mode_metrics)
        timing_df.insert(0, "question_text", mode_questions)

        # --- Chạy RAGAS evaluation (chỉ trên câu mới) ---
        print(f"\n🔬 Chạy RAGAS evaluation ({mode}) — {len(mode_questions)} câu mới...")
        print(f"   LLM Judge: {EVAL_LLM_MODEL}")
        print(f"   Embedding: {EMBEDDING_MODEL} (Ollama)")

        eval_start = time.time()
        try:
            ragas_df = run_ragas_evaluation(
                mode_questions, mode_answers, mode_contexts_list, mode_ground_truths
            )
            eval_time = time.time() - eval_start
            print(f"✅ RAGAS evaluation xong trong {eval_time:.1f}s")

            # Ghép timing + RAGAS metrics
            ragas_df.insert(0, "question_text", mode_questions)
            ragas_df["mode"] = mode

            # Tính RAGAS score
            ragas_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
            valid_ragas = [c for c in ragas_cols if c in ragas_df.columns]
            ragas_df["ragas_score"] = ragas_df[valid_ragas].mean(axis=1)

            # Ghép timing vào
            for col in ["keyword_extraction_ms", "graph_search_ms",
                        "input_tokens", "output_tokens", "retrieval_latency_ms",
                        "generation_latency_ms", "total_latency_ms"]:
                if col in timing_df.columns:
                    ragas_df[col] = timing_df[col].values

            new_results_df = ragas_df

        except Exception as e:
            print(f"   ❌ Lỗi RAGAS: {e}")
            # Vẫn lưu timing data dù không có RAGAS
            ragas_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
            timing_df["mode"] = mode
            for col in ragas_cols + ["ragas_score"]:
                timing_df[col] = None
            new_results_df = timing_df

        # Ghép kết quả mới với kết quả cũ
        if existing_df is not None and len(existing_df) > 0:
            combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
            # Loại bỏ duplicate (giữ kết quả mới nhất nếu trùng)
            combined_df = combined_df.drop_duplicates(subset=["question_text"], keep="last")
            # Sắp xếp lại theo thứ tự gốc
            q_order = {q: i for i, q in enumerate(all_questions)}
            combined_df["_sort_key"] = combined_df["question_text"].map(q_order)
            combined_df = combined_df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
            all_mode_results[mode] = combined_df
            print(f"   📊 Tổng: {len(existing_df)} cũ + {len(new_results_df)} mới = {len(combined_df)} câu")
        else:
            all_mode_results[mode] = new_results_df

        # In kết quả mode này
        print(f"\n📈 Kết quả {mode.upper()}:")
        print(f"   Input Tokens         : {all_mode_results[mode]['input_tokens'].mean():.1f}")
        print(f"   Output Tokens        : {all_mode_results[mode]['output_tokens'].mean():.1f}")
        print(f"   Keyword Extraction   : {all_mode_results[mode]['keyword_extraction_ms'].mean():.1f}ms")
        print(f"   Graph Search         : {all_mode_results[mode]['graph_search_ms'].mean():.1f}ms")
        print(f"   Retrieval Latency    : {all_mode_results[mode]['retrieval_latency_ms'].mean():.1f}ms")
        print(f"   Generation Latency   : {all_mode_results[mode]['generation_latency_ms'].mean():.1f}ms")
        print(f"   Total Latency        : {all_mode_results[mode]['total_latency_ms'].mean():.1f}ms")
        ragas_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        for col in ragas_cols:
            if col in all_mode_results[mode].columns:
                print(f"   {col:<25s}: {all_mode_results[mode][col].mean():.4f}")
        if "ragas_score" in all_mode_results[mode].columns:
            print(f"   {'RAGAS Score (avg)':<25s}: {all_mode_results[mode]['ragas_score'].mean():.4f}")

        # Lưu checkpoint sau mỗi mode (tránh mất data nếu crash)
        save_results_incremental(all_mode_results)

    # ==================== LOAD BASELINES ====================
    baselines = load_baseline_results()
    if baselines:
        print(f"\n📊 Đã tải baseline để so sánh: {', '.join(m.upper() for m in baselines)}")

    # ==================== SO SÁNH ====================
    print(f"\n{'=' * 70}")
    print(f"📊 SO SÁNH CÁC MODE: {', '.join(m.upper() for m in MODES)}")
    print(f"{'=' * 70}")
    print_comparison_table(all_mode_results, baselines)
    print(f"{'=' * 70}")

    # In số câu mỗi mode
    for mode in MODES:
        print(f"   📊 {mode.upper()}: {len(all_mode_results[mode])} câu")

    # ==================== LƯU KẾT QUẢ CUỐI ====================
    save_results_incremental(all_mode_results, baselines)

    print(f"\n💾 Kết quả đã lưu: {OUTPUT_FILE}")
    for mode in MODES:
        print(f"   📄 Sheet '{mode.capitalize()[:31]}' — {len(all_mode_results[mode])} câu")
    print("   📄 Sheet 'Summary' — tổng hợp so sánh")

    # # In chi tiết từng câu
    # print(f"\n{'=' * 70}")
    # print("📋 CHI TIẾT TỪNG CÂU HỎI")
    # print(f"{'=' * 70}")
    # for i, question in enumerate(all_questions):
    #     print(f"\n--- Câu {i+1}: {question[:60]}...")
    #     for mode in MODES:
    #         df = all_mode_results.get(mode)
    #         if df is None:
    #             continue
    #         match = df[df["question_text"] == question]
    #         if len(match) > 0:
    #             row = match.iloc[0]
    #             print(f"  [{mode.upper()}]")
    #             print(f"    Input Tokens: {int(row.get('input_tokens', 0))}")
    #             print(f"    Retrieval: {row.get('retrieval_latency_ms', 0):.1f}ms")
    #             print(f"    Generation: {row.get('generation_latency_ms', 0):.1f}ms")
    #             for col in ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]:
    #                 if col in row.index and row[col] is not None:
    #                     try:
    #                         print(f"    {col}: {float(row[col]):.4f}")
    #                     except (ValueError, TypeError):
    #                         pass


if __name__ == "__main__":
    main()
