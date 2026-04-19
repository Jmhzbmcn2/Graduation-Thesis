"""
RAGAS Evaluation — So sánh Hybrid vs Mix vs Beam
Khóa Luận Tốt Nghiệp

So sánh 3 mode: hybrid, mix, beam trên các tiêu chí:
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

Output: eval_ragas_hybrid_mix_beam_1case.xlsx (1 sheet/mode + Summary)

Cách dùng:
  python testcase/eval_ragas_hybrid_mix_beam.py
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

warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*token usage.*", category=UserWarning)

# ======================== CẤU HÌNH ========================
LIGHTRAG_URL = "http://localhost:9621"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(_SCRIPT_DIR, "500_cases.csv")
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "eval_ragas_hybrid_mix_beam_1case.xlsx")

# RAGAS LLM Judge — OpenRouter
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "qwen/qwen3-30b-a3b-instruct-2507")
EVAL_LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", os.getenv("EVAL_LLM_BINDING_API_KEY", ""))
EVAL_LLM_HOST = os.getenv("EVAL_LLM_BINDING_HOST", "https://openrouter.ai/api/v1")

# RAGAS Embedding — Ollama local
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
EMBEDDING_HOST  = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")

# Số test case (None = tất cả, 3 = 3 câu đầu tiên)
TEST_LIMIT = 5

# Giới hạn độ dài context để tránh vượt token limit của LLM Judge
MAX_CONTEXT_CHARS = None

# Giới hạn riêng cho BEAM mode
# QUAN TRỌNG: KHÔNG cắt ngắn context beam, vì RAGAS cần thấy toàn bộ context
# mà LLM đã dùng để sinh answer. Cắt ngắn → Faithfulness = 0 giả.
BEAM_MAX_CONTEXT_CHARS = None  # None = không giới hạn (giống các mode khác)

# Modes cần đánh giá
MODES = ["hybrid", "mix","beam"]
# MODES = ["beam"]
# Batch size cho RAGAS
EVAL_BATCH_SIZE = 100

# ======================== CẤU HÌNH MODE-SPECIFIC ========================
# Beam search tối ưu: beam_width lớn + depth sâu hơn để lấy nhiều context
BEAM_BEAM_WIDTH = 5     # Mặc định = 3. Tăng lên 7 để mỗi hop giữ nhiều candidate hơn
BEAM_MAX_DEPTH = 2     # Mặc định = 1. Tăng lên 2 để khám phá indirect relationships
BEAM_CHUNK_TOP_K = 5   # Mặc định = 10. Tăng để lấy nhiều text chunks hơn

# Các mode khác dùng top_k mặc định
DEFAULT_TOP_K = 5
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


def save_results_incremental(all_mode_results: dict):
    """Lưu kết quả hiện tại ra Excel (ghi đè). Gọi sau mỗi mode để tránh mất data."""
    try:
        summary_df = create_summary_sheet(all_mode_results)
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
                            graph_texts.append(f"{name}: {desc}")
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
                            graph_texts.append(f"{e1} - {e2}: {desc}")
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
            "include_context": True,  # Server trả context luôn
        }
    else:
        base = {
            "query": question,
            "mode": mode,
            "stream": False,
            "top_k": DEFAULT_TOP_K,
            "include_context": True,  # Server trả context luôn
        }

    for attempt in range(retries):
        try:
            # --- DUY NHẤT 1 CALL: answer + timing + context ---
            resp = requests.post(
                f"{LIGHTRAG_URL}/query",
                json=base,
                timeout=180,
            )
            resp.raise_for_status()
            resp_json = resp.json()

            answer = clean_answer(resp_json.get("response", ""))

            # Server-side timing (đo chính xác tại server)
            timing = resp_json.get("timing") or {}
            retrieval_latency_ms = timing.get("retrieval_ms", 0)
            generation_latency_ms = timing.get("generation_ms", 0)
            total_latency_ms = timing.get("total_ms", 0)

            # Context từ cùng response (không cần call thêm)
            full_context = resp_json.get("context", "")
            if full_context:
                full_context = extract_chunks_from_context(full_context)
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
                "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                "generation_latency_ms": round(generation_latency_ms, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_latency_ms": round(total_latency_ms, 2),
            }

            return answer, contexts, metrics

        except Exception as e:
            print(f"    ⚠️ Lỗi (lần {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(3)

    return "Error: Không lấy được response", ["No context"], {
        "retrieval_latency_ms": 0,
        "generation_latency_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_latency_ms": 0,
    }


def run_ragas_evaluation(questions, answers, contexts_list, ground_truths):
    """Chạy RAGAS evaluation trên dataset — LLM Judge dùng OpenRouter"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings

    # LLM Judge — OpenRouter qua OpenAI-compatible endpoint
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


def create_summary_sheet(all_mode_results: dict) -> pd.DataFrame:
    """Tạo sheet Summary: thống kê theo từng mode"""
    metric_cols = [
        "faithfulness", "answer_relevancy", "context_recall", "context_precision",
        "ragas_score", "input_tokens", "output_tokens",
        "retrieval_latency_ms", "generation_latency_ms", "total_latency_ms"
    ]
    metric_labels = [
        "Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision",
        "RAGAS Score", "Input Tokens", "Output Tokens",
        "Retrieval Latency (ms)", "Generation Latency (ms)", "Total Latency (ms)"
    ]

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

        if col in ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]:
            best = max(means, key=means.get)
            worst = min(means, key=means.get)
            row["Winner"] = best.capitalize()
            row["Spread"] = round(means[best] - means[worst], 4)
        else:
            # Latency/tokens: lower is better
            best = min(means, key=means.get)
            worst = max(means, key=means.get)
            row["Winner"] = best.capitalize()
            row["Spread"] = round(means[worst] - means[best], 4)

        rows.append(row)

    return pd.DataFrame(rows)


def print_comparison_table(all_mode_results: dict):
    """In bảng so sánh ra console"""
    metric_cols = [
        "faithfulness", "answer_relevancy", "context_recall", "context_precision",
        "ragas_score", "input_tokens", "output_tokens",
        "retrieval_latency_ms", "generation_latency_ms", "total_latency_ms"
    ]
    metric_labels = [
        "Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision",
        "RAGAS Score", "Input Tokens", "Output Tokens",
        "Retrieval Latency (ms)", "Generation Latency (ms)", "Total Latency (ms)"
    ]

    col_w = max(10, max(len(m) for m in MODES) + 2)
    header = f"  {'Metric':<28s}" + "".join(f"{m.upper():>{col_w}s}" for m in MODES) + f"  {'Winner':>10s}"
    print(header)
    print(f"  {'─' * (28 + col_w * len(MODES) + 12)}")

    for label, col in zip(metric_labels, metric_cols):
        means = {}
        for mode in MODES:
            df = all_mode_results[mode]
            if col in df.columns:
                means[mode] = df[col].mean()
            else:
                means[mode] = None

        if col in ["ragas_score"]:
            print(f"  {'─' * (28 + col_w * len(MODES) + 12)}")

        # Determine winner
        valid_means = {m: v for m, v in means.items() if v is not None and v != 0}
        if not valid_means:
            continue

        if col in ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]:
            best_m = max(valid_means, key=valid_means.get)
            # TIE if all same
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

        # Format values
        vals_str = ""
        for mode in MODES:
            v = means[mode]
            if v is None or v == 0:
                vals_str += f"{'N/A':>{col_w}s}"
            elif col in ["ragas_score", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
                vals_str += f"{v:>{col_w}.4f}"
            else:
                vals_str += f"{v:>{col_w}.1f}"

        line = f"  {label:<28s}{vals_str}  {winner:>10s}"
        print(line)


def main():
    print("=" * 70)
    print(f"📊 RAGAS Evaluation — So sánh Hybrid vs Mix vs Beam")
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

            print(f"      Answer: {answer[:80]}...")
            print(f"      Input tokens: {metrics['input_tokens']}")
            print(f"      Output tokens: {metrics['output_tokens']}")
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
            for col in ["input_tokens", "output_tokens", "retrieval_latency_ms",
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

    # ==================== SO SÁNH ====================
    print(f"\n{'=' * 70}")
    print(f"📊 SO SÁNH CÁC MODE: {', '.join(m.upper() for m in MODES)}")
    print(f"{'=' * 70}")
    print_comparison_table(all_mode_results)
    print(f"{'=' * 70}")

    # In số câu mỗi mode
    for mode in MODES:
        print(f"   📊 {mode.upper()}: {len(all_mode_results[mode])} câu")

    # ==================== LƯU KẾT QUẢ CUỐI ====================
    save_results_incremental(all_mode_results)

    print(f"\n💾 Kết quả đã lưu: {OUTPUT_FILE}")
    for mode in MODES:
        print(f"   📄 Sheet '{mode.capitalize()[:31]}' — {len(all_mode_results[mode])} câu")
    print("   📄 Sheet 'Summary' — tổng hợp so sánh")

    # In chi tiết từng câu
    print(f"\n{'=' * 70}")
    print("📋 CHI TIẾT TỪNG CÂU HỎI")
    print(f"{'=' * 70}")
    for i, question in enumerate(all_questions):
        print(f"\n--- Câu {i+1}: {question[:60]}...")
        for mode in MODES:
            df = all_mode_results.get(mode)
            if df is None:
                continue
            match = df[df["question_text"] == question]
            if len(match) > 0:
                row = match.iloc[0]
                print(f"  [{mode.upper()}]")
                print(f"    Input Tokens: {int(row.get('input_tokens', 0))}")
                print(f"    Retrieval: {row.get('retrieval_latency_ms', 0):.1f}ms")
                print(f"    Generation: {row.get('generation_latency_ms', 0):.1f}ms")
                for col in ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]:
                    if col in row.index and row[col] is not None:
                        try:
                            print(f"    {col}: {float(row[col]):.4f}")
                        except (ValueError, TypeError):
                            pass


if __name__ == "__main__":
    main()
