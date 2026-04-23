"""
RAGAS Re-evaluation — Đọc kết quả từ Excel và chạy lại RAGAS metrics
Khóa Luận Tốt Nghiệp

Thay vì query lại LightRAG từ đầu, script này:
  1. Đọc câu hỏi, câu trả lời, context đã có từ eval_ragas_focused.xlsx
  2. Chạy RAGAS evaluation (Faithfulness, Answer Relevancy, Context Recall, Context Precision)
  3. Tính RAGAS Score trung bình
  4. Xuất kết quả ra eval_3_mode_reeval.xlsx (cùng format với eval_ragas_hybrid_mix_beam.py)

Cấu hình:
  - LLM Judge : vLLM local (Qwen/Qwen2.5-32B-Instruct-AWQ hoặc theo .env)
  - Embedding  : Ollama (embeddinggemma:300m) tại http://localhost:11434
  - Input      : testcase/eval_ragas_focused.xlsx (sheets: Hybrid, Mix, Focused)
  - Output     : testcase/eval_3_mode_reeval.xlsx

Cách dùng:
  python testcase/eval_3_mode_from_excell.py
"""

import os
import re
import ast
import time
import warnings
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=False)

warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*token usage.*", category=UserWarning)

# ======================== CẤU HÌNH ========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File đầu vào — chứa kết quả đã có (answers + contexts)
INPUT_EXCEL  = os.path.join(_SCRIPT_DIR, "eval_ragas_focused.xlsx")

# File đầu ra — kết quả re-evaluation
OUTPUT_FILE  = os.path.join(_SCRIPT_DIR, "eval_3_mode_reeval.xlsx")

# RAGAS LLM Judge — dùng vLLM local (hoặc OpenRouter nếu set OPENROUTER_API_KEY)
EVAL_LLM_MODEL   = os.getenv("EVAL_LLM_MODEL",   "Qwen/Qwen2.5-32B-Instruct-AWQ")
EVAL_LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", os.getenv("EVAL_LLM_BINDING_API_KEY", "sk-123456"))
EVAL_LLM_HOST    = os.getenv("EVAL_LLM_BINDING_HOST", "http://localhost:8000/v1")

# RAGAS Embedding — Ollama local
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
EMBEDDING_HOST  = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")

# Modes cần đánh giá (phải tồn tại trong INPUT_EXCEL dưới dạng sheet)
MODES = ["hybrid", "mix", "focused"]

# Giới hạn số câu (None = tất cả)
TEST_LIMIT = 5

# Batch size cho RAGAS
EVAL_BATCH_SIZE = 100

# ======================== HELPERS ========================

def parse_contexts_column(val) -> list:
    """
    Chuyển cột retrieved_contexts (string hoặc list) → list[str] cho RAGAS.
    Hỗ trợ: Python-literal list, plain string, và chuỗi trống.
    """
    if isinstance(val, list):
        return [str(c) for c in val if c]
    if not val or (isinstance(val, float)):
        return ["No context retrieved"]
    s = str(val).strip()
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(c) for c in parsed if c] or ["No context retrieved"]
        except Exception:
            pass
    return [s] if s else ["No context retrieved"]


def load_mode_from_excel(xls: pd.ExcelFile, mode: str) -> pd.DataFrame | None:
    """Đọc sheet của 1 mode từ Excel. Trả về None nếu không tìm thấy."""
    sheet_name = mode.capitalize()[:31]
    if sheet_name not in xls.sheet_names:
        print(f"   ⚠️  Không tìm thấy sheet '{sheet_name}' trong {INPUT_EXCEL}")
        return None
    df = pd.read_excel(xls, sheet_name=sheet_name)
    # Chuẩn hóa tên cột (loại newline thừa)
    df.columns = [c.strip().replace("\n", "") for c in df.columns]
    print(f"   📄 Sheet '{sheet_name}': {len(df)} dòng — cột: {list(df.columns)}")
    return df


def run_ragas_evaluation(questions, answers, contexts_list, ground_truths):
    """Chạy RAGAS evaluation — LLM Judge dùng vLLM local (hoặc OpenRouter)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts_list,
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
    """Tạo sheet Summary: thống kê theo từng mode + % cải thiện so với Hybrid/Mix."""
    metric_cols = [
        "faithfulness", "answer_relevancy", "context_recall", "context_precision",
        "ragas_score", "input_tokens", "output_tokens",
        "keyword_extraction_ms", "graph_search_ms",
        "retrieval_latency_ms", "generation_latency_ms", "total_latency_ms",
    ]
    metric_labels = [
        "Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision",
        "RAGAS Score", "Input Tokens", "Output Tokens",
        "Keyword Extraction (ms)", "Graph Search (ms)",
        "Retrieval Latency (ms)", "Generation Latency (ms)", "Total Latency (ms)",
    ]
    higher_is_better = {"faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"}
    available_modes  = [m for m in MODES if m in all_mode_results]

    rows = []
    for label, col in zip(metric_labels, metric_cols):
        row = {"Metric": label}
        means = {}
        for mode in available_modes:
            df = all_mode_results[mode]
            if col in df.columns:
                means[mode] = float(df[col].mean())
                row[f"{mode.capitalize()}_Mean"] = round(means[mode], 4)
            else:
                means[mode] = 0.0
                row[f"{mode.capitalize()}_Mean"] = "-"

        valid = {m: v for m, v in means.items() if v != 0.0}
        if valid:
            if col in higher_is_better:
                best  = max(valid, key=valid.get)
                worst = min(valid, key=valid.get)
            else:
                best  = min(valid, key=valid.get)
                worst = max(valid, key=valid.get)
            row["Winner"] = best.capitalize()
            row["Spread"] = round(abs(valid.get(best, 0) - valid.get(worst, 0)), 4)
        else:
            row["Winner"] = "-"
            row["Spread"] = 0

        # % cải thiện của non-baseline modes vs Hybrid / Mix
        non_baseline = [m for m in available_modes if m not in ("hybrid", "mix")]
        for cmp_mode in non_baseline:
            cmp_val = means.get(cmp_mode)
            if cmp_val is not None:
                for bm in ("hybrid", "mix"):
                    bm_val = means.get(bm)
                    if bm_val is not None and bm_val != 0:
                        pct = ((cmp_val - bm_val) / abs(bm_val)) * 100
                        row[f"{cmp_mode.capitalize()}_vs_{bm.capitalize()}_%"] = f"{pct:+.1f}%"
                    else:
                        row[f"{cmp_mode.capitalize()}_vs_{bm.capitalize()}_%"] = "-"
        rows.append(row)

    return pd.DataFrame(rows)


def print_comparison_table(all_mode_results: dict):
    """In bảng so sánh ra console (cùng format với eval_ragas_hybrid_mix_beam.py)."""
    metric_cols = [
        "faithfulness", "answer_relevancy", "context_recall", "context_precision",
        "ragas_score", "input_tokens", "output_tokens",
        "keyword_extraction_ms", "graph_search_ms",
        "retrieval_latency_ms", "generation_latency_ms", "total_latency_ms",
    ]
    metric_labels = [
        "Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision",
        "RAGAS Score", "Input Tokens", "Output Tokens",
        "Keyword Extraction (ms)", "Graph Search (ms)",
        "Retrieval Latency (ms)", "Generation Latency (ms)", "Total Latency (ms)",
    ]
    higher_is_better = {"faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"}
    available_modes  = [m for m in MODES if m in all_mode_results]
    available_baselines = [m for m in ("hybrid", "mix") if m in available_modes]

    col_w = max(10, max(len(m) for m in available_modes) + 2)
    pct_w = 12

    header = f"  {'Metric':<28s}"
    header += "".join(f"{m.upper():>{col_w}s}" for m in available_modes)
    for bm in available_baselines:
        header += f"  {'vs ' + bm.capitalize():>{pct_w}s}"
    header += f"  {'Winner':>10s}"
    print(header)

    sep_len = 28 + col_w * len(available_modes) + pct_w * len(available_baselines) + 2 * len(available_baselines) + 12
    print(f"  {'─' * sep_len}")

    for label, col in zip(metric_labels, metric_cols):
        means = {}
        for mode in available_modes:
            df = all_mode_results[mode]
            means[mode] = df[col].mean() if col in df.columns else None

        if col == "ragas_score":
            print(f"  {'─' * sep_len}")

        valid_means = {m: v for m, v in means.items() if v is not None and v != 0}
        if not valid_means:
            continue

        if col in higher_is_better:
            best_m = max(valid_means, key=valid_means.get)
            winner = "TIE" if len(set(round(v, 4) for v in valid_means.values())) == 1 else best_m.upper()
        else:
            best_m = min(valid_means, key=valid_means.get)
            winner = "TIE" if len(set(round(v, 4) for v in valid_means.values())) == 1 else best_m.upper()

        vals_str = ""
        for mode in available_modes:
            v = means[mode]
            if v is None or v == 0:
                vals_str += f"{'N/A':>{col_w}s}"
            elif col in higher_is_better:
                vals_str += f"{v:>{col_w}.4f}"
            else:
                vals_str += f"{v:>{col_w}.1f}"

        non_baseline = [m for m in available_modes if m not in ("hybrid", "mix")]
        for cmp_mode in non_baseline:
            cmp_val = means.get(cmp_mode)
            for bm in available_baselines:
                bm_val = means.get(bm)
                if bm_val is not None and bm_val != 0 and cmp_val is not None:
                    pct = ((cmp_val - bm_val) / abs(bm_val)) * 100
                    vals_str += f"  {pct:>+{pct_w}.1f}%"
                else:
                    vals_str += f"  {'-':>{pct_w}s}"

        print(f"  {label:<28s}{vals_str}  {winner:>10s}")

    if available_baselines:
        print(f"\n  📝 Ghi chú: Cột 'vs Hybrid/Mix' = % thay đổi của Focused so với baseline")
        print(f"     Chất lượng (Faithfulness, RAGAS...): + = tốt hơn, - = kém hơn")
        print(f"     Tốc độ/Token (Latency, Tokens...):  - = nhanh/ít hơn (tốt), + = chậm/nhiều hơn (xấu)")


def save_results(all_mode_results: dict):
    """Lưu kết quả ra Excel."""
    try:
        summary_df = create_summary_sheet(all_mode_results)
        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            for mode in MODES:
                if mode in all_mode_results:
                    sheet = mode.capitalize()[:31]
                    all_mode_results[mode].to_excel(writer, sheet_name=sheet, index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
        print(f"   💾 Đã lưu → {OUTPUT_FILE}")
    except PermissionError:
        print(f"   ⚠️ Không lưu được — file đang mở. Đóng Excel và thử lại.")
    except Exception as e:
        print(f"   ⚠️ Lỗi lưu: {e}")


# ======================== MAIN ========================

def main():
    print("=" * 70)
    print(f"📊 RAGAS Re-Evaluation — Đọc từ Excel, chạy lại RAGAS")
    print(f"   Input : {INPUT_EXCEL}")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Modes : {', '.join(m.upper() for m in MODES)}")
    print(f"   LLM Judge : {EVAL_LLM_MODEL} @ {EVAL_LLM_HOST}")
    print(f"   Embedding : {EMBEDDING_MODEL} @ {EMBEDDING_HOST}")
    print("=" * 70)

    if not os.path.exists(INPUT_EXCEL):
        print(f"❌ Không tìm thấy file đầu vào: {INPUT_EXCEL}")
        return

    print(f"\n📂 Đọc dữ liệu từ: {INPUT_EXCEL}")
    try:
        xls = pd.ExcelFile(INPUT_EXCEL, engine="openpyxl")
    except Exception as e:
        print(f"❌ Lỗi mở file Excel: {e}")
        return

    all_mode_results = {}

    for mode in MODES:
        print(f"\n{'=' * 70}")
        print(f"🔍 XỬ LÝ MODE: {mode.upper()}")
        print(f"{'=' * 70}")

        df = load_mode_from_excel(xls, mode)
        if df is None:
            continue

        if TEST_LIMIT is not None:
            df = df.head(TEST_LIMIT)
            print(f"   ⚡ Giới hạn {TEST_LIMIT} câu đầu tiên")

        # --- Trích xuất các trường cần thiết ---
        # Hỗ trợ cả 2 tên cột: question_text và user_input
        if "question_text" in df.columns:
            questions = df["question_text"].fillna("").astype(str).tolist()
        elif "user_input" in df.columns:
            questions = df["user_input"].fillna("").astype(str).tolist()
        else:
            print(f"   ❌ Không tìm thấy cột câu hỏi (question_text / user_input)")
            continue

        if "response" in df.columns:
            answers = df["response"].fillna("").astype(str).tolist()
        else:
            print(f"   ❌ Không tìm thấy cột 'response'")
            continue

        if "reference" in df.columns:
            ground_truths = df["reference"].fillna("").astype(str).tolist()
        elif "ground_truth" in df.columns:
            ground_truths = df["ground_truth"].fillna("").astype(str).tolist()
        else:
            print(f"   ⚠️  Không tìm thấy cột reference/ground_truth — dùng chuỗi rỗng")
            ground_truths = [""] * len(questions)

        if "retrieved_contexts" in df.columns:
            contexts_list = [parse_contexts_column(v) for v in df["retrieved_contexts"].tolist()]
        else:
            print(f"   ⚠️  Không tìm thấy cột 'retrieved_contexts' — dùng context rỗng")
            contexts_list = [["No context retrieved"]] * len(questions)

        # Lọc câu hỏi rỗng
        valid_idx = [i for i, q in enumerate(questions) if q and q != "nan"]
        if len(valid_idx) < len(questions):
            print(f"   ⚠️  Loại bỏ {len(questions) - len(valid_idx)} câu rỗng")
        questions      = [questions[i]      for i in valid_idx]
        answers        = [answers[i]        for i in valid_idx]
        ground_truths  = [ground_truths[i]  for i in valid_idx]
        contexts_list  = [contexts_list[i]  for i in valid_idx]
        df             = df.iloc[valid_idx].reset_index(drop=True)

        total = len(questions)
        print(f"   📊 Tổng số câu hợp lệ: {total}")

        # --- Chạy RAGAS ---
        print(f"\n🔬 Chạy RAGAS evaluation ({mode.upper()}) — {total} câu...")
        print(f"   LLM Judge : {EVAL_LLM_MODEL}")
        print(f"   Embedding : {EMBEDDING_MODEL} (Ollama)")

        eval_start = time.time()
        try:
            ragas_df = run_ragas_evaluation(questions, answers, contexts_list, ground_truths)
            eval_time = time.time() - eval_start
            print(f"✅ RAGAS evaluation xong trong {eval_time:.1f}s")

            # Gán lại question_text và mode
            ragas_df.insert(0, "question_text", questions)
            ragas_df["mode"] = mode

            # Tính RAGAS Score tổng hợp
            ragas_cols  = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
            valid_cols  = [c for c in ragas_cols if c in ragas_df.columns]
            ragas_df["ragas_score"] = ragas_df[valid_cols].mean(axis=1)

            # Giữ lại các cột timing/token gốc (nếu có)
            timing_cols = [
                "input_tokens", "output_tokens",
                "keyword_extraction_ms", "graph_search_ms",
                "retrieval_latency_ms", "generation_latency_ms", "total_latency_ms",
            ]
            for col in timing_cols:
                src_col = col.strip().replace("\n", "")
                # tìm cột khớp (bao gồm cột có newline)
                matched = [c for c in df.columns if c.strip().replace("\n", "") == src_col]
                if matched:
                    ragas_df[col] = df[matched[0]].values
                elif col not in ragas_df.columns:
                    ragas_df[col] = 0

            result_df = ragas_df

        except Exception as e:
            print(f"   ❌ Lỗi RAGAS: {e}")
            # Fallback: giữ timing data, để RAGAS columns là None
            result_df = df.copy()
            result_df.insert(0, "question_text", questions) if "question_text" not in result_df.columns else None
            result_df["mode"] = mode
            for col in ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]:
                result_df[col] = None

        all_mode_results[mode] = result_df

        # --- In tóm tắt mode ---
        print(f"\n📈 Kết quả {mode.upper()}:")
        for col_name, label in [
            ("input_tokens",          "Input Tokens        "),
            ("output_tokens",         "Output Tokens       "),
            ("keyword_extraction_ms", "Keyword Extraction  "),
            ("graph_search_ms",       "Graph Search        "),
            ("retrieval_latency_ms",  "Retrieval Latency   "),
            ("generation_latency_ms", "Generation Latency  "),
            ("total_latency_ms",      "Total Latency       "),
        ]:
            if col_name in result_df.columns:
                val = result_df[col_name].mean()
                suffix = "ms" if "ms" in col_name else ""
                print(f"   {label}: {val:.1f}{suffix}")

        for col in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            if col in result_df.columns:
                print(f"   {col:<25s}: {result_df[col].mean():.4f}")
        if "ragas_score" in result_df.columns:
            print(f"   {'RAGAS Score (avg)':<25s}: {result_df['ragas_score'].mean():.4f}")

        # Lưu checkpoint sau mỗi mode
        save_results(all_mode_results)

    xls.close()

    # ==================== SO SÁNH ====================
    if len(all_mode_results) > 1:
        print(f"\n{'=' * 70}")
        print(f"📊 SO SÁNH CÁC MODE: {', '.join(m.upper() for m in MODES if m in all_mode_results)}")
        print(f"{'=' * 70}")
        print_comparison_table(all_mode_results)
        print(f"{'=' * 70}")

    # Lưu kết quả cuối
    save_results(all_mode_results)

    print(f"\n💾 Kết quả đã lưu: {OUTPUT_FILE}")
    for mode in MODES:
        if mode in all_mode_results:
            print(f"   📄 Sheet '{mode.capitalize()[:31]}' — {len(all_mode_results[mode])} câu")
    print("   📄 Sheet 'Summary' — tổng hợp so sánh")
    print(f"\n✅ Re-evaluation hoàn tất!")


if __name__ == "__main__":
    main()
