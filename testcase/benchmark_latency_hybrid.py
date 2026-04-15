"""
Benchmark Latency — LightRAG Hybrid Mode (v2)
Khóa Luận Tốt Nghiệp

Đo thời gian truy xuất trung bình (latency) trên N câu hỏi ở mode hybrid.
Kết quả sẽ được dùng làm **baseline** để so sánh với phương pháp cải tiến.

Pipeline LightRAG hybrid query gồm 3 giai đoạn chính:
  1. Keyword Extraction  : LLM trích xuất từ khóa (HL + LL) từ câu hỏi
  2. Graph Retrieval      : Vector Search + đồ thị (entities, relations, chunks)
  3. LLM Generation       : Sinh câu trả lời từ context

Script này đo ĐỘC LẬP TỪNG GIAI ĐOẠN bằng cách gọi trực tiếp Python API
của LightRAG (không qua HTTP) để tránh nhiễu mạng và cache artifacts.

Output:
  - In bảng thống kê ra console
  - Lưu chi tiết từng câu hỏi ra file Excel

Cách dùng:
  python testcase/benchmark_latency_hybrid.py          # 50 câu (mặc định)
  python testcase/benchmark_latency_hybrid.py -n 100   # 100 câu
  python testcase/benchmark_latency_hybrid.py --no-warmup  # bỏ warmup

Yêu cầu: File .env cấu hình đúng (LLM model, embedding model, working_dir)
"""

import os
import sys
import time
import statistics
import argparse
import asyncio
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=False)

# ======================== CẤU HÌNH ========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(_SCRIPT_DIR, "500_cases.csv")

# Working directory của LightRAG (nơi chứa graph data)
WORKING_DIR = os.getenv("WORKING_DIR", os.path.join(
    os.path.dirname(_SCRIPT_DIR), "medical_rag", "medical_rag_v2"
))

# Số câu hỏi benchmark (mặc định)
DEFAULT_N = 5

# Mode benchmark
MODE = "hybrid"

# Query parameters
QUERY_TOP_K = 10

# Số lần warmup (không tính vào kết quả)
WARMUP_COUNT = 2
# ===========================================================


def create_lightrag_instance():
    """Khởi tạo LightRAG instance trực tiếp (không qua HTTP server)."""
    from lightrag.lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc

    # Đọc cấu hình từ .env (giống server)
    llm_binding_host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
    llm_model = os.getenv("LLM_MODEL", "gemma3:12b")
    embedding_binding_host = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    max_tokens = int(os.getenv("MAX_TOKENS", "32768"))
    max_embed_tokens = int(os.getenv("MAX_EMBED_TOKENS", "8192"))

    print(f"   LLM Model       : {llm_model} ({llm_binding_host})")
    print(f"   Embedding Model  : {embedding_model} ({embedding_binding_host})")
    print(f"   Working Dir      : {WORKING_DIR}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model,
        llm_model_kwargs={
            "host": llm_binding_host,
            "options": {"num_ctx": max_tokens},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_embed_tokens,
            func=lambda texts: ollama_embed(
                texts, embed_model=embedding_model, host=embedding_binding_host
            ),
        ),
    )
    return rag


async def benchmark_single_query(rag, question: str, mode: str):
    """
    Đo latency chi tiết cho 1 câu hỏi, tách biệt từng giai đoạn.

    Returns:
        dict: {
            keyword_extraction_s: Thời gian LLM trích từ khóa,
            graph_retrieval_s: Thời gian vector search + graph retrieval,
            context_build_s: Thời gian build context string,
            total_retrieval_s: Tổng thời gian retrieval (keyword + graph + context),
            full_e2e_s: Tổng thời gian end-to-end (bao gồm LLM generation),
            hl_keywords: High-level keywords,
            ll_keywords: Low-level keywords,
            num_entities: Số entities tìm được,
            num_relations: Số relations tìm được,
        }
    """
    from lightrag.base import QueryParam
    from lightrag.operate import (
        get_keywords_from_query,
        _perform_kg_search,
        _apply_token_truncation,
        _build_context_str,
    )
    from dataclasses import asdict
    from functools import partial

    query_param = QueryParam(mode=mode, top_k=QUERY_TOP_K)

    # Chuẩn bị các storage references
    knowledge_graph = rag.chunk_entity_relation_graph
    entities_vdb = rag.entities_vdb
    relationships_vdb = rag.relationships_vdb
    text_chunks_db = rag.text_chunks
    chunks_vdb = rag.chunks_vdb
    hashing_kv = rag.llm_response_cache

    global_config = {
        k: v for k, v in asdict(rag).items()
    }
    # Restore non-serializable objects
    global_config["llm_model_func"] = rag.llm_model_func
    global_config["embedding_func"] = rag.embedding_func
    global_config["tokenizer"] = rag.tokenizer

    # ─── Giai đoạn 1: Keyword Extraction (LLM) ───
    t1_start = time.perf_counter()
    hl_keywords, ll_keywords = await get_keywords_from_query(
        question, query_param, global_config, hashing_kv
    )
    t1_end = time.perf_counter()
    keyword_time = t1_end - t1_start

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # ─── Giai đoạn 2: Graph Retrieval (Vector + Graph) ───
    t2_start = time.perf_counter()
    search_result = await _perform_kg_search(
        question,
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )
    t2_end = time.perf_counter()
    graph_time = t2_end - t2_start

    num_entities = len(search_result.get("final_entities", []))
    num_relations = len(search_result.get("final_relations", []))

    # ─── Giai đoạn 3: Context Building (Token truncation + formatting) ───
    t3_start = time.perf_counter()
    truncation_result = await _apply_token_truncation(
        search_result, query_param, global_config
    )
    t3_end = time.perf_counter()
    context_time = t3_end - t3_start

    total_retrieval = keyword_time + graph_time + context_time

    # ─── Giai đoạn 4 (Tùy chọn): Full E2E (bao gồm LLM Generation) ───
    t4_start = time.perf_counter()
    result = await rag.aquery(question, param=query_param)
    t4_end = time.perf_counter()
    full_e2e_time = t4_end - t4_start

    return {
        "keyword_extraction_s": round(keyword_time, 4),
        "graph_retrieval_s": round(graph_time, 4),
        "context_build_s": round(context_time, 4),
        "total_retrieval_s": round(total_retrieval, 4),
        "full_e2e_s": round(full_e2e_time, 4),
        "llm_generation_s": round(full_e2e_time - total_retrieval, 4),
        "hl_keywords": str(hl_keywords),
        "ll_keywords": str(ll_keywords),
        "num_entities": num_entities,
        "num_relations": num_relations,
    }


async def run_benchmark(rag, questions: list[str], mode: str, warmup: int = WARMUP_COUNT):
    """Chạy benchmark trên danh sách câu hỏi."""
    results = []
    total = len(questions)

    # Warmup
    if warmup > 0:
        warmup_qs = questions[:min(warmup, total)]
        print(f"\n🔥 Warmup ({len(warmup_qs)} câu, không tính vào kết quả)...")
        for i, q in enumerate(warmup_qs):
            print(f"  [warmup {i+1}/{len(warmup_qs)}] {q[:60]}...")
            try:
                await benchmark_single_query(rag, q, mode)
            except Exception as e:
                print(f"    ⚠️ Warmup error: {e}")
        print(f"  ✅ Warmup xong\n")

    print(f"🚀 Bắt đầu benchmark {total} câu hỏi (mode: {mode.upper()})...\n")
    col_fmt = "{:>10s}" * 5
    header = f"  {'#':>5s}  " + col_fmt.format("Keyword", "Graph", "Context", "Total Ret", "Full E2E")
    print(header)
    print(f"  {'─' * 60}")

    for i, question in enumerate(questions):
        try:
            r = await benchmark_single_query(rag, question, mode)
            r["question"] = question
            r["status"] = "OK"

            # In kết quả dạng bảng compact
            row = f"  [{i+1:>3}/{total}]  "
            row += f"{r['keyword_extraction_s']:>8.2f}s "
            row += f"{r['graph_retrieval_s']:>8.2f}s "
            row += f"{r['context_build_s']:>8.2f}s "
            row += f"{r['total_retrieval_s']:>8.2f}s "
            row += f"{r['full_e2e_s']:>8.2f}s "
            row += f"  E:{r['num_entities']} R:{r['num_relations']}"
            print(row)

        except Exception as e:
            print(f"  [{i+1:>3}/{total}]  ❌ ERROR: {e}")
            r = {
                "question": question,
                "keyword_extraction_s": None,
                "graph_retrieval_s": None,
                "context_build_s": None,
                "total_retrieval_s": None,
                "full_e2e_s": None,
                "llm_generation_s": None,
                "hl_keywords": "",
                "ll_keywords": "",
                "num_entities": 0,
                "num_relations": 0,
                "status": "ERROR",
            }

        results.append(r)

    return results


def compute_stats(values: list[float], label: str) -> dict:
    """Tính toán thống kê mô tả."""
    if not values:
        return {"metric": label}

    return {
        "metric": label,
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "p95": round(sorted(values)[int(len(values) * 0.95)], 4) if len(values) >= 20 else None,
        "p99": round(sorted(values)[int(len(values) * 0.99)], 4) if len(values) >= 100 else None,
    }


def print_stats_table(stats_list: list[dict]):
    """In bảng thống kê ra console."""
    cols = ["metric", "count", "mean", "median", "std", "min", "max", "p95"]
    widths = {"metric": 24, "count": 7, "mean": 10, "median": 10, "std": 10, "min": 10, "max": 10, "p95": 10}

    header = "  " + "".join(f"{c:>{widths[c]}s}" for c in cols)
    print(header)
    print("  " + "─" * sum(widths.values()))

    for stats in stats_list:
        row = "  "
        for c in cols:
            val = stats.get(c, "")
            if val is None:
                val = "N/A"
            elif isinstance(val, float):
                val = f"{val:.4f}"
            elif isinstance(val, int):
                val = str(val)
            row += f"{val:>{widths[c]}s}"
        print(row)


async def async_main():
    parser = argparse.ArgumentParser(description="Benchmark latency cho LightRAG Hybrid mode")
    parser.add_argument("-n", "--num-questions", type=int, default=DEFAULT_N,
                        help=f"Số câu hỏi benchmark (default: {DEFAULT_N})")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Bỏ qua bước warmup")
    parser.add_argument("--output", type=str, default=None,
                        help="Đường dẫn file output Excel")
    args = parser.parse_args()

    n = args.num_questions
    warmup = 0 if args.no_warmup else WARMUP_COUNT

    print("=" * 70)
    print(f"⏱️  BENCHMARK LATENCY — LightRAG {MODE.upper()} Mode (v2 — Per-Stage)")
    print(f"   Khóa Luận Tốt Nghiệp")
    print(f"   Đo tách biệt: Keyword Extraction | Graph Retrieval | LLM Generation")
    print(f"   Số câu hỏi: {n} | Warmup: {warmup}")
    print("=" * 70)

    # 1. Khởi tạo LightRAG
    print(f"\n🔧 Khởi tạo LightRAG instance...")
    try:
        rag = create_lightrag_instance()
        await rag.initialize_storages()
        print(f"   ✅ Khởi tạo thành công")
    except Exception as e:
        print(f"   ❌ Lỗi khởi tạo: {e}")
        print(f"   💡 Kiểm tra file .env và working_dir: {WORKING_DIR}")
        sys.exit(1)

    # 2. Đọc test cases
    print(f"\n📂 Đọc file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Tổng số test cases: {len(df)}")
    if n is not None:
        df = df.head(n)
    questions = df["question"].astype(str).tolist()
    print(f"   Sử dụng: {len(questions)} câu")

    # 3. Chạy benchmark
    benchmark_start = time.time()
    results = await run_benchmark(rag, questions, MODE, warmup=warmup)
    total_time = time.time() - benchmark_start

    # 4. Finalize storages
    await rag.finalize_storages()

    # 5. Tính toán thống kê
    ok_results = [r for r in results if r["status"] == "OK"]
    error_count = len(results) - len(ok_results)

    def extract_values(key):
        return [r[key] for r in ok_results if r.get(key) is not None]

    stats_list = [
        compute_stats(extract_values("keyword_extraction_s"), "1. Keyword Extract (LLM)"),
        compute_stats(extract_values("graph_retrieval_s"),    "2. Graph Retrieval"),
        compute_stats(extract_values("context_build_s"),      "3. Context Build"),
        compute_stats(extract_values("total_retrieval_s"),     "── Total Retrieval ──"),
        compute_stats(extract_values("llm_generation_s"),      "4. LLM Generation"),
        compute_stats(extract_values("full_e2e_s"),            "══ Full E2E ══"),
    ]

    # 6. In kết quả
    print(f"\n{'=' * 70}")
    print(f"📊 KẾT QUẢ BENCHMARK — {MODE.upper()} MODE")
    print(f"{'=' * 70}")
    print(f"   Tổng câu hỏi     : {len(results)}")
    print(f"   Thành công (OK)   : {len(ok_results)}")
    print(f"   Lỗi (ERROR)       : {error_count}")
    print(f"   Tổng thời gian    : {total_time:.1f}s")

    # Thống kê entities/relations
    avg_entities = statistics.mean([r["num_entities"] for r in ok_results]) if ok_results else 0
    avg_relations = statistics.mean([r["num_relations"] for r in ok_results]) if ok_results else 0
    print(f"   Avg Entities/query: {avg_entities:.1f}")
    print(f"   Avg Relations/query: {avg_relations:.1f}")
    print()

    print(f"📈 THỐNG KÊ LATENCY (đơn vị: giây)")
    print_stats_table(stats_list)

    # Phân tích tỷ lệ thời gian
    retrieval_vals = extract_values("total_retrieval_s")
    full_vals = extract_values("full_e2e_s")
    keyword_vals = extract_values("keyword_extraction_s")
    graph_vals = extract_values("graph_retrieval_s")

    if retrieval_vals and full_vals:
        avg_retrieval = statistics.mean(retrieval_vals)
        avg_full = statistics.mean(full_vals)
        avg_keyword = statistics.mean(keyword_vals)
        avg_graph = statistics.mean(graph_vals)

        print(f"\n📊 PHÂN TÍCH TỶ LỆ THỜI GIAN:")
        print(f"   Keyword Extraction : {avg_keyword:.2f}s ({avg_keyword/avg_full*100:.1f}% of E2E)")
        print(f"   Graph Retrieval    : {avg_graph:.2f}s ({avg_graph/avg_full*100:.1f}% of E2E)")
        print(f"   LLM Generation     : {avg_full-avg_retrieval:.2f}s ({(avg_full-avg_retrieval)/avg_full*100:.1f}% of E2E)")

    print(f"\n{'=' * 70}")

    # 7. Lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or os.path.join(_SCRIPT_DIR, f"benchmark_latency_{MODE}_{timestamp}.xlsx")

    detail_df = pd.DataFrame(results)
    stats_df = pd.DataFrame(stats_list)
    meta_df = pd.DataFrame([{
        "timestamp": timestamp,
        "mode": MODE,
        "total_questions": len(results),
        "ok_count": len(ok_results),
        "error_count": error_count,
        "total_time_s": round(total_time, 2),
        "warmup_count": warmup,
        "top_k": QUERY_TOP_K,
        "working_dir": WORKING_DIR,
        "input_file": INPUT_FILE,
    }])

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        detail_df.to_excel(writer, sheet_name="Detail", index=False)
        stats_df.to_excel(writer, sheet_name="Statistics", index=False)
        meta_df.to_excel(writer, sheet_name="Metadata", index=False)

    print(f"💾 Kết quả đã lưu: {output_file}")
    print(f"   📄 Sheet 'Detail'     — Chi tiết {len(results)} câu")
    print(f"   📄 Sheet 'Statistics' — Thống kê latency từng giai đoạn")
    print(f"   📄 Sheet 'Metadata'   — Thông tin cấu hình")


if __name__ == "__main__":
    asyncio.run(async_main())
