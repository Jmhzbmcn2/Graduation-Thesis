"""
Benchmark Latency — LightRAG Hybrid Mode
Khóa Luận Tốt Nghiệp

Đo thời gian truy xuất trung bình (latency) trên N câu hỏi ở mode hybrid.
Kết quả sẽ được dùng làm **baseline** để so sánh với phương pháp cải tiến.

Đo 2 loại latency:
  1. Full latency   : Gọi LightRAG query bình thường (bao gồm cả LLM generation)
  2. Retrieval-only  : Gọi LightRAG với only_need_context=True (chỉ truy xuất, không sinh câu trả lời)

Output:
  - In bảng thống kê ra console
  - Lưu chi tiết từng câu hỏi ra file Excel

Cách dùng:
  python testcase/benchmark_latency_hybrid.py

Yêu cầu: LightRAG server đang chạy tại http://localhost:9621
"""

import os
import sys
import time
import statistics
import argparse
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=False)

# ======================== CẤU HÌNH ========================
LIGHTRAG_URL = os.getenv("LIGHTRAG_URL", "http://localhost:9621")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(_SCRIPT_DIR, "500_cases.csv")

# Số câu hỏi benchmark (None = dùng tất cả)
DEFAULT_N = 50

# Mode benchmark (có thể gán qua CLI)
DEFAULT_MODE = "hybrid"

# Query parameters (giữ giống cấu hình evaluation để kết quả nhất quán)
QUERY_TOP_K = 10

# Số lần warmup (không tính vào kết quả)
WARMUP_COUNT = 2

# Retry khi gặp lỗi
MAX_RETRIES = 3
# ===========================================================


def check_server():
    """Kiểm tra LightRAG server có đang chạy không."""
    try:
        resp = requests.get(f"{LIGHTRAG_URL}/health", timeout=5)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    # Thử endpoint khác
    try:
        resp = requests.get(f"{LIGHTRAG_URL}/", timeout=5)
        if resp.status_code in [200, 404, 405]:
            return True
    except Exception:
        pass

    return False


def query_lightrag(question: str, mode: str, only_context: bool = False, retries: int = MAX_RETRIES):
    """
    Gọi LightRAG API và trả về (response, elapsed_seconds).
    Nếu only_context=True, chỉ lấy context (retrieval-only, không qua LLM generation).
    """
    payload = {
        "query": question,
        "mode": mode,
        "stream": False,
        "top_k": QUERY_TOP_K,
        "only_need_context": only_context,
    }

    for attempt in range(retries):
        try:
            start = time.perf_counter()
            resp = requests.post(
                f"{LIGHTRAG_URL}/query",
                json=payload,
                timeout=300,
            )
            elapsed = time.perf_counter() - start

            resp.raise_for_status()
            response_text = resp.json().get("response", "")
            return response_text, elapsed

        except Exception as e:
            if attempt < retries - 1:
                print(f"    ⚠️ Lỗi lần {attempt+1}/{retries}: {e}")
                time.sleep(2)
            else:
                print(f"    ❌ Thất bại sau {retries} lần: {e}")
                return f"ERROR: {e}", -1.0

    return "ERROR: Unknown", -1.0


def run_benchmark(questions: list[str], mode: str, warmup: int = WARMUP_COUNT):
    """
    Chạy benchmark trên danh sách câu hỏi.

    Returns:
        list[dict]: Kết quả chi tiết từng câu hỏi, bao gồm:
            - question: Câu hỏi
            - full_latency_s: Thời gian full query (bao gồm LLM generation)
            - retrieval_latency_s: Thời gian retrieval-only (chỉ truy xuất context)
            - answer_preview: 100 ký tự đầu của câu trả lời
            - status: "OK" hoặc "ERROR"
    """
    results = []
    total = len(questions)

    # Warmup: chạy vài câu đầu để "khởi động" cache, connection pool, etc.
    if warmup > 0:
        warmup_questions = questions[:min(warmup, total)]
        print(f"\n🔥 Warmup ({len(warmup_questions)} câu, không tính vào kết quả)...")
        for i, q in enumerate(warmup_questions):
            print(f"  [warmup {i+1}/{len(warmup_questions)}] {q[:60]}...")
            query_lightrag(q, mode, only_context=False)
            query_lightrag(q, mode, only_context=True)
        print(f"  ✅ Warmup xong\n")

    print(f"🚀 Bắt đầu benchmark {total} câu hỏi (mode: {mode.upper()})...\n")

    for i, question in enumerate(questions):
        print(f"  [{i+1:>3}/{total}] {question[:70]}...")

        # 1. Full query (bao gồm LLM generation)
        answer, full_latency = query_lightrag(question, mode, only_context=False)

        # 2. Retrieval-only (chỉ lấy context, không qua LLM)
        _, retrieval_latency = query_lightrag(question, mode, only_context=True)

        status = "OK" if full_latency >= 0 and retrieval_latency >= 0 else "ERROR"
        answer_preview = answer[:100].replace("\n", " ") if isinstance(answer, str) else ""

        results.append({
            "question": question,
            "full_latency_s": round(full_latency, 4) if full_latency >= 0 else None,
            "retrieval_latency_s": round(retrieval_latency, 4) if retrieval_latency >= 0 else None,
            "answer_preview": answer_preview,
            "status": status,
        })

        # In latency inline
        if status == "OK":
            print(f"           Full: {full_latency:.2f}s | Retrieval: {retrieval_latency:.2f}s")
        else:
            print(f"           ❌ ERROR")

    return results


def compute_stats(values: list[float], label: str) -> dict:
    """Tính toán thống kê mô tả cho một danh sách giá trị latency."""
    if not values:
        return {}

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
    if not stats_list:
        print("  (Không có dữ liệu)")
        return

    # Header
    cols = ["metric", "count", "mean", "median", "std", "min", "max", "p95", "p99"]
    col_widths = {
        "metric": 22, "count": 7, "mean": 10, "median": 10,
        "std": 10, "min": 10, "max": 10, "p95": 10, "p99": 10,
    }

    header = "  " + "".join(f"{c:>{col_widths[c]}s}" for c in cols)
    print(header)
    print("  " + "─" * sum(col_widths.values()))

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
            row += f"{val:>{col_widths[c]}s}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Benchmark latency cho LightRAG")
    parser.add_argument("-n", "--num-questions", type=int, default=DEFAULT_N,
                        help=f"Số câu hỏi benchmark (default: {DEFAULT_N})")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE,
                        help=f"Mode để benchmark (hybrid, beam, local, global) (default: {DEFAULT_MODE})")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Bỏ qua bước warmup")
    parser.add_argument("--output", type=str, default=None,
                        help="Đường dẫn file output Excel (default: auto-generate)")
    args = parser.parse_args()

    n = args.num_questions
    mode = args.mode
    warmup = 0 if args.no_warmup else WARMUP_COUNT

    print("=" * 65)
    print(f"⏱️  BENCHMARK LATENCY — LightRAG {mode.upper()} Mode")
    print(f"   Khóa Luận Tốt Nghiệp")
    print(f"   Server: {LIGHTRAG_URL}")
    print(f"   Số câu hỏi: {n}")
    print(f"   Warmup: {warmup} câu")
    print("=" * 65)

    # 1. Kiểm tra server
    print(f"\n🔌 Kiểm tra LightRAG server...")
    if not check_server():
        print(f"   ❌ Không kết nối được tới {LIGHTRAG_URL}")
        print(f"   💡 Hãy chạy: lightrag-server --working-dir <path>")
        sys.exit(1)
    print(f"   ✅ Server đang chạy")

    # 2. Đọc test cases
    print(f"\n📂 Đọc file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Tổng số test cases: {len(df)}")

    # Lấy N câu hỏi
    if n is not None:
        df = df.head(n)
    questions = df["question"].astype(str).tolist()
    print(f"   Sử dụng: {len(questions)} câu")

    # 3. Chạy benchmark
    benchmark_start = time.time()
    results = run_benchmark(questions, mode, warmup=warmup)
    total_time = time.time() - benchmark_start

    # 4. Tính toán thống kê
    ok_results = [r for r in results if r["status"] == "OK"]
    error_count = len(results) - len(ok_results)

    full_latencies = [r["full_latency_s"] for r in ok_results if r["full_latency_s"] is not None]
    retrieval_latencies = [r["retrieval_latency_s"] for r in ok_results if r["retrieval_latency_s"] is not None]
    llm_gen_latencies = [
        round(r["full_latency_s"] - r["retrieval_latency_s"], 4)
        for r in ok_results
        if r["full_latency_s"] is not None and r["retrieval_latency_s"] is not None
    ]

    stats_list = [
        compute_stats(full_latencies, "Full (E2E)"),
        compute_stats(retrieval_latencies, "Retrieval-only"),
        compute_stats(llm_gen_latencies, "LLM Generation"),
    ]

    # 5. In kết quả
    print(f"\n{'=' * 65}")
    print(f"📊 KẾT QUẢ BENCHMARK — {mode.upper()} MODE")
    print(f"{'=' * 65}")
    print(f"   Tổng câu hỏi    : {len(results)}")
    print(f"   Thành công (OK)  : {len(ok_results)}")
    print(f"   Lỗi (ERROR)      : {error_count}")
    print(f"   Tổng thời gian   : {total_time:.1f}s")
    print()

    print(f"📈 THỐNG KÊ LATENCY (đơn vị: giây)")
    print_stats_table(stats_list)

    # Thông tin bổ sung
    if full_latencies:
        print(f"\n   💡 Throughput trung bình: {len(ok_results) / sum(full_latencies):.2f} queries/s (full)")
        print(f"   💡 Throughput trung bình: {len(ok_results) / sum(retrieval_latencies):.2f} queries/s (retrieval-only)")

    print(f"\n{'=' * 65}")

    # 6. Lưu kết quả ra Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(_SCRIPT_DIR, f"benchmark_latency_{mode}_{timestamp}.xlsx")

    # Sheet 1: Chi tiết từng câu
    detail_df = pd.DataFrame(results)
    # Thêm cột LLM generation latency
    detail_df["llm_gen_latency_s"] = detail_df.apply(
        lambda r: round(r["full_latency_s"] - r["retrieval_latency_s"], 4)
        if r["full_latency_s"] is not None and r["retrieval_latency_s"] is not None
        else None,
        axis=1
    )

    # Sheet 2: Thống kê tổng hợp
    stats_df = pd.DataFrame(stats_list)

    # Sheet 3: Metadata
    meta_df = pd.DataFrame([{
        "timestamp": timestamp,
        "mode": mode,
        "total_questions": len(results),
        "ok_count": len(ok_results),
        "error_count": error_count,
        "total_time_s": round(total_time, 2),
        "warmup_count": warmup,
        "top_k": QUERY_TOP_K,
        "server_url": LIGHTRAG_URL,
        "input_file": INPUT_FILE,
    }])

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        detail_df.to_excel(writer, sheet_name="Detail", index=False)
        stats_df.to_excel(writer, sheet_name="Statistics", index=False)
        meta_df.to_excel(writer, sheet_name="Metadata", index=False)

    print(f"💾 Kết quả đã lưu: {output_file}")
    print(f"   📄 Sheet 'Detail'     — Chi tiết {len(results)} câu")
    print(f"   📄 Sheet 'Statistics' — Thống kê latency")
    print(f"   📄 Sheet 'Metadata'   — Thông tin cấu hình")


if __name__ == "__main__":
    main()
