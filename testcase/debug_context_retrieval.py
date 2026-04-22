"""
Debug Context Retrieval — Hybrid vs Mix vs Beam
================================================
Script này dùng ĐÚNG config từ eval_ragas_hybrid_mix_beam.py để:
  1. Gửi 1 câu hỏi qua cả 3 mode
  2. In raw context từ server (chưa parse)
  3. In parsed context sau extract_chunks_from_context()
  4. In overlap với ground truth
  5. In cấu trúc section của context (để thấy format khác nhau)

Cách chạy:
  python testcase/debug_context_retrieval.py
  python testcase/debug_context_retrieval.py --query "Câu hỏi tùy chỉnh"
  python testcase/debug_context_retrieval.py --modes hybrid beam
  python testcase/debug_context_retrieval.py --save-raw  # lưu raw context ra file
"""

import os
import re
import sys
import json
import argparse
import textwrap
import requests

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — Đồng bộ 100% với eval_ragas_hybrid_mix_beam.py
# ──────────────────────────────────────────────────────────────────────────────
LIGHTRAG_URL = "http://localhost:9621"

# Beam params (giống hệt eval script — v3 config)
BEAM_BEAM_WIDTH        = 10
BEAM_MAX_DEPTH         = 1
BEAM_CHUNK_TOP_K       = 15
BEAM_PRUNING_THRESHOLD = 0.25
BEAM_ANCHOR_ALPHA      = 0.7
BEAM_CHUNK_ALPHA       = 0.7
RELATED_CHUNK_NUMBER   = 5

# Default params cho hybrid/mix
DEFAULT_TOP_K = 10

# ──────────────────────────────────────────────────────────────────────────────
# Câu hỏi mặc định để test (thay bằng --query khi chạy)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_QUERY = "Bệnh nấm da đầu có những thể nào?"
DEFAULT_GROUND_TRUTH = (
    "Bệnh nấm da đầu có thể chia làm 3 thể: thể viêm, thể không viêm và thể Favus."
)

# ──────────────────────────────────────────────────────────────────────────────
# Helper: parse / extract (copy từ eval_ragas_hybrid_mix_beam.py)
# ──────────────────────────────────────────────────────────────────────────────
def extract_chunks_from_context(raw_context: str) -> str:
    """
    Trích xuất context từ LightRAG cho RAGAS evaluation.
    CHUNKS TRƯỚC, GRAPH SAU — đồng bộ với eval script.
    """
    chunk_start = raw_context.find("Document Chunks")
    if chunk_start == -1:
        return raw_context  # naive mode → trả nguyên gốc

    ref_start = raw_context.find("Reference Document List", chunk_start)
    chunk_section = (
        raw_context[chunk_start:ref_start]
        if ref_start != -1
        else raw_context[chunk_start:]
    )

    chunk_texts = []
    json_match = re.search(r"```json\s*\n(.*?)```", chunk_section, re.DOTALL)
    if json_match:
        for line in json_match.group(1).strip().split("\n"):
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
    rel_start    = raw_context.find("Knowledge Graph Data (Relationship):")

    if entity_start != -1:
        end_idx = rel_start if rel_start != -1 else chunk_start
        entity_section = raw_context[entity_start:end_idx]
        json_match = re.search(r"```json\s*\n(.*?)```", entity_section, re.DOTALL)
        if json_match:
            for line in json_match.group(1).strip().split("\n"):
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
        json_match = re.search(r"```json\s*\n(.*?)```", rel_section, re.DOTALL)
        if json_match:
            for line in json_match.group(1).strip().split("\n"):
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        e1   = obj.get("entity1", "")
                        e2   = obj.get("entity2", "")
                        desc = obj.get("description", "")
                        if e1 and e2 and desc:
                            graph_texts.append(
                                f"Mối quan hệ giữa [{e1}] và [{e2}] là: {desc}"
                            )
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


# ──────────────────────────────────────────────────────────────────────────────
# Phân tích cấu trúc sections của raw context
# ──────────────────────────────────────────────────────────────────────────────
SECTION_MARKERS = [
    "Knowledge Graph Data (Entity):",
    "Knowledge Graph Data (Relationship):",
    "Document Chunks",
    "Reference Document List",
    # beam-specific markers (nếu có)
    "Beam Search Path",
    "Graph Path",
]


def detect_sections(raw: str) -> list[dict]:
    """Trả về list các section tìm thấy trong raw context."""
    found = []
    for marker in SECTION_MARKERS:
        pos = raw.find(marker)
        if pos != -1:
            found.append({"marker": marker, "pos": pos})

    # Tìm thêm mọi section bằng pattern (tiêu đề markdown)
    for m in re.finditer(r"^(#{1,3}\s+.+)$", raw, re.MULTILINE):
        found.append({"marker": m.group(1).strip(), "pos": m.start()})

    found.sort(key=lambda x: x["pos"])
    return found


def count_json_objects_in_section(raw: str, start: int, end: int) -> int:
    """Đếm số JSON object trong đoạn [start:end] của raw."""
    section = raw[start:end]
    json_match = re.search(r"```json\s*\n(.*?)```", section, re.DOTALL)
    if not json_match:
        return 0
    count = 0
    for line in json_match.group(1).strip().split("\n"):
        line = line.strip()
        if line:
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                pass
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Overlap với ground truth
# ──────────────────────────────────────────────────────────────────────────────
def compute_overlap(text: str, ground_truth: str) -> dict:
    """Word-level overlap giữa text và ground truth."""
    gt_words  = set(ground_truth.lower().split())
    ctx_words = set(text.lower().split())
    if not gt_words:
        return {"overlap_pct": 0.0, "matched_words": [], "total_gt_words": 0}

    matched = gt_words & ctx_words
    return {
        "overlap_pct":    len(matched) / len(gt_words),
        "matched_words":  sorted(matched),
        "total_gt_words": len(gt_words),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Build payload — ĐÚNG config cho từng mode
# ──────────────────────────────────────────────────────────────────────────────
def build_payload(query: str, mode: str) -> dict:
    if mode == "beam":
        return {
            "query":               query,
            "mode":                mode,
            "stream":              False,
            "top_k":               DEFAULT_TOP_K,
            "beam_width":          BEAM_BEAM_WIDTH,
            "beam_max_depth":      BEAM_MAX_DEPTH,
            "chunk_top_k":         BEAM_CHUNK_TOP_K,
            "pruning_threshold":   BEAM_PRUNING_THRESHOLD,
            "anchor_alpha":        BEAM_ANCHOR_ALPHA,
            "chunk_alpha":         BEAM_CHUNK_ALPHA,
            "related_chunk_number": RELATED_CHUNK_NUMBER,
            "enable_rerank":       False,
            "include_context":     True,
        }
    else:
        return {
            "query":           query,
            "mode":            mode,
            "stream":          False,
            "top_k":           DEFAULT_TOP_K,
            "enable_rerank":   False,
            "include_context": True,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Query LightRAG
# ──────────────────────────────────────────────────────────────────────────────
def query_lightrag(query: str, mode: str, timeout: int = 180) -> dict:
    payload = build_payload(query, mode)
    print(f"\n  📤 Payload gửi lên server ({mode.upper()}):")
    for k, v in payload.items():
        if k != "query":
            print(f"      {k}: {v}")

    resp = requests.post(
        f"{LIGHTRAG_URL}/query",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────────────────────────────────────────────────────────
# Printer helpers
# ──────────────────────────────────────────────────────────────────────────────
DIVIDER  = "─" * 80
DIVIDER2 = "═" * 80

def wrap(text: str, width: int = 100, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=prefix)


def print_section_header(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def print_mode_banner(mode: str):
    print(f"\n{DIVIDER2}")
    print(f"  🔍 MODE: {mode.upper()}")
    print(DIVIDER2)


# ──────────────────────────────────────────────────────────────────────────────
# Main debug routine cho 1 mode
# ──────────────────────────────────────────────────────────────────────────────
def debug_mode(
    mode: str,
    query: str,
    ground_truth: str,
    save_raw: bool = False,
    output_dir: str = ".",
):
    print_mode_banner(mode)

    # 1. Query server
    try:
        resp_json = query_lightrag(query, mode)
    except Exception as e:
        print(f"\n  ❌ Lỗi kết nối LightRAG: {e}")
        return None

    answer      = resp_json.get("response", "")
    raw_context = resp_json.get("context", "")
    timing      = resp_json.get("timing") or {}
    token_cnt   = resp_json.get("token_counts") or {}

    # ── 2. Timing & token info ──────────────────────────────────────────────
    print_section_header("⏱  Timing & Token Info")
    print(f"    keyword_extraction_ms : {timing.get('keyword_extraction_ms', '?')}")
    print(f"    graph_search_ms       : {timing.get('graph_search_ms', '?')}")
    print(f"    retrieval_ms          : {timing.get('retrieval_ms', '?')}")
    print(f"    generation_ms         : {timing.get('generation_ms', '?')}")
    print(f"    total_ms              : {timing.get('total_ms', '?')}")
    print(f"    input_tokens          : {token_cnt.get('input_tokens', '?')}")
    print(f"    output_tokens         : {token_cnt.get('output_tokens', '?')}")

    # ── 3. Cấu trúc raw context ─────────────────────────────────────────────
    print_section_header("📐 Cấu trúc Raw Context (Sections found)")
    if not raw_context:
        print("    ⚠️  raw_context TRỐNG (server không trả context)")
    else:
        sections = detect_sections(raw_context)
        if sections:
            for s in sections:
                print(f"    pos={s['pos']:6d}  →  {s['marker']}")
        else:
            print("    ⚠️  Không tìm thấy section marker nào trong raw context")

        print(f"\n    📏 raw_context length: {len(raw_context)} chars")

        # Đếm số entities, relations, chunks
        entity_start = raw_context.find("Knowledge Graph Data (Entity):")
        rel_start    = raw_context.find("Knowledge Graph Data (Relationship):")
        chunk_start  = raw_context.find("Document Chunks")

        if entity_start != -1:
            end = rel_start if rel_start != -1 and rel_start > entity_start else chunk_start if chunk_start != -1 else len(raw_context)
            n = count_json_objects_in_section(raw_context, entity_start, end)
            print(f"    🔵 Entities trong context    : {n}")

        if rel_start != -1:
            end = chunk_start if chunk_start != -1 and chunk_start > rel_start else len(raw_context)
            n = count_json_objects_in_section(raw_context, rel_start, end)
            print(f"    🔗 Relations trong context   : {n}")

        if chunk_start != -1:
            ref_start = raw_context.find("Reference Document List", chunk_start)
            end = ref_start if ref_start != -1 else len(raw_context)
            n = count_json_objects_in_section(raw_context, chunk_start, end)
            print(f"    📄 Chunks trong context      : {n}")
        else:
            print(f"    ⚠️  Không tìm thấy 'Document Chunks' section")

    # ── 4. Raw context preview ──────────────────────────────────────────────
    print_section_header("📄 Raw Context Preview (500 chars đầu)")
    if raw_context:
        print(wrap(raw_context[:500], width=110))
        if len(raw_context) > 500:
            print(f"\n    ... [{len(raw_context) - 500} chars còn lại]")
    else:
        print("    (trống)")

    # ── 5. Parsed context (sau extract_chunks_from_context) ─────────────────
    parsed_ctx = extract_chunks_from_context(raw_context) if raw_context else ""
    print_section_header("✅ Parsed Context (sau extract_chunks_from_context)")
    print(f"    📏 Length: {len(parsed_ctx)} chars")
    if parsed_ctx:
        print(f"\n    Preview (600 chars):")
        print(wrap(parsed_ctx[:600], width=110))
        if len(parsed_ctx) > 600:
            print(f"\n    ... [{len(parsed_ctx) - 600} chars còn lại]")
    else:
        print("    ⚠️  parsed_ctx TRỐNG sau extract!")

    # ── 6. Ground truth overlap ──────────────────────────────────────────────
    print_section_header("🎯 Ground Truth Overlap")
    if ground_truth.strip():
        ov_raw    = compute_overlap(raw_context, ground_truth)
        ov_parsed = compute_overlap(parsed_ctx, ground_truth)

        print(f"    Ground Truth : {ground_truth[:120]}")
        print()
        print(f"    Raw context   overlap: {ov_raw['overlap_pct']:.2%}  "
              f"({len(ov_raw['matched_words'])}/{ov_raw['total_gt_words']} words)")
        print(f"    Parsed context overlap: {ov_parsed['overlap_pct']:.2%}  "
              f"({len(ov_parsed['matched_words'])}/{ov_parsed['total_gt_words']} words)")

        if ov_parsed["matched_words"]:
            print(f"\n    ✅ Matched words: {', '.join(ov_parsed['matched_words'][:20])}")
        if ground_truth.lower().split():
            gt_words  = set(ground_truth.lower().split())
            ctx_words = set(parsed_ctx.lower().split())
            missing   = gt_words - ctx_words
            if missing:
                print(f"    ❌ Missing words: {', '.join(sorted(missing)[:20])}")
    else:
        print("    (Không có ground truth để so sánh)")

    # ── 7. Answer preview ───────────────────────────────────────────────────
    print_section_header("💬 Answer Preview")
    print(wrap(answer[:400], width=110))
    if len(answer) > 400:
        print(f"\n    ... [{len(answer) - 400} chars còn lại]")

    # ── 8. Lưu raw context ra file nếu cần ──────────────────────────────────
    if save_raw and raw_context:
        fname = os.path.join(output_dir, f"debug_raw_context_{mode}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"MODE: {mode.upper()}\n")
            f.write(f"QUERY: {query}\n")
            f.write("=" * 80 + "\n")
            f.write(raw_context)
        print(f"\n  💾 Raw context → {fname}")

        fname2 = os.path.join(output_dir, f"debug_parsed_context_{mode}.txt")
        with open(fname2, "w", encoding="utf-8") as f:
            f.write(f"MODE: {mode.upper()}\n")
            f.write(f"QUERY: {query}\n")
            f.write("=" * 80 + "\n")
            f.write(parsed_ctx)
        print(f"  💾 Parsed context → {fname2}")

    return {
        "mode":             mode,
        "raw_len":          len(raw_context),
        "parsed_len":       len(parsed_ctx),
        "has_chunks":       raw_context.find("Document Chunks") != -1,
        "has_entities":     raw_context.find("Knowledge Graph Data (Entity):") != -1,
        "has_relations":    raw_context.find("Knowledge Graph Data (Relationship):") != -1,
        "gt_overlap_raw":   compute_overlap(raw_context, ground_truth)["overlap_pct"] if ground_truth.strip() else None,
        "gt_overlap_parsed": compute_overlap(parsed_ctx, ground_truth)["overlap_pct"] if ground_truth.strip() else None,
        "timing":           timing,
        "token_counts":     token_cnt,
    }


# ──────────────────────────────────────────────────────────────────────────────
# So sánh tổng hợp
# ──────────────────────────────────────────────────────────────────────────────
def print_comparison_summary(results: list[dict]):
    print(f"\n{DIVIDER2}")
    print("  📊 TỔNG HỢP SO SÁNH 3 MODE")
    print(DIVIDER2)

    w = 12
    header  = f"  {'Metric':<30}"
    header += "".join(f"{r['mode'].upper():>{w}}" for r in results)
    print(header)
    print(f"  {'─' * (30 + w * len(results))}")

    rows = [
        ("Raw context length (chars)", "raw_len",           False),
        ("Parsed context length (chars)", "parsed_len",     False),
        ("Has 'Document Chunks'",      "has_chunks",         None),
        ("Has Entity data",            "has_entities",       None),
        ("Has Relation data",          "has_relations",      None),
        ("GT overlap (raw)",           "gt_overlap_raw",     True),
        ("GT overlap (parsed)",        "gt_overlap_parsed",  True),
    ]

    for label, key, higher_is_better in rows:
        row = f"  {label:<30}"
        for r in results:
            val = r.get(key)
            if val is None:
                row += f"{'N/A':>{w}}"
            elif isinstance(val, bool):
                row += f"{'✅' if val else '❌':>{w}}"
            elif isinstance(val, float):
                row += f"{val:>{w}.2%}"
            else:
                row += f"{val:>{w}}"
        print(row)

    # Timing summary
    print(f"\n  {'─' * (30 + w * len(results))}")
    for key, label in [
        ("retrieval_ms", "Retrieval (ms)"),
        ("generation_ms", "Generation (ms)"),
        ("total_ms", "Total (ms)"),
    ]:
        row = f"  {label:<30}"
        for r in results:
            v = r.get("timing", {}).get(key, "?")
            row += f"{str(v):>{w}}"
        print(row)

    for key, label in [
        ("input_tokens", "Input tokens"),
        ("output_tokens", "Output tokens"),
    ]:
        row = f"  {label:<30}"
        for r in results:
            v = r.get("token_counts", {}).get(key, "?")
            row += f"{str(v):>{w}}"
        print(row)

    print(f"\n{DIVIDER2}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug context retrieval cho 3 mode: hybrid, mix, beam"
    )
    parser.add_argument(
        "--query", "-q",
        default=DEFAULT_QUERY,
        help="Câu hỏi để debug (mặc định: câu hỏi về nấm da đầu)",
    )
    parser.add_argument(
        "--ground-truth", "-gt",
        default=DEFAULT_GROUND_TRUTH,
        help="Ground truth để tính overlap (mặc định: câu trả lời về nấm da đầu)",
    )
    parser.add_argument(
        "--modes", "-m",
        nargs="+",
        default=["hybrid", "mix", "beam"],
        choices=["hybrid", "mix", "beam"],
        help="Danh sách mode cần debug (mặc định: hybrid mix beam)",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Lưu raw context và parsed context ra file .txt",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Thư mục lưu file output (mặc định: cùng thư mục script)",
    )
    parser.add_argument(
        "--show-full-raw",
        action="store_true",
        help="In toàn bộ raw context (mặc định chỉ in 500 chars preview)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(DIVIDER2)
    print("  🔬 DEBUG CONTEXT RETRIEVAL — Hybrid / Mix / Beam")
    print(DIVIDER2)
    print(f"  Query      : {args.query}")
    print(f"  Ground Truth: {args.ground_truth[:100]}")
    print(f"  Modes      : {args.modes}")
    print(f"  LightRAG   : {LIGHTRAG_URL}")
    print(f"\n  Beam config:")
    print(f"    beam_width={BEAM_BEAM_WIDTH}, max_depth={BEAM_MAX_DEPTH}")
    print(f"    chunk_top_k={BEAM_CHUNK_TOP_K}, pruning_threshold={BEAM_PRUNING_THRESHOLD}")
    print(f"    anchor_alpha={BEAM_ANCHOR_ALPHA}, chunk_alpha={BEAM_CHUNK_ALPHA}")
    print(f"    related_chunk_number={RELATED_CHUNK_NUMBER}")
    print(f"  Default top_k (hybrid/mix): {DEFAULT_TOP_K}")

    all_results = []
    for mode in args.modes:
        result = debug_mode(
            mode=mode,
            query=args.query,
            ground_truth=args.ground_truth,
            save_raw=args.save_raw,
            output_dir=args.output_dir,
        )
        if result:
            all_results.append(result)

    # In bảng so sánh tổng hợp
    if len(all_results) > 1:
        print_comparison_summary(all_results)

    # Kết luận tự động
    print("\n  🔎 Kết luận tự động:")
    for r in all_results:
        mode = r["mode"]
        issues = []

        if not r["has_chunks"]:
            issues.append("❌ KHÔNG có 'Document Chunks' section → extract_chunks_from_context() sẽ trả raw gốc")
        else:
            issues.append("✅ Có 'Document Chunks' section")

        if r["parsed_len"] == 0:
            issues.append("❌ parsed_ctx TRỐNG sau extract — cần kiểm tra lại format")
        elif r["parsed_len"] < 200:
            issues.append(f"⚠️  parsed_ctx rất ngắn ({r['parsed_len']} chars)")

        if r.get("gt_overlap_parsed") is not None:
            ov = r["gt_overlap_parsed"]
            if ov < 0.1:
                issues.append(f"❌ GT overlap thấp ({ov:.1%}) → context thiếu thông tin cần thiết")
            elif ov < 0.3:
                issues.append(f"⚠️  GT overlap trung bình ({ov:.1%})")
            else:
                issues.append(f"✅ GT overlap tốt ({ov:.1%})")

        print(f"\n    [{mode.upper()}]")
        for issue in issues:
            print(f"      {issue}")


if __name__ == "__main__":
    main()
