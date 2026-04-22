"""
Batch Debug Context Retrieval — 30 câu từ 500_cases_part2.csv
=============================================================
Đọc 30 câu hỏi, query LightRAG (hybrid, mix, beam), so sánh:
  - Overlap từng word với `answer` (ground truth câu trả lời)
  - Overlap với `context` gốc từ bài báo (text chunk chuẩn)
  - Phát hiện câu nào context retrieval fail hoàn toàn

Output:
  - Console: bảng tóm tắt per-câu + aggregated
  - File: batch_debug_context_result.csv

Cách chạy:
  python testcase/batch_debug_context.py
  python testcase/batch_debug_context.py --n 10
  python testcase/batch_debug_context.py --modes hybrid beam --n 30
  python testcase/batch_debug_context.py --offset 50 --n 30   # câu 51-80
"""
import os
import re
import sys
import json
import time
import argparse
import requests
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — Đồng bộ với eval_ragas_hybrid_mix_beam.py
# ──────────────────────────────────────────────────────────────────────────────
LIGHTRAG_URL = "http://localhost:9621"

BEAM_BEAM_WIDTH        = 10
BEAM_MAX_DEPTH         = 1
BEAM_CHUNK_TOP_K       = 15
BEAM_PRUNING_THRESHOLD = 0.25
BEAM_ANCHOR_ALPHA      = 0.7
BEAM_CHUNK_ALPHA       = 0.7
RELATED_CHUNK_NUMBER   = 5
DEFAULT_TOP_K          = 10

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(_SCRIPT_DIR, "500_cases_part2.csv")
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "batch_debug_context_result.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def extract_chunks_from_context(raw_context: str) -> str:
    chunk_start = raw_context.find("Document Chunks")
    if chunk_start == -1:
        return raw_context

    ref_start = raw_context.find("Reference Document List", chunk_start)
    chunk_section = (
        raw_context[chunk_start:ref_start]
        if ref_start != -1 else raw_context[chunk_start:]
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
        end = rel_start if rel_start != -1 else chunk_start
        entity_sec = raw_context[entity_start:end]
        jm = re.search(r"```json\s*\n(.*?)```", entity_sec, re.DOTALL)
        if jm:
            for line in jm.group(1).strip().split("\n"):
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
        rel_sec = raw_context[rel_start:chunk_start]
        jm = re.search(r"```json\s*\n(.*?)```", rel_sec, re.DOTALL)
        if jm:
            for line in jm.group(1).strip().split("\n"):
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        e1   = obj.get("entity1", "")
                        e2   = obj.get("entity2", "")
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
    return "\n\n".join(result_parts) if result_parts else chunk_section.strip()


def word_overlap(text: str, reference: str) -> float:
    """Tỷ lệ từ trong reference xuất hiện trong text (recall-like)."""
    if not reference.strip() or not text.strip():
        return 0.0
    ref_words = set(reference.lower().split())
    txt_words = set(text.lower().split())
    return len(ref_words & txt_words) / len(ref_words)


def count_section(raw: str, marker: str) -> int:
    """Đếm số JSON object trong section bắt đầu bằng marker."""
    start = raw.find(marker)
    if start == -1:
        return 0
    # tìm section tiếp theo
    next_markers = [
        "Knowledge Graph Data (Entity):",
        "Knowledge Graph Data (Relationship):",
        "Document Chunks",
        "Reference Document List",
    ]
    end = len(raw)
    for nm in next_markers:
        pos = raw.find(nm, start + len(marker))
        if pos != -1:
            end = min(end, pos)
    section = raw[start:end]
    jm = re.search(r"```json\s*\n(.*?)```", section, re.DOTALL)
    if not jm:
        return 0
    count = 0
    for line in jm.group(1).strip().split("\n"):
        line = line.strip()
        if line:
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                pass
    return count


def build_payload(query: str, mode: str) -> dict:
    if mode == "beam":
        return {
            "query": query, "mode": mode, "stream": False,
            "top_k": DEFAULT_TOP_K,
            "beam_width": BEAM_BEAM_WIDTH,
            "beam_max_depth": BEAM_MAX_DEPTH,
            "chunk_top_k": BEAM_CHUNK_TOP_K,
            "pruning_threshold": BEAM_PRUNING_THRESHOLD,
            "anchor_alpha": BEAM_ANCHOR_ALPHA,
            "chunk_alpha": BEAM_CHUNK_ALPHA,
            "related_chunk_number": RELATED_CHUNK_NUMBER,
            "enable_rerank": False,
            "include_context": True,
        }
    return {
        "query": query, "mode": mode, "stream": False,
        "top_k": DEFAULT_TOP_K,
        "enable_rerank": False,
        "include_context": True,
    }


def query_one(query: str, mode: str, timeout: int = 120) -> dict:
    payload = build_payload(query, mode)
    try:
        resp = requests.post(f"{LIGHTRAG_URL}/query", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=30, help="Số câu cần debug")
    parser.add_argument("--offset", type=int, default=0,  help="Bắt đầu từ dòng thứ N")
    parser.add_argument("--modes",  nargs="+", default=["hybrid", "mix", "beam"],
                        choices=["hybrid", "mix", "beam"])
    parser.add_argument("--delay",  type=float, default=0.5,
                        help="Thời gian nghỉ giữa các request (giây)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print(f"  🔬 Batch Debug Context — {args.n} câu từ 500_cases_part2.csv")
    print(f"  Offset: {args.offset} | Modes: {args.modes}")
    print("=" * 80)

    # Load data
    df_all = pd.read_csv(INPUT_FILE)
    df = df_all.iloc[args.offset: args.offset + args.n].reset_index(drop=True)
    print(f"  Đọc {len(df)} câu (dòng {args.offset+1}–{args.offset+len(df)})\n")

    records = []

    col_w = {"mode": 8, "n_chunks": 9, "n_ent": 6, "n_rel": 6,
              "ov_ans": 10, "ov_ctx": 10}
    header = (f"{'#':>3}  {'Câu hỏi':<40}  "
              + "  ".join(
                  f"[{m}] chunks / ov_ans / ov_ctx"
                  for m in args.modes
              ))
    print(header)
    print("─" * 120)

    for idx, row in df.iterrows():
        q          = str(row.get("question", "")).strip()
        ans_gt     = str(row.get("answer",   "")).strip()
        ctx_gt     = str(row.get("context",  "")).strip()
        q_idx      = str(row.get("question_idx", idx))

        rec = {"idx": q_idx, "question": q[:80]}

        mode_parts = []

        for mode in args.modes:
            resp = query_one(q, mode)
            if "error" in resp:
                print(f"    ❌ {mode}: {resp['error']}")
                rec.update({
                    f"{mode}_n_chunks": -1,
                    f"{mode}_n_entities": -1,
                    f"{mode}_n_relations": -1,
                    f"{mode}_parsed_len": 0,
                    f"{mode}_ov_ans": 0.0,
                    f"{mode}_ov_ctx": 0.0,
                    f"{mode}_has_chunks": False,
                })
                mode_parts.append(f"  [{mode}] ERROR")
                continue

            raw = resp.get("context", "")
            parsed = extract_chunks_from_context(raw) if raw else ""

            has_chunks  = "Document Chunks" in raw
            n_chunks    = count_section(raw, "Document Chunks")
            n_entities  = count_section(raw, "Knowledge Graph Data (Entity):")
            n_relations = count_section(raw, "Knowledge Graph Data (Relationship):")

            ov_ans = word_overlap(parsed, ans_gt)
            ov_ctx = word_overlap(parsed, ctx_gt)

            rec.update({
                f"{mode}_n_chunks":    n_chunks,
                f"{mode}_n_entities":  n_entities,
                f"{mode}_n_relations": n_relations,
                f"{mode}_parsed_len":  len(parsed),
                f"{mode}_ov_ans":      round(ov_ans, 4),
                f"{mode}_ov_ctx":      round(ov_ctx, 4),
                f"{mode}_has_chunks":  has_chunks,
            })

            flag = "⚠️ " if (ov_ans < 0.15 or not has_chunks) else "✅ "
            mode_parts.append(
                f"  {flag}[{mode}] chunks={n_chunks:2d} "
                f"ov_ans={ov_ans:.0%} ov_ctx={ov_ctx:.0%}"
            )

            time.sleep(args.delay)

        records.append(rec)

        # Per-row print
        short_q = q[:42].ljust(42)
        print(f"{idx+1:>3}. {short_q}  {'  '.join(mode_parts)}")

    # ── Aggregated summary ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  📊 TỔNG HỢP")
    print("=" * 80)

    result_df = pd.DataFrame(records)

    print(f"\n  {'Metric':<35}", end="")
    for mode in args.modes:
        print(f"  {mode.upper():>10}", end="")
    print()
    print(f"  {'─'*65}")

    metrics = [
        ("Avg chunks retrieved",     "n_chunks",    "{:.1f}"),
        ("Avg entities retrieved",   "n_entities",  "{:.1f}"),
        ("Avg relations retrieved",  "n_relations", "{:.1f}"),
        ("Avg parsed len (chars)",   "parsed_len",  "{:.0f}"),
        ("Avg overlap vs answer",    "ov_ans",      "{:.2%}"),
        ("Avg overlap vs src_ctx",   "ov_ctx",      "{:.2%}"),
        ("% câu có chunks",          "has_chunks",  "{:.0%}"),
        ("% câu ov_ans < 15%",       None,          None),   # computed separately
        ("% câu ov_ans = 0%",        None,          None),
    ]

    for label, key, fmt in metrics:
        if key is None:
            continue
        print(f"  {label:<35}", end="")
        for mode in args.modes:
            col = f"{mode}_{key}"
            if col not in result_df.columns:
                print(f"  {'N/A':>10}", end="")
                continue
            vals = result_df[col]
            if key == "has_chunks":
                val = vals.mean()
                print(f"  {val:>10.0%}", end="")
            else:
                val = vals.mean()
                print(f"  {fmt.format(val):>10}", end="")
        print()

    # Low recall rows
    print(f"  {'% câu ov_ans < 15%':<35}", end="")
    for mode in args.modes:
        col = f"{mode}_ov_ans"
        if col in result_df.columns:
            pct = (result_df[col] < 0.15).mean()
            print(f"  {pct:>10.0%}", end="")
        else:
            print(f"  {'N/A':>10}", end="")
    print()

    print(f"  {'% câu ov_ans = 0%':<35}", end="")
    for mode in args.modes:
        col = f"{mode}_ov_ans"
        if col in result_df.columns:
            pct = (result_df[col] == 0.0).mean()
            print(f"  {pct:>10.0%}", end="")
        else:
            print(f"  {'N/A':>10}", end="")
    print()

    # ── Chi tiết câu overlap thấp ───────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  🔴 Câu có overlap thấp (ov_ans < 20% ít nhất 1 mode):")
    print("─" * 80)
    fail_rows = []
    for _, r in result_df.iterrows():
        low = any(
            r.get(f"{m}_ov_ans", 1.0) < 0.20
            for m in args.modes
        )
        if low:
            fail_rows.append(r)

    if fail_rows:
        for r in fail_rows[:15]:  # in tối đa 15 câu
            q_short = str(r["question"])[:60]
            parts = []
            for m in args.modes:
                ov = r.get(f"{m}_ov_ans", "?")
                parts.append(f"{m}={ov:.0%}" if isinstance(ov, float) else f"{m}=?")
            print(f"  • {q_short}")
            print(f"    {' | '.join(parts)}")
    else:
        print("  Không có câu nào overlap thấp — retrieval tốt! ✅")

    # ── Save CSV ────────────────────────────────────────────────────────────
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n  💾 Kết quả chi tiết → {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
