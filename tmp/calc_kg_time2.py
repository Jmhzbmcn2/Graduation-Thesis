"""Calculate KG build time - fixed timezone handling."""
import json
from datetime import datetime, timezone

with open(r"/home/linhvd/Graduation-Thesis/medical_rag/medical_rag_ollama/kv_store_doc_status.json", "r", encoding="utf-8") as f:
    data = json.load(f)

total_docs = len(data)
processed_docs = 0
failed_docs = 0
pending_docs = 0
total_processing_seconds = 0
total_chunks = 0
total_content_length = 0

all_processing_start = []
all_processing_end = []
per_doc_durations = []

for doc_id, doc in data.items():
    status = doc.get("status", "unknown")
    if status == "processed":
        processed_docs += 1
    elif status == "failed":
        failed_docs += 1
    elif status in ("pending", "processing"):
        pending_docs += 1

    chunks_count = doc.get("chunks_count", 0) or 0
    total_chunks += chunks_count
    total_content_length += doc.get("content_length", 0) or 0

    metadata = doc.get("metadata", {})
    start_time = metadata.get("processing_start_time")
    end_time = metadata.get("processing_end_time")

    if start_time and end_time:
        all_processing_start.append(start_time)
        all_processing_end.append(end_time)
        duration = end_time - start_time
        per_doc_durations.append(duration)
        total_processing_seconds += duration

def fmt(s):
    h, m, sec = int(s//3600), int((s%3600)//60), int(s%60)
    return f"{h}h {m}m {sec}s" if h else (f"{m}m {sec}s" if m else f"{sec}s")

lines = []
lines.append("=" * 60)
lines.append("  KG BUILD TIME ANALYSIS")
lines.append("=" * 60)

remaining = total_docs - processed_docs - failed_docs
avg_chunks_processed = total_chunks / processed_docs if processed_docs else 0

lines.append(f"")
lines.append(f"DOCUMENT STATS:")
lines.append(f"  Total docs:        {total_docs}")
lines.append(f"  Processed:         {processed_docs}")
lines.append(f"  Failed:            {failed_docs}")
lines.append(f"  Remaining:         {remaining}")
lines.append(f"  Progress:          {processed_docs/total_docs*100:.1f}%")
lines.append(f"  Total chunks:      {total_chunks}")
lines.append(f"  Avg chunks/doc:    {avg_chunks_processed:.1f} (processed docs)")
lines.append(f"")

if all_processing_start:
    gs = min(all_processing_start)
    ge = max(all_processing_end)
    gw = ge - gs
    lines.append(f"PROCESSING WINDOW:")
    lines.append(f"  Start:             {datetime.fromtimestamp(gs).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  End:               {datetime.fromtimestamp(ge).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Elapsed:           {fmt(gw)}")
    lines.append(f"")

if per_doc_durations:
    avg = total_processing_seconds / len(per_doc_durations)
    med = sorted(per_doc_durations)[len(per_doc_durations)//2]
    lines.append(f"PER-DOC PROCESSING TIME:")
    lines.append(f"  Cumulative total:  {fmt(total_processing_seconds)}")
    lines.append(f"  Average:           {avg:.0f}s ({fmt(avg)})")
    lines.append(f"  Median:            {med}s ({fmt(med)})")
    lines.append(f"  Min:               {min(per_doc_durations)}s")
    lines.append(f"  Max:               {max(per_doc_durations)}s")
    lines.append(f"")

    # Parallelism
    if gw > 0:
        parallelism = total_processing_seconds / gw
        docs_per_hour = processed_docs / (gw / 3600)
        lines.append(f"THROUGHPUT:")
        lines.append(f"  Parallelism:       {parallelism:.2f}x")
        lines.append(f"  Docs/hour:         {docs_per_hour:.1f}")
        lines.append(f"")

        # Estimate remaining
        est_remaining_hours = remaining / docs_per_hour if docs_per_hour > 0 else 0
        est_remaining_seconds = est_remaining_hours * 3600
        lines.append(f"ESTIMATE REMAINING:")
        lines.append(f"  Remaining docs:    {remaining}")
        lines.append(f"  Est. time left:    {fmt(est_remaining_seconds)}")
        lines.append(f"  Est. total time:   {fmt(gw + est_remaining_seconds)}")

        # RPD estimate
        est_total_calls = remaining * avg_chunks_processed * 3
        lines.append(f"")
        lines.append(f"RPD ESTIMATE:")
        lines.append(f"  Est. LLM calls left: ~{est_total_calls:.0f}")
        lines.append(f"  RPD limit: 10,000/day")
        if est_total_calls > 10000:
            days_needed = est_total_calls / 10000
            lines.append(f"  Days needed:       ~{days_needed:.1f} days")
        else:
            lines.append(f"  Can finish today:  YES")

lines.append(f"")
lines.append("=" * 60)

output = "\n".join(lines)
print(output)

with open(r"/home/linhvd/Graduation-Thesis/tmp/kg_time_report.txt", "w", encoding="utf-8") as f:
    f.write(output)
