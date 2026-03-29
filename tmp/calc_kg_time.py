"""Calculate the total time to build the Knowledge Graph from doc_status timestamps."""
import json
from datetime import datetime, timezone

with open(r"c:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\medical_rag_ollama\kv_store_doc_status.json", "r", encoding="utf-8") as f:
    data = json.load(f)

total_docs = len(data)
processed_docs = 0
failed_docs = 0
total_processing_seconds = 0
total_chunks = 0

all_created_at = []
all_updated_at = []
all_processing_start = []
all_processing_end = []
per_doc_durations = []

for doc_id, doc in data.items():
    status = doc.get("status", "unknown")
    if status == "processed":
        processed_docs += 1
    elif status == "failed":
        failed_docs += 1
    
    chunks_count = doc.get("chunks_count", 0) or 0
    total_chunks += chunks_count
    
    # Parse created_at and updated_at
    created_at_str = doc.get("created_at")
    updated_at_str = doc.get("updated_at")
    
    if created_at_str:
        try:
            ca = datetime.fromisoformat(created_at_str)
            all_created_at.append(ca)
        except:
            pass
    
    if updated_at_str:
        try:
            ua = datetime.fromisoformat(updated_at_str)
            all_updated_at.append(ua)
        except:
            pass
    
    # Parse processing times from metadata
    metadata = doc.get("metadata", {})
    start_time = metadata.get("processing_start_time")
    end_time = metadata.get("processing_end_time")
    
    if start_time and end_time:
        all_processing_start.append(start_time)
        all_processing_end.append(end_time)
        duration = end_time - start_time
        per_doc_durations.append(duration)
        total_processing_seconds += duration

# Overall wall-clock time
if all_created_at and all_updated_at:
    earliest_created = min(all_created_at)
    latest_updated = max(all_updated_at)
    wall_clock = latest_updated - earliest_created
    wall_clock_seconds = wall_clock.total_seconds()
else:
    wall_clock_seconds = 0

# Processing time from metadata (actual processing_start -> processing_end)
if all_processing_start and all_processing_end:
    global_start = min(all_processing_start)
    global_end = max(all_processing_end)
    global_processing_wall = global_end - global_start
else:
    global_start = global_end = global_processing_wall = 0

# Stats
avg_per_doc = total_processing_seconds / len(per_doc_durations) if per_doc_durations else 0
min_per_doc = min(per_doc_durations) if per_doc_durations else 0
max_per_doc = max(per_doc_durations) if per_doc_durations else 0
median_per_doc = sorted(per_doc_durations)[len(per_doc_durations)//2] if per_doc_durations else 0

def fmt_time(seconds):
    """Format seconds into hours, minutes, seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def fmt_timestamp(ts):
    """Format Unix timestamp to readable string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

print("=" * 70)
print("   KNOWLEDGE GRAPH BUILD TIME ANALYSIS")
print("=" * 70)

print(f"\n📊 DOCUMENT STATISTICS:")
print(f"   Total documents:     {total_docs}")
print(f"   Processed:           {processed_docs}")
print(f"   Failed:              {failed_docs}")
print(f"   Total chunks:        {total_chunks}")
print(f"   Avg chunks/doc:      {total_chunks/total_docs:.1f}")

print(f"\n⏱️  WALL-CLOCK TIME (from created_at → updated_at):")
if all_created_at and all_updated_at:
    print(f"   First doc created:   {min(all_created_at).isoformat()}")
    print(f"   Last doc updated:    {max(all_updated_at).isoformat()}")
    print(f"   Wall-clock duration: {fmt_time(wall_clock_seconds)}")

print(f"\n🔧 PROCESSING TIME (from metadata timestamps):")
if all_processing_start:
    print(f"   Global start:        {fmt_timestamp(global_start)}")
    print(f"   Global end:          {fmt_timestamp(global_end)}")
    print(f"   Wall-clock (processing): {fmt_time(global_processing_wall)}")

print(f"\n📈 PER-DOCUMENT PROCESSING DURATION:")
print(f"   Total cumulative:    {fmt_time(total_processing_seconds)}")
print(f"   Average per doc:     {fmt_time(avg_per_doc)} ({avg_per_doc:.1f}s)")
print(f"   Median per doc:      {fmt_time(median_per_doc)} ({median_per_doc}s)")
print(f"   Min per doc:         {fmt_time(min_per_doc)} ({min_per_doc}s)")
print(f"   Max per doc:         {fmt_time(max_per_doc)} ({max_per_doc}s)")

# Distribution
print(f"\n📊 PROCESSING TIME DISTRIBUTION:")
buckets = [(0, 60, "< 1 min"), (60, 120, "1-2 min"), (120, 180, "2-3 min"), 
           (180, 300, "3-5 min"), (300, 600, "5-10 min"), (600, float('inf'), "> 10 min")]
for low, high, label in buckets:
    count = sum(1 for d in per_doc_durations if low <= d < high)
    if count > 0:
        bar = "█" * (count // 2) if count > 1 else "█"
        print(f"   {label:>10}: {count:3d} docs {bar}")

print(f"\n{'=' * 70}")

# Parallelism estimation
if global_processing_wall > 0 and total_processing_seconds > 0:
    parallelism = total_processing_seconds / global_processing_wall
    print(f"\n💡 PARALLELISM ESTIMATE:")
    print(f"   Effective parallelism: {parallelism:.2f}x")
    print(f"   (cumulative processing / wall-clock = parallel workers)")

print()
