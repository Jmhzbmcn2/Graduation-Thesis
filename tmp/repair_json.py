"""Repair corrupted JSON file by truncating at corruption point and closing properly."""
import json
import os
import shutil

filepath = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\medical_rag_ollama\kv_store_relation_chunks.json"
backup_path = filepath + ".bak"

# Backup first
shutil.copy2(filepath, backup_path)
print(f"Backup created: {backup_path}")

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

print(f"File size: {len(content):,} chars")

# Try to find the last valid JSON by progressively truncating
# Strategy: find the last complete key-value pair before corruption point
# The file is a JSON object like { "key1": {...}, "key2": {...}, ... }

# First, try to parse to find exact error position
try:
    json.loads(content)
    print("File is actually valid JSON!")
    exit(0)
except json.JSONDecodeError as e:
    print(f"Corruption at: line {e.lineno}, col {e.colno}, char {e.pos}")
    error_pos = e.pos

# Strategy: find the last '},' or '}' before the corruption point
# that would properly close a value in the top-level object
search_start = max(0, error_pos - 5000)
truncated = content[:error_pos]

# Find the last complete entry - look for pattern like:  },\n  "  or  }\n}
best_pos = -1
# Search backwards for a valid cut point
for i in range(error_pos - 1, max(0, error_pos - 50000), -1):
    if content[i] == '}':
        # Try to parse with this as the end
        candidate = content[:i+1] + "\n}"
        try:
            parsed = json.loads(candidate)
            best_pos = i + 1
            print(f"Found valid cut point at char {best_pos}")
            print(f"Recovered {len(parsed)} entries")
            
            # Write repaired file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
            
            repaired_size = os.path.getsize(filepath)
            original_size = os.path.getsize(backup_path)
            lost_pct = (1 - repaired_size/original_size) * 100
            
            print(f"\nRepair successful!")
            print(f"  Original size:  {original_size:,} bytes")
            print(f"  Repaired size:  {repaired_size:,} bytes")
            print(f"  Data lost:      ~{lost_pct:.1f}%")
            print(f"  Entries saved:  {len(parsed)}")
            break
        except json.JSONDecodeError:
            continue

if best_pos == -1:
    print("Could not repair file automatically!")
    print("Restoring backup...")
    shutil.copy2(backup_path, filepath)
