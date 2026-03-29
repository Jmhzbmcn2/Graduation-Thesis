"""Find which JSON file is corrupted."""
import json
import os

dir_path = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\medical_rag_ollama"

for fname in sorted(os.listdir(dir_path)):
    if fname.endswith(".json"):
        fpath = os.path.join(dir_path, fname)
        size = os.path.getsize(fpath)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                json.load(f)
            print(f"  OK  {fname} ({size:,} bytes)")
        except json.JSONDecodeError as e:
            print(f"  CORRUPTED  {fname} ({size:,} bytes)")
            print(f"             Error: {e}")
            # Show content around the error
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            start = max(0, e.pos - 100)
            end = min(len(content), e.pos + 100)
            print(f"             Around char {e.pos}:")
            print(f"             ...{repr(content[start:e.pos])}<<<HERE>>>{repr(content[e.pos:end])}...")
