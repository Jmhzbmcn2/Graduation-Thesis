"""Quick test: verify chunking by ## works on a sample file."""
import sys
sys.path.insert(0, ".")
from lightrag.operate import chunking_by_token_size
from lightrag.utils import TiktokenTokenizer
from pathlib import Path

tokenizer = TiktokenTokenizer()
content = Path("data/azanex.txt").read_text(encoding="utf-8")
chunks = chunking_by_token_size(
    tokenizer=tokenizer,
    content=content,
    split_by_character="##",
    split_by_character_only=False,
    chunk_overlap_token_size=100,
    chunk_token_size=800,
)
print(f"Total chunks from azanex.txt: {len(chunks)}")
for i, c in enumerate(chunks):
    preview = c["content"][:80].replace("\n", " ").replace("\r", "")
    skip = "(SKIP - too short)" if len(c["content"].strip()) < 50 else ""
    print(f"  [{i}] tokens={c['tokens']:>4} {skip} | {preview}...")
