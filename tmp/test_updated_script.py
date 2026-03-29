"""Quick verify of the updated script."""
import sys
sys.path.insert(0, ".")
from finetune.generate_training_data import (
    split_train_eval_files, read_all_text_files,
    chunk_all_files, build_eval_dataset, DATA_DIR
)
from lightrag.utils import TiktokenTokenizer

files = read_all_text_files(DATA_DIR)
train_f, eval_f = split_train_eval_files(files)
t = TiktokenTokenizer()
tc = chunk_all_files(train_f, t)
ec = chunk_all_files(eval_f, t)
all_c = tc + ec

print(f"Files: {len(files)} total, {len(train_f)} train, {len(eval_f)} eval")
print(f"Chunks: {len(tc)} train, {len(ec)} eval, {len(all_c)} corpus")
print(f"Sample chunk_id: {tc[0]['chunk_id']}")
print(f"Sample eval chunk_id: {ec[0]['chunk_id']}")

# Quick test build_eval_dataset
fake_eval_pairs = [
    {"query_id": "q1", "query": "test?", "positive": ec[0]["content"], "chunk_id": ec[0]["chunk_id"]}
]
queries, corpus, relevant = build_eval_dataset(fake_eval_pairs, all_c)
print(f"Corpus size: {len(corpus)}")
print(f"Ground truth check: q1 -> {relevant.get('q1')}")
print("\nAll OK!")
