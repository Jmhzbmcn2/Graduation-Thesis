"""
Fine-tune nomic-embed-text-v1.5 cho Medical Vietnamese Data.

Notebook này dùng trên Kaggle (GPU T4 miễn phí).
Upload các file JSON từ finetune/ lên Kaggle Dataset trước khi chạy.

Steps:
  1. Install dependencies
  2. Load training data + evaluation data
  3. Fine-tune nomic-embed-text-v1.5 với MultipleNegativesRankingLoss
  4. Evaluate: MRR@10, Recall@1, Recall@5, Recall@10
  5. Save model

Usage on Kaggle:
  - Upload training_data.json, eval_queries.json, eval_corpus.json,
    eval_relevant_docs.json lên Kaggle Dataset
  - Tạo notebook, chọn GPU T4, copy code này vào và chạy
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 1: Install & Setup                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

import os

# Chỉ dùng 1 GPU để tránh lỗi DataParallel với nomic-bert
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# !pip install -U sentence-transformers datasets

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 2: Configuration                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Paths ──
# Thay đổi đường dẫn này theo tên Kaggle Dataset của bạn
KAGGLE_INPUT = "/kaggle/input/medical-embedding-data"  # ← ĐỔI TÊN NÀY
OUTPUT_DIR = "/kaggle/working/finetuned-nomic-medical"

# ── Model ──
BASE_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# ── Training hyperparameters ──
EPOCHS = 3
BATCH_SIZE = 32          # Giảm xuống 16 nếu OOM
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
EVAL_STEPS = 200         # Đánh giá sau mỗi N steps
FP16 = True              # T4 hỗ trợ FP16, tiết kiệm VRAM

# ── Nomic prefix (BẮT BUỘC cho nomic-embed-text) ──
QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 3: Load Data                                               ║
# ╚══════════════════════════════════════════════════════════════════╝

def load_json(filename):
    path = f"{KAGGLE_INPUT}/{filename}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Training data
train_data = load_json("training_data.json")
print(f"✓ Training pairs: {len(train_data)}")

# Evaluation data
eval_queries = load_json("eval_queries.json")
eval_corpus = load_json("eval_corpus.json")
eval_relevant_docs = load_json("eval_relevant_docs.json")
print(f"✓ Eval queries: {len(eval_queries)}")
print(f"✓ Eval corpus:  {len(eval_corpus)} chunks")
print(f"✓ Ground truth:  {len(eval_relevant_docs)} query-doc mappings")

# Preview
print(f"\n── Sample training pair ──")
sample = train_data[0]
print(f"  Query:    {sample['query'][:80]}...")
print(f"  Positive: {sample['positive'][:80]}...")
print(f"  File:     {sample['source_file']}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 4: Prepare Dataset                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

from datasets import Dataset

# Thêm prefix bắt buộc của nomic-embed-text
# search_query: cho câu hỏi, search_document: cho tài liệu
train_dataset = Dataset.from_dict({
    "anchor": [QUERY_PREFIX + item["query"] for item in train_data],
    "positive": [DOCUMENT_PREFIX + item["positive"] for item in train_data],
})

print(f"✓ Train dataset: {len(train_dataset)} examples")
print(f"  Columns: {train_dataset.column_names}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 5: Load Model & Setup Training                             ║
# ╚══════════════════════════════════════════════════════════════════╝

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Load base model
print(f"Loading {BASE_MODEL}...")
model = SentenceTransformer(BASE_MODEL, trust_remote_code=True)
print(f"✓ Model loaded: {model.get_sentence_embedding_dimension()} dimensions")

# Loss function
# MNRL: in-batch negatives tự động, rất hiệu quả với data dạng pairs
loss = MultipleNegativesRankingLoss(model)

# Evaluator - thêm prefix cho eval data
# Queries cần prefix "search_query: ", corpus cần prefix "search_document: "
prefixed_queries = {qid: QUERY_PREFIX + q for qid, q in eval_queries.items()}
prefixed_corpus = {cid: DOCUMENT_PREFIX + c for cid, c in eval_corpus.items()}
eval_relevant_docs_set = {qid: set(cids) for qid, cids in eval_relevant_docs.items()}

evaluator = InformationRetrievalEvaluator(
    queries=prefixed_queries,
    corpus=prefixed_corpus,
    relevant_docs=eval_relevant_docs_set,
    name="medical-vi",
    show_progress_bar=True,
)

# ── Baseline: Đánh giá model GỐC trước khi fine-tune ──
print("\n📊 Đánh giá model GỐC (chưa fine-tune)...")
baseline_results = evaluator(model)

print("\n── Baseline Results ──")
for key, value in sorted(baseline_results.items()):
    if any(m in key for m in ["mrr", "recall", "ndcg"]):
        print(f"  {key}: {value:.4f}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 6: Train!                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    save_total_limit=2,
    fp16=FP16,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="medical-vi_cosine_mrr@10",
    report_to="none",  # Tắt wandb trên Kaggle
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=evaluator,
)

print(f"\n🚀 Bắt đầu fine-tune {BASE_MODEL}")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Training examples: {len(train_dataset)}")
print(f"   Steps per epoch: {len(train_dataset) // BATCH_SIZE}")
print()

trainer.train()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 7: Final Evaluation & Save                                ║
# ╚══════════════════════════════════════════════════════════════════╝

# Đánh giá model SAU fine-tune
print("\n📊 Đánh giá model SAU fine-tune...")
final_results = evaluator(model)

print("\n" + "=" * 60)
print("📊 SO SÁNH KẾT QUẢ: TRƯỚC vs SAU FINE-TUNE")
print("=" * 60)
print(f"{'Metric':<35} {'Baseline':>10} {'Fine-tuned':>10} {'Δ':>10}")
print("-" * 65)

for key in sorted(baseline_results.keys()):
    if any(m in key for m in ["mrr", "recall", "ndcg"]):
        base_val = baseline_results[key]
        fine_val = final_results[key]
        delta = fine_val - base_val
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  {key:<33} {base_val:>9.4f} {fine_val:>10.4f} {arrow}{abs(delta):>8.4f}")

# Save model
model.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Model saved to {OUTPUT_DIR}")

# Save kết quả so sánh
comparison = {
    "base_model": BASE_MODEL,
    "training_examples": len(train_dataset),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "baseline": baseline_results,
    "finetuned": final_results,
    "timestamp": datetime.now().isoformat(),
}
with open(f"{OUTPUT_DIR}/eval_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)
print(f"✓ Comparison saved to {OUTPUT_DIR}/eval_comparison.json")

print(f"\n🎉 Fine-tune hoàn tất!")
print(f"   Download model từ: {OUTPUT_DIR}")
