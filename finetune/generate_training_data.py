"""
Generate Training Data for Embedding Model Fine-tuning.

Đọc các file medical text từ data/, chunk theo heading (##),
gọi Gemini API sinh câu hỏi tiếng Việt, chia train/eval theo file (80/20),
và lưu thành training data + evaluation data cho MRR/Recall@K.

Usage:
    cd C:\\Users\\VUDUYLINH\\PycharmProjects\\KLTN\\LightRAG
    python finetune/generate_training_data.py

Output files:
    finetune/training_data.json          - Cặp (query, positive) để fine-tune
    finetune/training_data_triplet.json  - Bộ 3 (query, positive, negative)
    finetune/eval_queries.json           - Queries cho evaluation
    finetune/eval_corpus.json            - Toàn bộ corpus (tất cả chunks)
    finetune/eval_relevant_docs.json     - Ground truth: query_id -> [chunk_ids]
    finetune/split_info.json             - Thông tin chia train/eval files
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

# Add project root to path so we can import lightrag modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

from lightrag.operate import chunking_by_token_size
from lightrag.utils import TiktokenTokenizer

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "finetune"

# Chunking config
CHUNK_TOKEN_SIZE = 800
CHUNK_OVERLAP_TOKEN_SIZE = 100
SPLIT_BY_CHARACTER = "##"  # Chia theo heading markdown

# Gemini config (đọc từ .env)
GEMINI_API_KEY = os.getenv("LLM_BINDING_API_KEY", "")
GEMINI_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Rate limiting
MAX_CONCURRENT = 2  # Max concurrent Gemini requests
DELAY_BETWEEN_REQUESTS = 1.0  # Seconds between requests
NUM_QUESTIONS_PER_CHUNK = 3  # Số câu hỏi sinh ra cho mỗi chunk
MIN_CHUNK_LENGTH = 50  # Skip chunks quá ngắn (ký tự)

# Train/Eval split
EVAL_RATIO = 0.2  # 20% files dùng cho evaluation
RANDOM_SEED = 42  # Seed cố định để reproducible

# Hard negatives config
NUM_HARD_NEGATIVES = 1  # Số hard negative cho mỗi cặp trong triplet data

# Giới hạn số file để test (set None để đọc tất cả)
MAX_FILES = 20

# Progress file (để resume khi bị gián đoạn)
PROGRESS_FILE = OUTPUT_DIR / "_progress.jsonl"

# ─── Prompt Template ─────────────────────────────────────────────────────────

QUESTION_GENERATION_PROMPT = """Bạn là một chuyên gia y tế. Dựa vào đoạn văn bản y tế dưới đây, hãy tạo ra chính xác {num_questions} câu hỏi thực tế mà một bệnh nhân hoặc người dùng có thể hỏi bác sĩ hoặc tra cứu trên mạng.

Yêu cầu:
- Câu hỏi phải bằng tiếng Việt, tự nhiên, ngắn gọn
- Câu hỏi phải có thể được trả lời bằng thông tin trong đoạn văn bản
- Đa dạng loại câu hỏi: hỏi về công dụng, cách dùng, tác dụng phụ, liều lượng, chống chỉ định, v.v.
- Không lặp lại câu hỏi
- CHỈ trả về JSON array, không giải thích gì thêm

Đoạn văn bản:
---
{chunk_text}
---

Trả lời dưới dạng JSON array:
["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"]"""


# ─── Core Functions ──────────────────────────────────────────────────────────


def read_all_text_files(data_dir: Path, max_files: int | None = MAX_FILES) -> list[dict[str, str]]:
    """Đọc file .txt từ data directory.

    Args:
        max_files: Giới hạn số file đọc (None = đọc tất cả)

    Returns:
        List of dicts with keys: filename, content
    """
    files = []
    for txt_file in sorted(data_dir.glob("*.txt")):
        if max_files is not None and len(files) >= max_files:
            break
        try:
            content = txt_file.read_text(encoding="utf-8")
            if content.strip():
                files.append({"filename": txt_file.name, "content": content})
        except Exception as e:
            print(f"  ⚠ Lỗi đọc file {txt_file.name}: {e}")
    return files


def split_train_eval_files(
    files: list[dict[str, str]],
    eval_ratio: float = EVAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    """Chia files thành train và eval sets.

    Chia theo file (không theo query) để đảm bảo eval data
    hoàn toàn tách biệt khỏi train data → đánh giá chính xác hơn.

    Returns:
        (train_files, eval_files)
    """
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)

    num_eval = max(1, int(len(shuffled) * eval_ratio))
    eval_files = shuffled[:num_eval]
    train_files = shuffled[num_eval:]

    return train_files, eval_files


def chunk_all_files(
    files: list[dict[str, str]], tokenizer: TiktokenTokenizer
) -> list[dict]:
    """Chunk tất cả file theo heading (##).

    Mỗi chunk được gán chunk_id duy nhất: "{filename}_chunk_{index}"

    Returns:
        List of dicts with keys: content, source_file, chunk_order_index, tokens, chunk_id
    """
    all_chunks = []
    for file_info in files:
        chunks = chunking_by_token_size(
            tokenizer=tokenizer,
            content=file_info["content"],
            split_by_character=SPLIT_BY_CHARACTER,
            split_by_character_only=False,
            chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
            chunk_token_size=CHUNK_TOKEN_SIZE,
        )
        valid_idx = 0
        for chunk in chunks:
            # Skip chunks quá ngắn (header-only hoặc empty)
            if len(chunk["content"].strip()) < MIN_CHUNK_LENGTH:
                continue
            chunk["source_file"] = file_info["filename"]
            # Tạo chunk_id duy nhất cho evaluation
            base_name = file_info["filename"].replace(".txt", "")
            chunk["chunk_id"] = f"{base_name}_chunk_{valid_idx}"
            valid_idx += 1
            all_chunks.append(chunk)
    return all_chunks


async def generate_questions_for_chunk(
    client,
    chunk_text: str,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """Gọi Gemini API để sinh câu hỏi cho 1 chunk.

    Returns:
        List of generated questions
    """
    prompt = QUESTION_GENERATION_PROMPT.format(
        num_questions=NUM_QUESTIONS_PER_CHUNK,
        chunk_text=chunk_text,
    )

    async with semaphore:
        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt],
            )

            # Extract text from response
            if not response.candidates:
                return []

            text = ""
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            text += part.text

            if not text.strip():
                return []

            # Parse JSON array from response
            text = text.strip()
            start_idx = text.find("[")
            end_idx = text.rfind("]")
            if start_idx == -1 or end_idx == -1:
                print(f"    ⚠ Không tìm thấy JSON array trong response")
                return []

            json_text = text[start_idx : end_idx + 1]
            questions = json.loads(json_text)

            if isinstance(questions, list):
                return [q.strip() for q in questions if isinstance(q, str) and q.strip()]

            return []

        except json.JSONDecodeError as e:
            print(f"    ⚠ Lỗi parse JSON: {e}")
            return []
        except Exception as e:
            print(f"    ⚠ Lỗi Gemini API: {e}")
            await asyncio.sleep(3)
            return []


def load_progress() -> dict[str, list[dict]]:
    """Load kết quả đã xử lý từ progress file.

    Returns:
        Dict mapping chunk_id -> list of training pairs
    """
    progress: dict[str, list[dict]] = {}
    if not PROGRESS_FILE.exists():
        return progress

    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                chunk_id = entry.get("chunk_id", "")
                if chunk_id:
                    if chunk_id not in progress:
                        progress[chunk_id] = []
                    progress[chunk_id].append(entry)
            except json.JSONDecodeError:
                continue

    return progress


def save_progress_batch(pairs: list[dict]) -> None:
    """Append một batch kết quả vào progress file (JSONL)."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


async def process_all_chunks(
    client,
    chunks: list[dict],
    label: str = "",
) -> list[dict]:
    """Xử lý tất cả chunks, sinh câu hỏi song song (có rate limit).
    Tự động skip chunks đã xử lý (resume support).

    Returns:
        List of training pairs: {"query", "positive", "source_file", "chunk_id"}
    """
    # Load progress để skip chunks đã xử lý
    existing_progress = load_progress()
    training_pairs = []
    skipped = 0

    # Thu thập kết quả đã có từ progress
    chunks_to_process = []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid in existing_progress:
            training_pairs.extend(existing_progress[cid])
            skipped += 1
        else:
            chunks_to_process.append(chunk)

    if skipped > 0:
        print(f"  ⏩ {label}Skip {skipped} chunks đã xử lý, còn {len(chunks_to_process)} chunks mới")

    if not chunks_to_process:
        print(f"  ✓ {label}Tất cả chunks đã được xử lý trước đó!")
        return training_pairs

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    total = len(chunks_to_process)
    completed = 0
    failed = 0
    query_counter = len(training_pairs)  # Tiếp tục đánh số từ progress

    # Process in batches to respect rate limits
    batch_size = MAX_CONCURRENT * 2
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = chunks_to_process[batch_start:batch_end]

        tasks = []
        for chunk in batch:
            task = generate_questions_for_chunk(
                client=client,
                chunk_text=chunk["content"],
                semaphore=semaphore,
            )
            tasks.append((chunk, task))

        # Run batch concurrently
        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        batch_new_pairs = []
        for (chunk, _), result in zip(tasks, results):
            completed += 1
            if isinstance(result, Exception):
                failed += 1
                print(f"  {label}[{completed}/{total}] ✗ {chunk['source_file']} - Error: {result}")
                continue

            questions = result
            if not questions:
                failed += 1
                print(f"  {label}[{completed}/{total}] ✗ {chunk['source_file']} - Không có câu hỏi")
                continue

            for q in questions:
                query_counter += 1
                pair = {
                    "query_id": f"{label}q_{query_counter}",
                    "query": q,
                    "positive": chunk["content"],
                    "chunk_id": chunk["chunk_id"],
                    "source_file": chunk["source_file"],
                }
                training_pairs.append(pair)
                batch_new_pairs.append(pair)

            print(
                f"  {label}[{completed}/{total}] ✓ {chunk['source_file']}"
                f" {chunk['chunk_id']} → {len(questions)} câu hỏi"
            )

        # Lưu progress sau mỗi batch
        if batch_new_pairs:
            save_progress_batch(batch_new_pairs)

        # Rate limiting delay between batches
        if batch_end < total:
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\n  {label}Hoàn tất: {completed - failed}/{total} chunks mới, {skipped} đã có, {failed} thất bại")
    return training_pairs


def create_triplet_data(
    training_pairs: list[dict],
    all_chunks: list[dict],
    num_negatives: int = NUM_HARD_NEGATIVES,
) -> list[dict]:
    """Tạo training data dạng triplet (query, positive, negative).

    Hard negative: chọn ngẫu nhiên chunk từ FILE KHÁC.
    """
    chunks_by_file: dict[str, list[str]] = {}
    for chunk in all_chunks:
        fname = chunk["source_file"]
        if fname not in chunks_by_file:
            chunks_by_file[fname] = []
        chunks_by_file[fname].append(chunk["content"])

    all_file_names = list(chunks_by_file.keys())
    triplets = []

    for pair in training_pairs:
        source = pair["source_file"]
        other_files = [f for f in all_file_names if f != source]
        if not other_files:
            continue

        for _ in range(num_negatives):
            neg_file = random.choice(other_files)
            neg_chunk = random.choice(chunks_by_file[neg_file])
            triplets.append(
                {
                    "query": pair["query"],
                    "positive": pair["positive"],
                    "negative": neg_chunk,
                    "source_file": pair["source_file"],
                }
            )

    return triplets


def build_eval_dataset(
    eval_pairs: list[dict],
    all_chunks: list[dict],
) -> tuple[dict, dict, dict]:
    """Xây dựng evaluation dataset cho MRR / Recall@K.

    Args:
        eval_pairs: Các cặp query-positive từ eval files
        all_chunks: TOÀN BỘ chunks (train + eval) làm corpus

    Returns:
        (queries, corpus, relevant_docs)
        - queries: {query_id: query_text}
        - corpus: {chunk_id: chunk_text}  (toàn bộ kho)
        - relevant_docs: {query_id: [chunk_id]}  (ground truth)
    """
    # Corpus = toàn bộ chunks (cả train và eval)
    corpus = {chunk["chunk_id"]: chunk["content"] for chunk in all_chunks}

    # Queries và ground truth từ eval pairs
    queries = {}
    relevant_docs = {}

    for pair in eval_pairs:
        qid = pair["query_id"]
        queries[qid] = pair["query"]

        if qid not in relevant_docs:
            relevant_docs[qid] = []
        relevant_docs[qid].append(pair["chunk_id"])

    return queries, corpus, relevant_docs


def save_all_results(
    train_pairs: list[dict],
    triplets: list[dict],
    eval_queries: dict,
    eval_corpus: dict,
    eval_relevant_docs: dict,
    split_info: dict,
) -> None:
    """Lưu tất cả kết quả ra file JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _save(data, filename, label):
        path = OUTPUT_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        count = len(data) if isinstance(data, (list, dict)) else "N/A"
        print(f"  ✓ {label}: {count} items → {path.name}")

    _save(train_pairs, "training_data.json", "Training pairs")
    _save(triplets, "training_data_triplet.json", "Training triplets")
    _save(eval_queries, "eval_queries.json", "Eval queries")
    _save(eval_corpus, "eval_corpus.json", "Eval corpus (all chunks)")
    _save(eval_relevant_docs, "eval_relevant_docs.json", "Eval ground truth")
    _save(split_info, "split_info.json", "Split info")


# ─── Main ─────────────────────────────────────────────────────────────────────


async def main():
    print("=" * 60)
    print("🧬 MEDICAL EMBEDDING TRAINING DATA GENERATOR")
    print("=" * 60)

    # Validate
    if not GEMINI_API_KEY:
        print("✗ LLM_BINDING_API_KEY not found in .env")
        sys.exit(1)

    print(f"\n📁 Data directory: {DATA_DIR}")
    print(f"🤖 Gemini model: {GEMINI_MODEL}")
    print(f"📐 Chunk: split_by='##', token_size={CHUNK_TOKEN_SIZE}, overlap={CHUNK_OVERLAP_TOKEN_SIZE}")
    print(f"❓ Questions per chunk: {NUM_QUESTIONS_PER_CHUNK}")
    print(f"⚡ Concurrency: {MAX_CONCURRENT}")
    print(f"📊 Train/Eval split: {int((1-EVAL_RATIO)*100)}% / {int(EVAL_RATIO*100)}% (by file)")

    # ── Step 1: Đọc files ──
    print(f"\n{'─' * 40}")
    print("📖 Step 1: Đọc data files...")
    all_files = read_all_text_files(DATA_DIR)
    print(f"  ✓ Đọc được {len(all_files)} files")

    if not all_files:
        print("✗ Không tìm thấy file .txt nào trong data/")
        sys.exit(1)

    # ── Step 2: Chia train/eval theo file ──
    print(f"\n{'─' * 40}")
    print("🔀 Step 2: Chia train/eval files...")
    train_files, eval_files = split_train_eval_files(all_files)
    print(f"  ✓ Train: {len(train_files)} files")
    print(f"  ✓ Eval:  {len(eval_files)} files")

    split_info = {
        "seed": RANDOM_SEED,
        "eval_ratio": EVAL_RATIO,
        "train_files": [f["filename"] for f in train_files],
        "eval_files": [f["filename"] for f in eval_files],
    }

    # ── Step 3: Chunk ──
    print(f"\n{'─' * 40}")
    print("✂️  Step 3: Chunking theo heading (##)...")
    tokenizer = TiktokenTokenizer()

    train_chunks = chunk_all_files(train_files, tokenizer)
    eval_chunks = chunk_all_files(eval_files, tokenizer)
    all_chunks = train_chunks + eval_chunks

    print(f"  ✓ Train chunks: {len(train_chunks)}")
    print(f"  ✓ Eval chunks:  {len(eval_chunks)}")
    print(f"  ✓ Total corpus: {len(all_chunks)}")

    # ── Step 4: Generate questions via Gemini ──
    print(f"\n{'─' * 40}")
    print("🤖 Step 4: Gọi Gemini sinh câu hỏi...")
    start_time = time.time()

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Generate cho train
    print(f"\n  ── TRAIN ({len(train_chunks)} chunks) ──")
    train_pairs = await process_all_chunks(client, train_chunks, label="TRAIN ")

    # Generate cho eval
    print(f"\n  ── EVAL ({len(eval_chunks)} chunks) ──")
    eval_pairs = await process_all_chunks(client, eval_chunks, label="EVAL ")

    elapsed = time.time() - start_time
    print(f"\n  ⏱️  Thời gian: {elapsed:.1f}s")

    if not train_pairs:
        print("✗ Không tạo được cặp training nào!")
        sys.exit(1)

    # ── Step 5: Create triplet data (chỉ cho train) ──
    print(f"\n{'─' * 40}")
    print("🔀 Step 5: Tạo triplet data (with hard negatives)...")
    triplets = create_triplet_data(train_pairs, train_chunks)
    print(f"  ✓ {len(triplets)} triplets")

    # ── Step 6: Build evaluation dataset ──
    print(f"\n{'─' * 40}")
    print("📊 Step 6: Xây dựng evaluation dataset...")
    eval_queries, eval_corpus, eval_relevant_docs = build_eval_dataset(
        eval_pairs, all_chunks  # Corpus = toàn bộ chunks
    )
    print(f"  ✓ Eval queries:       {len(eval_queries)}")
    print(f"  ✓ Corpus size:        {len(eval_corpus)} chunks")
    print(f"  ✓ Ground truth pairs: {sum(len(v) for v in eval_relevant_docs.values())}")

    # ── Step 7: Save ──
    print(f"\n{'─' * 40}")
    print("💾 Step 7: Lưu kết quả...")
    save_all_results(
        train_pairs=train_pairs,
        triplets=triplets,
        eval_queries=eval_queries,
        eval_corpus=eval_corpus,
        eval_relevant_docs=eval_relevant_docs,
        split_info=split_info,
    )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("📊 TỔNG KẾT")
    print(f"{'=' * 60}")
    print(f"  📁 Total files:            {len(all_files)}")
    print(f"     ├─ Train files:         {len(train_files)}")
    print(f"     └─ Eval files:          {len(eval_files)}")
    print(f"  ✂️  Total chunks (corpus):  {len(all_chunks)}")
    print(f"     ├─ Train chunks:        {len(train_chunks)}")
    print(f"     └─ Eval chunks:         {len(eval_chunks)}")
    print(f"  ❓ Train query-positive:   {len(train_pairs)}")
    print(f"  🔀 Train triplets:         {len(triplets)}")
    print(f"  📊 Eval queries:           {len(eval_queries)}")
    print(f"  ⏱️  Tổng thời gian:        {elapsed:.1f}s")
    print(f"\n  📄 Output files (trong {OUTPUT_DIR.name}/):")
    print(f"     training_data.json          ← fine-tune embedding")
    print(f"     training_data_triplet.json  ← fine-tune với hard negatives")
    print(f"     eval_queries.json           ← queries để đánh giá")
    print(f"     eval_corpus.json            ← corpus cho retrieval")
    print(f"     eval_relevant_docs.json     ← ground truth (MRR/Recall@K)")
    print(f"     split_info.json             ← danh sách files train/eval")
    print(f"\n✅ Hoàn tất! Dùng các file này để fine-tune và đánh giá embedding model.")


if __name__ == "__main__":
    asyncio.run(main())
