# -*- coding: utf-8 -*-
"""
Semantic Markdown Chunking Pipeline for Medical Data
=====================================================

Phases 1-5 of the chunking strategy:
  1. Data Preprocessing  – clean SOURCE_URL, extract article title
  2. Markdown Header Splitting – split by ## (H2) sections
  3. Context Injection – prepend [Chủ đề: ... - Mục: ...] to each chunk
  4. Fallback Recursive Splitting – guard-rail at ~1800 tokens
  5. LightRAG Insert – pass-through chunking into LightRAG

Usage:
  python scripts/custom_insert.py --data-dir /path/to/vietmed_crawled
  python scripts/custom_insert.py --data-dir /path/to/vietmed_crawled --limit 30   # sample run
  python scripts/custom_insert.py --data-dir /path/to/vietmed_crawled --dry-run     # preview only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

# Ensure Vietnamese output is handled correctly on all platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import chardet
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `lightrag` package resolves
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, logger, Tokenizer

# ---------------------------------------------------------------------------
# Load environment from .env (same as the rest of the project)
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"), override=False)


# ============================================================================
#  CONFIGURATION
# ============================================================================

WORKING_DIR = os.getenv("WORKING_DIR", "./medical_rag_v4")

# LLM settings (OpenAI-compatible vLLM)
LLM_HOST = os.getenv("LLM_BINDING_HOST", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_BINDING_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")

# Embedding settings (Ollama)
EMBEDDING_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Chunking guard-rail ceiling (tokens)
# embeddinggemma:300m max = 2048; context header [Chủ đề|Mục|Nguồn] ~50-80 tokens
# → dùng 1950 để còn đủ chỗ cho context header
GUARD_RAIL_TOKEN_SIZE = 1950

# Overlap = 0 giữa các H2 section (section độc lập về ngữ nghĩa)
# Overlap chỉ có ý nghĩa với sliding-window chunking, không áp dụng ở đây
GUARD_RAIL_OVERLAP_TOKENS = 0

# LightRAG ceiling – must be >= GUARD_RAIL_TOKEN_SIZE
LIGHTRAG_CHUNK_TOKEN_SIZE = 2048


# ============================================================================
#  PHASE 1 — DATA PREPROCESSING
# ============================================================================

def read_file_safe(filepath: Path) -> str:
    """Read a file with automatic encoding detection."""
    raw = filepath.read_bytes()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return raw.decode("utf-8", errors="replace")


def extract_source_url(text: str) -> tuple[str, str]:
    """
    Extract the SOURCE_URL line from the top of the document.
    Returns (source_url, cleaned_text).
    """
    lines = text.split("\n")
    source_url = ""
    start_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# SOURCE_URL:"):
            source_url = stripped.replace("# SOURCE_URL:", "").strip()
            start_idx = i + 1
            break
        # Skip empty lines at the top before hitting real content
        if stripped:
            break

    cleaned = "\n".join(lines[start_idx:]).strip()
    return source_url, cleaned


def extract_article_title(text: str, filename: str) -> str:
    """
    Extract the article title from the first H1 heading.
    Falls back to reformatting the filename if no H1 exists.
    """
    for line in text.split("\n"):
        stripped = line.strip()
        # Match H1 but not H2+
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()  # Remove leading "# " exactly
            if title:
                return title
    # Fallback: filename -> title
    name = Path(filename).stem
    # Replace hyphens/underscores with spaces, title-case
    title = re.sub(r"[-_]+", " ", name)
    title = unicodedata.normalize("NFC", title).title()
    return title


def preprocess_file(filepath: Path) -> dict[str, str]:
    """
    Phase 1: Read, detect encoding, extract metadata.
    Returns dict with keys: source_url, title, clean_text, filename
    """
    text = read_file_safe(filepath)
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    source_url, clean_text = extract_source_url(text)
    title = extract_article_title(clean_text, filepath.name)

    return {
        "source_url": source_url,
        "title": title,
        "clean_text": clean_text,
        "filename": filepath.name,
    }


# ============================================================================
#  PHASE 2 — MARKDOWN HEADER SPLITTING
# ============================================================================

# Only split on H2 — keep H3+ content within the parent H2 section
MD_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("##", "Header 2")],
    strip_headers=False,  # Keep H2/H3 headers inside the chunk text
)


def split_by_markdown_headers(text: str) -> list[dict]:
    """
    Phase 2: Split text by ## headers.
    Returns list of dicts with keys: header, content
    """
    docs = MD_SPLITTER.split_text(text)

    chunks = []
    for doc in docs:
        header = doc.metadata.get("Header 2", "")
        content = doc.page_content.strip()
        if content:
            chunks.append({"header": header, "content": content})

    # Fallback: if splitter returned nothing useful (no H2 in file)
    if not chunks and text.strip():
        chunks.append({"header": "", "content": text.strip()})

    return chunks


# ============================================================================
#  PHASE 3 — CONTEXT INJECTION
# ============================================================================

def inject_context(chunks: list[dict], title: str, source_url: str) -> list[dict]:
    """
    Phase 3: Prepend semantic context header to each chunk.
    """
    enriched = []
    for chunk in chunks:
        header = chunk["header"]
        content = chunk["content"]

        # Build context line
        parts = [f"Chủ đề: {title}"]
        if header:
            parts.append(f"Mục: {header}")
        if source_url:
            parts.append(f"Nguồn: {source_url}")

        context_line = "[" + " | ".join(parts) + "]"
        enriched_content = f"{context_line}\n\n{content}"

        enriched.append({
            "header": header,
            "content": enriched_content,
            "context_line": context_line,
        })

    return enriched


# ============================================================================
#  PHASE 4 — FALLBACK RECURSIVE SPLITTING (token-based)
# ============================================================================

def _make_token_length_func():
    """Create a token-counting function using tiktoken (same tokenizer as LightRAG default)."""
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return lambda text: len(enc.encode(text))


def apply_guard_rail(
    chunks: list[dict],
    max_tokens: int = GUARD_RAIL_TOKEN_SIZE,
    overlap_tokens: int = GUARD_RAIL_OVERLAP_TOKENS,
) -> list[dict]:
    """
    Phase 4: Split any chunk that exceeds max_tokens.
    Chunks under the limit pass through untouched.
    """
    token_len = _make_token_length_func()

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        length_function=token_len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    result = []
    for chunk in chunks:
        content = chunk["content"]
        n_tokens = token_len(content)

        if n_tokens <= max_tokens:
            # Pass through — keep section intact
            result.append(chunk)
        else:
            # Split oversized chunk
            logger.info(
                f"Guard-rail split: {n_tokens} tokens > {max_tokens} max "
                f"(header: {chunk.get('header', 'N/A')[:40]})"
            )
            sub_texts = recursive_splitter.split_text(content)
            for i, sub in enumerate(sub_texts):
                result.append({
                    "header": chunk.get("header", "") + f" (part {i+1})",
                    "content": sub,
                    "context_line": chunk.get("context_line", ""),
                })

    return result


# ============================================================================
#  FULL PIPELINE: PHASES 1–4
# ============================================================================

def process_single_file(filepath: Path) -> list[str]:
    """
    Run phases 1–4 on a single file.
    Returns a list of ready-to-insert chunk strings.
    """
    # Phase 1
    meta = preprocess_file(filepath)

    # Phase 2
    raw_chunks = split_by_markdown_headers(meta["clean_text"])

    # Phase 3
    enriched_chunks = inject_context(
        raw_chunks, meta["title"], meta["source_url"]
    )

    # Phase 4
    safe_chunks = apply_guard_rail(enriched_chunks)

    return [c["content"] for c in safe_chunks]


def process_directory(
    data_dir: Path,
    limit: int | None = None,
) -> tuple[list[str], list[str], dict]:
    """
    Process all .txt files in data_dir through phases 1–4.

    Returns:
        all_chunks: flat list of chunk strings
        file_paths: corresponding file name per chunk (for LightRAG citation)
        stats: dict with processing statistics
    """
    txt_files = sorted(data_dir.rglob("*.txt"))
    if limit:
        txt_files = txt_files[:limit]

    all_chunks: list[str] = []
    file_paths: list[str] = []
    stats = {
        "total_files": len(txt_files),
        "total_chunks": 0,
        "files_processed": 0,
        "files_skipped": 0,
        "guard_rail_splits": 0,
        "chunks_per_file": [],
    }

    token_len = _make_token_length_func()

    for filepath in txt_files:
        try:
            chunks = process_single_file(filepath)
            if not chunks:
                stats["files_skipped"] += 1
                continue

            all_chunks.extend(chunks)
            file_paths.extend([filepath.name] * len(chunks))

            stats["files_processed"] += 1
            stats["chunks_per_file"].append(len(chunks))

        except Exception as e:
            logger.error(f"Failed to process {filepath.name}: {e}")
            stats["files_skipped"] += 1

    stats["total_chunks"] = len(all_chunks)

    # Token distribution stats
    if all_chunks:
        token_counts = [token_len(c) for c in all_chunks]
        stats["min_tokens"] = min(token_counts)
        stats["max_tokens"] = max(token_counts)
        stats["avg_tokens"] = sum(token_counts) / len(token_counts)
        stats["median_tokens"] = sorted(token_counts)[len(token_counts) // 2]

    return all_chunks, file_paths, stats


# ============================================================================
#  PHASE 5 — LIGHTRAG INSERT (pass-through chunker)
# ============================================================================

def passthrough_chunker(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 2048,
) -> list[dict[str, Any]]:
    """
    Custom chunking function that passes content through as-is.
    Each 'document' fed to LightRAG is already a pre-processed chunk,
    so we return it as a single chunk without further splitting.
    """
    tokens = tokenizer.encode(content)
    return [
        {
            "tokens": len(tokens),
            "content": content.strip(),
            "chunk_order_index": 0,
        }
    ]


async def ollama_embed_func(texts: list[str]) -> np.ndarray:
    """Embedding function using Ollama (embeddinggemma:300m)."""
    import httpx

    embed_url = f"{EMBEDDING_HOST}/api/embeddings"
    embeddings = []
    for text in texts:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                embed_url,
                json={"model": EMBEDDING_MODEL, "prompt": text},
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            embeddings.append(np.array(embedding, dtype=np.float32))
    return np.array(embeddings, dtype=np.float32)


async def llm_complete(prompt: str, **kwargs) -> str:
    """LLM function using OpenAI-compatible vLLM endpoint."""
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        base_url=LLM_HOST,
        api_key=LLM_API_KEY,
        **kwargs,
    )


def create_rag_instance(working_dir: str | None = None) -> LightRAG:
    """
    Create a LightRAG instance configured for semantic-chunked medical data.
    """
    wdir = working_dir or WORKING_DIR

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=512,
        func=ollama_embed_func,
    )

    rag = LightRAG(
        working_dir=wdir,
        llm_model_func=llm_complete,
        llm_model_name=LLM_MODEL,
        embedding_func=embedding_func,

        # Chunking: pass-through — our chunks are already processed
        chunking_func=passthrough_chunker,
        chunk_token_size=LIGHTRAG_CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=GUARD_RAIL_OVERLAP_TOKENS,

        # Match existing .env settings
        llm_model_max_async=int(os.getenv("MAX_ASYNC", "2")),
        max_parallel_insert=int(os.getenv("MAX_PARALLEL_INSERT", "1")),
        entity_extract_max_gleaning=int(os.getenv("MAX_GLEANING", "2")),

        addon_params={
            "language": os.getenv("SUMMARY_LANGUAGE", "Vietnamese"),
            "entity_types": json.loads(
                os.getenv(
                    "ENTITY_TYPES",
                    '["Disease","Symptom","Drug","Chemical compound","Protein",'
                    '"Anatomy","Biological process","Exposure","Diagnostic test",'
                    '"Treatment method"]',
                )
            ),
        },
    )
    return rag


async def insert_chunks(
    all_chunks: list[str],
    file_paths: list[str],
    working_dir: str | None = None,
    batch_size: int = 50,
) -> None:
    """
    Phase 5: Insert pre-processed chunks into LightRAG.

    Each chunk is inserted as a separate 'document'. The passthrough_chunker
    ensures LightRAG does not re-split them.
    """
    rag = create_rag_instance(working_dir)
    await rag.initialize_storages()

    total = len(all_chunks)
    logger.info(f"Starting insert of {total} chunks into LightRAG ({rag.working_dir})")

    try:
        # Insert in batches to avoid memory pressure
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_chunks = all_chunks[start:end]
            batch_paths = file_paths[start:end]

            logger.info(f"Inserting batch {start+1}–{end} / {total}")

            await rag.ainsert(
                batch_chunks,
                file_paths=batch_paths,
            )

            logger.info(f"Batch {start+1}–{end} done")

    finally:
        await rag.finalize_storages()

    logger.info(f"All {total} chunks inserted successfully")


# ============================================================================
#  CLI ENTRY POINT
# ============================================================================

def print_stats(stats: dict) -> None:
    """Pretty-print processing statistics."""
    print("\n" + "=" * 60)
    print("  SEMANTIC MARKDOWN CHUNKING — STATISTICS")
    print("=" * 60)
    print(f"  Files found:      {stats['total_files']}")
    print(f"  Files processed:  {stats['files_processed']}")
    print(f"  Files skipped:    {stats['files_skipped']}")
    print(f"  Total chunks:     {stats['total_chunks']}")
    if stats["total_chunks"] > 0:
        print(f"  Token range:      {stats['min_tokens']}–{stats['max_tokens']}")
        print(f"  Avg tokens/chunk: {stats['avg_tokens']:.0f}")
        print(f"  Median tokens:    {stats['median_tokens']}")
        avg_chunks = (
            sum(stats["chunks_per_file"]) / len(stats["chunks_per_file"])
            if stats["chunks_per_file"]
            else 0
        )
        print(f"  Avg chunks/file:  {avg_chunks:.1f}")
    print("=" * 60 + "\n")


def print_chunk_preview(chunks: list[str], n: int = 3) -> None:
    """Preview first N chunks."""
    print(f"\n--- Preview of first {min(n, len(chunks))} chunks ---\n")
    for i, chunk in enumerate(chunks[:n]):
        print(f"[Chunk {i+1}] ({len(chunk)} chars)")
        # Show first 300 chars
        preview = chunk[:300] + ("..." if len(chunk) > 300 else "")
        print(preview)
        print("-" * 40)


async def main():
    parser = argparse.ArgumentParser(
        description="Semantic Markdown Chunking Pipeline for Medical Data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing .txt medical files",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=None,
        help=f"LightRAG working directory (default: {WORKING_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of chunks per insert batch (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process files but do NOT insert into LightRAG (preview mode)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    print(f"Data directory: {data_dir}")
    print(f"Working directory: {args.working_dir or WORKING_DIR}")
    if args.limit:
        print(f"File limit: {args.limit}")
    if args.dry_run:
        print("** DRY RUN — will NOT insert into LightRAG **")

    # Phases 1–4
    t0 = time.time()
    all_chunks, file_paths, stats = process_directory(data_dir, limit=args.limit)
    elapsed = time.time() - t0

    print_stats(stats)
    print(f"Preprocessing time: {elapsed:.1f}s")

    if not all_chunks:
        print("No chunks to insert. Exiting.")
        sys.exit(0)

    print_chunk_preview(all_chunks)

    if args.dry_run:
        print("\n** Dry run complete. No data was inserted. **")
        return

    # Phase 5: Insert
    t1 = time.time()
    await insert_chunks(
        all_chunks,
        file_paths,
        working_dir=args.working_dir,
        batch_size=args.batch_size,
    )
    insert_elapsed = time.time() - t1
    print(f"\nInsert time: {insert_elapsed:.1f}s")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
