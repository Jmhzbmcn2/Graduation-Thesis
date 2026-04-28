# -*- coding: utf-8 -*-
"""
Semantic Markdown Chunking — LightRAG-native module
====================================================

Module này chứa custom chunking function tương thích hoàn toàn với
interface `chunking_func` của LightRAG. Có thể inject vào LightRAG
instance hoặc server mà không sửa bất kỳ core file nào.

Kích hoạt trong server: đặt `CHUNKING_MODE=semantic` trong .env
Fallback về mặc định: đặt `CHUNKING_MODE=token` (hoặc bỏ trống)
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from lightrag.utils import Tokenizer, logger

# ---------------------------------------------------------------------------
# Configuration (đọc từ env, override được từ code)
# ---------------------------------------------------------------------------

# Guard-rail ceiling: 1950 tokens
# - embeddinggemma:300m max = 2048 tokens
# - Context header [Chủ đề | Mục | Nguồn] chiếm ~50-80 tokens
# - Còn lại ~1950 tokens cho nội dung thực sự
SEMANTIC_CHUNK_MAX_TOKENS: int = int(os.getenv("SEMANTIC_CHUNK_MAX_TOKENS", "1950"))

# Overlap = 0 vì:
#   - Các H2 section là đơn vị ngữ nghĩa độc lập → không cần overlap giữa section
#   - Guard-rail chỉ cắt khi 1 section quá dài → sub-chunks cũng độc lập
#   - Overlap chỉ có ý nghĩa với sliding-window (token-based) chunking
SEMANTIC_CHUNK_OVERLAP_TOKENS: int = int(
    os.getenv("SEMANTIC_CHUNK_OVERLAP_TOKENS", "0")
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_source_url(text: str) -> tuple[str, str]:
    """Strip the `# SOURCE_URL:` header line and return (url, clean_text)."""
    # Normalize line endings FIRST — \r\n or bare \r → \n
    # Without this, MarkdownHeaderTextSplitter won't recognize '## heading\r' as a heading
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    source_url = ""
    start_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# SOURCE_URL:"):
            source_url = stripped.replace("# SOURCE_URL:", "").strip()
            start_idx = i + 1
            break
        if stripped:
            break

    return source_url, "\n".join(lines[start_idx:]).strip()


def _extract_article_title(text: str, fallback: str = "") -> str:
    """Return the first H1 heading; fallback to `fallback` string."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()
            if title:
                return title
    return fallback or "Không có tiêu đề"


def _split_by_h2(text: str) -> list[dict]:
    """
    Split text on ## headings using langchain MarkdownHeaderTextSplitter.
    Falls back to a single chunk if no ## exists.
    """
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "H2")],
            strip_headers=False,
        )
        docs = splitter.split_text(text)
        chunks = []
        for doc in docs:
            header = doc.metadata.get("H2", "")
            content = doc.page_content.strip()
            if content:
                chunks.append({"header": header, "content": content})
        if chunks:
            return chunks
    except ImportError:
        logger.warning(
            "langchain_text_splitters not installed. "
            "Falling back to single-chunk mode for semantic chunking."
        )

    # Fallback: no split
    return [{"header": "", "content": text.strip()}]


def _inject_context(
    chunks: list[dict], title: str, source_url: str
) -> list[dict]:
    """Prepend [Chủ đề: ... | Mục: ... | Nguồn: ...] to each chunk."""
    result = []
    for chunk in chunks:
        header = chunk["header"]
        parts = [f"Chủ đề: {title}"]
        if header:
            parts.append(f"Mục: {header}")
        if source_url:
            parts.append(f"Nguồn: {source_url}")
        context_line = "[" + " | ".join(parts) + "]"
        result.append({
            **chunk,
            "content": f"{context_line}\n\n{chunk['content']}",
            "context_line": context_line,
        })
    return result


def _apply_guard_rail(
    chunks: list[dict],
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[dict]:
    """
    Guard-rail: chỉ kích hoạt khi 1 H2 section vượt quá max_tokens.
    Chunks nằm trong giới hạn → pass through nguyên vẹn (KHÔNG thêm overlap).
    Overlap chỉ áp dụng khi cắt thêm bên trong một section quá dài.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        # If langchain not available, do a simple token-window fallback
        logger.warning(
            "langchain_text_splitters not installed. "
            "Guard-rail will use simple token window fallback."
        )
        return _simple_token_window_fallback(chunks, tokenizer, max_tokens, overlap_tokens)

    token_len = lambda text: len(tokenizer.encode(text))  # noqa: E731

    splitter = RecursiveCharacterTextSplitter(
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
            result.append(chunk)
        else:
            context_line = chunk.get("context_line", "")
            logger.info(
                f"[SemanticChunker] Guard-rail split: {n_tokens}tok > {max_tokens} "
                f"(header: {chunk.get('header', '')[:40]!r})"
            )
            # Split only the BODY so every sub-chunk can be re-prefixed with
            # the context line. Without this, sub-chunk 2+ lose the header.
            if context_line and content.startswith(context_line):
                body = content[len(context_line):].strip()
            else:
                body = content

            sub_texts = splitter.split_text(body)
            for i, sub in enumerate(sub_texts):
                sub_content = f"{context_line}\n\n{sub.strip()}" if context_line else sub.strip()
                result.append({
                    "header": (chunk.get("header") or "") + f" (part {i+1})",
                    "content": sub_content,
                    "context_line": context_line,
                })
    return result


def _simple_token_window_fallback(
    chunks: list[dict],
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict]:
    """Token-window fallback when langchain is unavailable."""
    result = []
    for chunk in chunks:
        context_line = chunk.get("context_line", "")
        content = chunk["content"]
        # Split only body so context header can be re-injected per sub-chunk
        if context_line and content.startswith(context_line):
            body = content[len(context_line):].strip()
        else:
            body = content
        body_tokens = tokenizer.encode(body)
        if len(tokenizer.encode(content)) <= max_tokens:
            result.append(chunk)
            continue
        step = max_tokens - overlap_tokens
        for i, start in enumerate(range(0, len(body_tokens), step)):
            sub_body = tokenizer.decode(body_tokens[start: start + max_tokens])
            sub_content = f"{context_line}\n\n{sub_body.strip()}" if context_line else sub_body.strip()
            result.append({
                "header": (chunk.get("header") or "") + f" (part {i+1})",
                "content": sub_content,
                "context_line": context_line,
            })
    return result


# ---------------------------------------------------------------------------
# Public: the chunking_func compatible with LightRAG's interface
# ---------------------------------------------------------------------------

def semantic_markdown_chunker(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = SEMANTIC_CHUNK_OVERLAP_TOKENS,
    chunk_token_size: int = SEMANTIC_CHUNK_MAX_TOKENS,
) -> list[dict[str, Any]]:
    """
    Semantic Markdown Chunking Function — drop-in replacement for
    `chunking_by_token_size`.

    Pipeline:
      1. Strip SOURCE_URL metadata
      2. Extract article title (H1)
      3. Split by ## sections (H2)
      4. Inject [Chủ đề | Mục | Nguồn] context header
      5. Guard-rail: split oversized chunks with RecursiveCharacterTextSplitter

    Compatible signature with LightRAG's `chunking_func` field.
    """
    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Phase 1 — Strip SOURCE_URL
    source_url, clean_text = _extract_source_url(content)

    # Phase 2 — Extract title
    title = _extract_article_title(clean_text)

    # Phase 3 — Split by H2
    raw_chunks = _split_by_h2(clean_text)

    # Phase 4 — Context injection
    enriched = _inject_context(raw_chunks, title, source_url)

    # Phase 5 — Guard-rail
    safe_chunks = _apply_guard_rail(
        enriched, tokenizer, chunk_token_size, chunk_overlap_token_size
    )

    # Build output format expected by LightRAG
    results = []
    for idx, chunk in enumerate(safe_chunks):
        chunk_content = chunk["content"].strip()
        if not chunk_content:
            continue

        # Filter out chunks that are ONLY the context header with no body text.
        # This happens when a ## section exists in the source but has no content body
        # (e.g. the heading is immediately followed by another heading).
        # Such chunks carry no useful information for retrieval or entity extraction.
        context_line = chunk.get("context_line", "")
        body = chunk_content[len(context_line):].strip() if context_line else chunk_content
        if not body:
            logger.debug(
                f"[SemanticChunker] Skipping header-only chunk: {context_line[:80]!r}"
            )
            continue

        n_tokens = len(tokenizer.encode(chunk_content))
        results.append({
            "tokens": n_tokens,
            "content": chunk_content,
            "chunk_order_index": idx,
        })

    if not results:
        # Absolute fallback: return content as-is
        tokens = tokenizer.encode(content)
        return [{"tokens": len(tokens), "content": content.strip(), "chunk_order_index": 0}]

    logger.debug(
        f"[SemanticChunker] '{title[:40]}': "
        f"{len(raw_chunks)} H2-sections → {len(results)} final chunks"
    )
    return results
