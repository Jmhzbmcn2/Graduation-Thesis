"""
BM25 Lexical Search Storage for Beam Search Hybrid Retrieval.

This module provides BM25 (sparse/lexical) search indices for entities and chunks,
used exclusively by the 'beam' query mode to complement dense vector search.

Architecture:
    - Build phase: build_and_save() reads from VDB/KV JSON files, builds BM25Okapi indices,
      and serializes them as .pkl files in the working directory.
    - Query phase: load() deserializes the .pkl files; query_entities() and query_chunks()
      return top-K results with Min-Max normalized scores in [0, 1].

IMPORTANT: BM25 files are stored in the effective working directory (working_dir + workspace
if workspace is non-empty, else working_dir alone). Use get_bm25_storage_from_config() to
resolve the correct path automatically — do NOT pass workspace alone as the search path.

KLTN — Graduation Thesis
"""

from __future__ import annotations

import json
import os
import pickle
import time
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from lightrag.utils import logger

# Module-level cache: effective_dir -> BM25Storage instance
_bm25_cache: dict[str, "BM25Storage"] = {}


def _resolve_effective_dir(global_config: dict, workspace: str) -> str:
    """Resolve the effective directory for BM25 index storage.

    Mirrors the logic in JsonKVStorage.__post_init__:
      - If workspace is non-empty: working_dir / workspace
      - If workspace is empty: working_dir (and normalize workspace to "")
    """
    working_dir = global_config.get("working_dir", "./rag_storage")
    if workspace and str(workspace).strip():
        return os.path.join(working_dir, str(workspace).strip())
    return working_dir


def get_bm25_storage_from_config(
    global_config: dict, workspace: str
) -> "BM25Storage | None":
    """Get or lazily load a cached BM25Storage instance using global_config + workspace.

    This is the preferred entry point. Use this instead of get_bm25_storage()
    when you have access to the LightRAG global_config dict.

    Returns None if the BM25 index files do not exist (build_bm25_index.py not yet run).
    """
    effective_dir = _resolve_effective_dir(global_config, workspace)
    if effective_dir in _bm25_cache:
        return _bm25_cache[effective_dir]

    storage = BM25Storage(effective_dir)
    if storage.load():
        _bm25_cache[effective_dir] = storage
        return storage

    return None


def get_bm25_storage(working_dir: str) -> "BM25Storage | None":
    """Get or lazily load a cached BM25Storage instance for the given working directory.

    Note: Prefer get_bm25_storage_from_config() when you have access to global_config,
    as it handles the workspace sub-directory correctly.

    Returns None if the BM25 index files do not exist (build_bm25_index.py not yet run).
    """
    if working_dir in _bm25_cache:
        return _bm25_cache[working_dir]

    storage = BM25Storage(working_dir)
    if storage.load():
        _bm25_cache[working_dir] = storage
        return storage

    return None


class BM25Storage:
    """BM25 Lexical Search index for Beam Search hybrid retrieval.

    Manages two separate BM25Okapi indices:
      1. Entity index: entity_name -> tokenized(name + description)
      2. Chunk index:  chunk_id    -> tokenized(content)

    Query methods return results with scores already normalized to [0, 1]
    via Min-Max normalization.  When all BM25 scores are 0 (purely semantic
    query), an empty list is returned so the caller can fallback to vector-only.
    """

    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.entities_index: BM25Okapi | None = None
        self.chunks_index: BM25Okapi | None = None
        self.entity_ids: list[str] = []  # Aligned with BM25 entity corpus
        self.chunk_ids: list[str] = []   # Aligned with BM25 chunk corpus
        self._loaded = False

    # ─── Tokenizer ────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Whitespace tokenizer with lowercase.

        Simple but effective for Latin/English medical terminology
        (drug names, disease names, etc.).
        """
        return text.lower().split()

    # ─── Load / Save ──────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load pre-built BM25 indices from pickle files.

        Returns True if both indices were loaded successfully.
        """
        entities_pkl = os.path.join(self.working_dir, "bm25_entities.pkl")
        chunks_pkl = os.path.join(self.working_dir, "bm25_chunks.pkl")

        if not os.path.exists(entities_pkl) or not os.path.exists(chunks_pkl):
            logger.warning(
                f"BM25 index files not found in {self.working_dir}. "
                "Run build_bm25_index.py to create them."
            )
            return False

        try:
            with open(entities_pkl, "rb") as f:
                data = pickle.load(f)
                self.entities_index = data["index"]
                self.entity_ids = data["ids"]

            with open(chunks_pkl, "rb") as f:
                data = pickle.load(f)
                self.chunks_index = data["index"]
                self.chunk_ids = data["ids"]

            self._loaded = True
            logger.info(
                f"BM25 indices loaded: {len(self.entity_ids)} entities, "
                f"{len(self.chunk_ids)} chunks"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 indices: {e}")
            return False

    # ─── Query Methods ────────────────────────────────────────────────────

    def query_entities(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Query BM25 entity index.

        Returns list of {"entity_name": str, "bm25_score": float} with
        Min-Max normalized scores in [0, 1].

        Returns empty list if:
          - Index not loaded
          - All BM25 scores are 0 (fallback to vector-only)
        """
        if not self._loaded or self.entities_index is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.entities_index.get_scores(tokenized_query)

        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
        if max_score == 0:
            # All BM25 scores are 0 → fallback to vector-only (avoid div-by-zero)
            return []

        # Get top-k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                normalized_score = float(scores[idx] / max_score)
                results.append({
                    "entity_name": self.entity_ids[idx],
                    "bm25_score": normalized_score,
                })

        return results

    def query_chunks(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Query BM25 chunk index.

        Returns list of {"chunk_id": str, "bm25_score": float} with
        Min-Max normalized scores in [0, 1].

        Returns empty list if all BM25 scores are 0.
        """
        if not self._loaded or self.chunks_index is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.chunks_index.get_scores(tokenized_query)

        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
        if max_score == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                normalized_score = float(scores[idx] / max_score)
                results.append({
                    "chunk_id": self.chunk_ids[idx],
                    "bm25_score": normalized_score,
                })

        return results

    # ─── Build (One-time) ─────────────────────────────────────────────────

    @staticmethod
    def build_and_save(working_dir: str) -> "BM25Storage":
        """Build BM25 indices from working directory data and save as .pkl files.

        Data sources:
          - Entities: vdb_entities.json  (nano_vectordb format with entity_name + content)
          - Chunks:   kv_store_text_chunks.json (KV store with chunk content)

        This should be run once after KG construction. Typical time: < 5 seconds.
        """
        t0 = time.time()
        storage = BM25Storage(working_dir)

        # ─── Build Entity Index ───
        entity_corpus: list[list[str]] = []

        # Try nano_vectordb format (vdb_entities.json)
        # Supports both formats:
        #   - Newer: {"data": [{"__id__": ..., "entity_name": ..., "content": ..., "vector": ...}]}
        #   - Older: {"__data__": [{"__id__": ..., "entity_name": ..., "content": ..., "__vector__": ...}]}
        vdb_entities_path = os.path.join(working_dir, "vdb_entities.json")
        if os.path.exists(vdb_entities_path):
            logger.info(f"Reading entities from {vdb_entities_path}...")
            with open(vdb_entities_path, "r", encoding="utf-8") as f:
                vdb_data = json.load(f)

            # Detect format: 'data' (list) or '__data__' (list)
            items = vdb_data.get("data", []) or vdb_data.get("__data__", [])
            if isinstance(items, dict):
                # Some formats use dict of {id: {fields}}
                items = list(items.values())

            for item in items:
                if not isinstance(item, dict):
                    continue
                entity_name = item.get("entity_name", "")
                content = item.get("content", "")
                if entity_name:
                    # Combine name + description for richer BM25 matching
                    text = f"{entity_name} {content}".strip()
                    storage.entity_ids.append(entity_name)
                    entity_corpus.append(BM25Storage._tokenize(text))

        if entity_corpus:
            storage.entities_index = BM25Okapi(entity_corpus)
            logger.info(f"Built BM25 entity index: {len(entity_corpus)} entities")
        else:
            logger.warning("No entity data found for BM25 index")

        # ─── Build Chunk Index ───
        chunk_corpus: list[list[str]] = []

        chunks_path = os.path.join(working_dir, "kv_store_text_chunks.json")
        if os.path.exists(chunks_path):
            logger.info(f"Reading chunks from {chunks_path}...")
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

            for chunk_id, chunk_info in chunks_data.items():
                # Skip internal metadata keys
                if chunk_id.startswith("__"):
                    continue
                content = chunk_info.get("content", "")
                if content:
                    storage.chunk_ids.append(chunk_id)
                    chunk_corpus.append(BM25Storage._tokenize(content))

        if chunk_corpus:
            storage.chunks_index = BM25Okapi(chunk_corpus)
            logger.info(f"Built BM25 chunk index: {len(chunk_corpus)} chunks")
        else:
            logger.warning("No chunk data found for BM25 index")

        # ─── Save to Pickle ───
        entities_pkl = os.path.join(working_dir, "bm25_entities.pkl")
        chunks_pkl = os.path.join(working_dir, "bm25_chunks.pkl")

        with open(entities_pkl, "wb") as f:
            pickle.dump({
                "index": storage.entities_index,
                "ids": storage.entity_ids,
            }, f)

        with open(chunks_pkl, "wb") as f:
            pickle.dump({
                "index": storage.chunks_index,
                "ids": storage.chunk_ids,
            }, f)

        elapsed = time.time() - t0
        logger.info(
            f"BM25 indices saved to {working_dir} in {elapsed:.2f}s "
            f"({len(storage.entity_ids)} entities, {len(storage.chunk_ids)} chunks)"
        )

        storage._loaded = True
        return storage
