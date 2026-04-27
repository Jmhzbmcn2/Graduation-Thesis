"""
IVF-inspired Entity Anchor Index for LightRAG Local Search.

This module implements an Inverted File Index (IVF) layer that accelerates
entity anchor discovery during local search by narrowing the search space
from O(N×d) to O(C×d + M×d), where M << N.

It does NOT replace NanoVectorDB. Instead, it sits as a supplementary layer
that reads entity vectors from the existing entities_vdb and builds a
KMeans-based clustering index for fast approximate nearest-neighbor lookup.

Reference: FAISS IndexIVFFlat design philosophy.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from lightrag.utils import logger


@dataclass
class IVFEntityIndex:
    """IVF-inspired index for fast entity anchor retrieval.

    Attributes:
        dimension: Embedding vector dimension (e.g., 1536 for OpenAI).
        n_clusters: Number of KMeans centroids (C). Auto-computed if 0.
        n_probe: Default number of clusters to probe per query.
        max_n_probe: Upper bound for adaptive n_probe expansion.
        cosine_threshold: Minimum cosine similarity to include a result.
        min_vectors_for_index: Minimum N required to build IVF index;
            below this threshold, falls back to exact search.
        random_state: Seed for KMeans reproducibility.
        batch_size: MiniBatchKMeans batch size.
        working_dir: Directory to persist index files.
    """

    dimension: int = 0
    n_clusters: int = 0
    n_probe: int = 5
    max_n_probe: int = 20
    cosine_threshold: float = 0.2
    min_vectors_for_index: int = 1000
    random_state: int = 42
    batch_size: int = 4096
    working_dir: str = ""

    # ── Internal state (populated by build_index / load) ──
    centroids: np.ndarray | None = field(default=None, repr=False)
    vectors: np.ndarray | None = field(default=None, repr=False)
    assignments: np.ndarray | None = field(default=None, repr=False)

    # Bucket cache: cluster_id -> array of vector indices (int)
    cluster_to_indices: dict[int, np.ndarray] = field(
        default_factory=dict, repr=False
    )

    # ID list preserves insertion order; metadata stores per-vector info
    ids: list[str] = field(default_factory=list, repr=False)
    id_to_index: dict[str, int] = field(default_factory=dict, repr=False)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    # Bookkeeping
    _is_indexed: bool = field(default=False, repr=False)
    _entity_count_at_build: int = field(default=0, repr=False)
    _built_at: str = field(default="", repr=False)

    # ──────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────

    async def build_index_from_storage(
        self,
        entities_vdb,
    ) -> None:
        """Read all entity vectors from NanoVectorDB and build the IVF index.

        Args:
            entities_vdb: A NanoVectorDBStorage instance (entities namespace).
        """
        logger.info("[IVF] Starting index build from entities_vdb …")

        # ── Step 1: Extract raw data from NanoVectorDB internal storage ──
        storage = await entities_vdb.client_storage
        raw_data: list[dict] = storage.get("data", [])

        if not raw_data:
            logger.warning("[IVF] entities_vdb is empty – nothing to index.")
            return

        vec_ids: list[str] = []
        vec_list: list[np.ndarray] = []
        meta_map: dict[str, dict[str, Any]] = {}

        import base64
        import zlib
        
        for item in raw_data:
            vid = item.get("__id__")
            raw_vec = item.get("vector")
            if vid is None or raw_vec is None:
                continue

            try:
                decoded = base64.b64decode(raw_vec)
                decompressed = zlib.decompress(decoded)
                vector_f16 = np.frombuffer(decompressed, dtype=np.float16)
                vec = vector_f16.astype(np.float32)
            except Exception as e:
                logger.error(f"[IVF] Failed to decode vector for {vid}: {e}")
                continue

            vec_ids.append(vid)
            vec_list.append(vec)

            # Preserve metadata that operate.py needs
            meta_map[vid] = {
                "entity_name": item.get("entity_name", ""),
                "created_at": item.get("__created_at__"),
            }

        if not vec_list:
            logger.warning("[IVF] No valid vectors found in entities_vdb.")
            return

        vectors_matrix = np.stack(vec_list, axis=0).astype(np.float32)
        self.dimension = vectors_matrix.shape[1]

        logger.info(
            f"[IVF] Extracted {len(vec_ids)} entity vectors "
            f"(dim={self.dimension}) from NanoVectorDB."
        )

        self._build_index(vec_ids, vectors_matrix, meta_map)

    def _build_index(
        self,
        vec_ids: list[str],
        vectors_matrix: np.ndarray,
        meta_map: dict[str, dict[str, Any]],
    ) -> None:
        """Core index construction: L2-normalize, KMeans, bucket cache."""
        n = vectors_matrix.shape[0]

        # ── L2 Normalize ──
        norms = np.linalg.norm(vectors_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        vectors_normed = vectors_matrix / norms

        # ── Auto-compute n_clusters if not set ──
        if self.n_clusters <= 0:
            self.n_clusters = min(max(32, int(np.sqrt(n))), 256)
            logger.info(f"[IVF] Auto n_clusters = {self.n_clusters} (N={n})")

        # ── Fallback guard: exact search if too few vectors ──
        if n < self.min_vectors_for_index or n < self.n_clusters:
            logger.info(
                f"[IVF] N={n} < min_vectors_for_index={self.min_vectors_for_index} "
                f"or < n_clusters={self.n_clusters}. Will use exact fallback."
            )
            self.vectors = vectors_normed
            self.ids = vec_ids
            self.id_to_index = {vid: i for i, vid in enumerate(vec_ids)}
            self.metadata = meta_map
            self._is_indexed = False
            self._entity_count_at_build = n
            self._built_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            return

        # ── Run MiniBatchKMeans ──
        logger.info(
            f"[IVF] Running MiniBatchKMeans "
            f"(C={self.n_clusters}, batch={self.batch_size}, seed={self.random_state}) …"
        )
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=self.batch_size,
            n_init=3,
        )
        assignments = kmeans.fit_predict(vectors_normed)

        # L2-normalize centroids
        raw_centroids = kmeans.cluster_centers_.astype(np.float32)
        c_norms = np.linalg.norm(raw_centroids, axis=1, keepdims=True)
        c_norms = np.where(c_norms == 0, 1.0, c_norms)
        centroids_normed = raw_centroids / c_norms

        # ── Populate state ──
        self.vectors = vectors_normed
        self.centroids = centroids_normed
        self.assignments = assignments
        self.ids = vec_ids
        self.id_to_index = {vid: i for i, vid in enumerate(vec_ids)}
        self.metadata = meta_map
        self._is_indexed = True
        self._entity_count_at_build = n
        self._built_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        # ── Build bucket cache for O(1) candidate retrieval ──
        self._rebuild_bucket_cache()

        # Log cluster distribution
        bucket_sizes = [len(v) for v in self.cluster_to_indices.values()]
        if bucket_sizes:
            logger.info(
                f"[IVF] Index built: {n} vectors → {self.n_clusters} clusters. "
                f"Bucket sizes — min={min(bucket_sizes)}, "
                f"avg={np.mean(bucket_sizes):.1f}, max={max(bucket_sizes)}"
            )

    def _rebuild_bucket_cache(self) -> None:
        """Build cluster_to_indices from assignments array."""
        self.cluster_to_indices = {}
        if self.assignments is None:
            return
        for cid in range(self.n_clusters):
            indices = np.where(self.assignments == cid)[0]
            self.cluster_to_indices[cid] = indices

    # ──────────────────────────────────────────────
    #  Query
    # ──────────────────────────────────────────────

    async def query_anchors(
        self,
        query: str | None = None,
        query_embedding: list[float] | np.ndarray | None = None,
        embedding_func=None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Find the top-k entity anchors using IVF approximate search.

        Args:
            query: Raw query string (used if query_embedding is None).
            query_embedding: Pre-computed query embedding.
            embedding_func: Callable to embed query string.
            top_k: Number of results to return.

        Returns:
            List of dicts compatible with ``_get_node_data`` output:
            ``[{"id": ..., "entity_name": ..., "distance": ..., "created_at": ...}]``
        """
        if self.vectors is None or len(self.ids) == 0:
            logger.warning("[IVF] Index is empty – returning no results.")
            return []

        # ── Resolve query vector ──
        if query_embedding is not None:
            qvec = np.asarray(query_embedding, dtype=np.float32)
        elif query is not None and embedding_func is not None:
            emb = await embedding_func([query])
            qvec = np.asarray(emb[0], dtype=np.float32)
        else:
            logger.error("[IVF] No query vector or embedding_func provided.")
            return []

        # L2-normalize query
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qvec = qvec / qnorm

        # ── Branch: Exact search (fallback) vs IVF search ──
        if not self._is_indexed:
            return self._exact_search(qvec, top_k)

        return self._ivf_search(qvec, top_k)

    def _exact_search(
        self, qvec: np.ndarray, top_k: int
    ) -> list[dict[str, Any]]:
        """Fallback: brute-force dot-product on all vectors."""
        scores = self.vectors @ qvec  # [N]
        self.last_search_stats = {
            "probed_clusters": 0,
            "candidates": len(self.ids),
            "total": len(self.ids),
            "n_clusters": 0,
        }
        return self._collect_results(scores, top_k)

    def _ivf_search(
        self, qvec: np.ndarray, top_k: int
    ) -> list[dict[str, Any]]:
        """IVF approximate search with adaptive n_probe."""
        # Step 1: Score centroids
        centroid_scores = self.centroids @ qvec  # [C]
        sorted_cluster_ids = np.argsort(centroid_scores)[::-1]

        # Step 2: Adaptive probe – start with n_probe, expand if needed
        effective_probe = min(self.n_probe, self.n_clusters)
        max_probe = min(self.max_n_probe, self.n_clusters)

        while effective_probe <= max_probe:
            selected_clusters = sorted_cluster_ids[:effective_probe]

            # Step 3: Gather candidate indices from bucket cache
            candidate_indices = np.concatenate(
                [
                    self.cluster_to_indices[cid]
                    for cid in selected_clusters
                    if cid in self.cluster_to_indices
                    and len(self.cluster_to_indices[cid]) > 0
                ]
            )

            if len(candidate_indices) >= top_k * 3 or effective_probe >= max_probe:
                break

            effective_probe = min(effective_probe * 2, max_probe)

        n_total = len(self.ids)
        logger.info(
            f"[IVF] Search: probed {effective_probe}/{self.n_clusters} clusters, "
            f"{len(candidate_indices)} candidates / {n_total} total "
            f"(reduction={1 - len(candidate_indices) / max(n_total, 1):.1%})"
        )

        # Step 4: Compute scores only for candidates
        candidate_vectors = self.vectors[candidate_indices]  # [M, d]
        scores_candidates = candidate_vectors @ qvec  # [M]

        # Build full-size scores array (only candidates get real scores)
        scores = np.full(n_total, -1.0, dtype=np.float32)
        scores[candidate_indices] = scores_candidates

        n_total = len(self.ids)
        self.last_search_stats = {
            "probed_clusters": effective_probe,
            "candidates": len(candidate_indices),
            "total": n_total,
            "n_clusters": self.n_clusters,
        }
        return self._collect_results(scores, top_k)

    def _collect_results(
        self, scores: np.ndarray, top_k: int
    ) -> list[dict[str, Any]]:
        """Filter by threshold, sort, format as LightRAG-compatible dicts."""
        # Apply cosine threshold
        valid_mask = scores >= self.cosine_threshold
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return []

        valid_scores = scores[valid_indices]

        # Get top-k
        if len(valid_indices) <= top_k:
            sorted_local = np.argsort(valid_scores)[::-1]
        else:
            # Partial sort for efficiency
            partitioned = np.argpartition(valid_scores, -top_k)[-top_k:]
            sorted_local = partitioned[np.argsort(valid_scores[partitioned])[::-1]]

        results = []
        for local_idx in sorted_local:
            global_idx = valid_indices[local_idx]
            vid = self.ids[global_idx]
            meta = self.metadata.get(vid, {})
            results.append(
                {
                    "id": vid,
                    "entity_name": meta.get("entity_name", ""),
                    "distance": float(valid_scores[local_idx]),
                    "created_at": meta.get("created_at"),
                }
            )

        return results

    # ──────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────

    def save(self) -> None:
        """Persist index to disk using atomic write."""
        if self.vectors is None:
            logger.warning("[IVF] Nothing to save – index is empty.")
            return

        os.makedirs(self.working_dir, exist_ok=True)

        npz_path = os.path.join(self.working_dir, "ivf_entities.npz")
        meta_path = os.path.join(self.working_dir, "ivf_entities_meta.json")

        # ── Atomic write: NPZ ──
        # np.savez_compressed auto-appends '.npz', so we use a tempfile approach
        npz_tmp_base = npz_path + ".tmp_save"
        arrays_to_save = {"vectors": self.vectors}
        if self.centroids is not None:
            arrays_to_save["centroids"] = self.centroids
        if self.assignments is not None:
            arrays_to_save["assignments"] = self.assignments

        np.savez_compressed(npz_tmp_base, **arrays_to_save)
        # np.savez_compressed creates file at npz_tmp_base + ".npz"
        npz_tmp_actual = npz_tmp_base + ".npz"
        if os.path.exists(npz_tmp_actual):
            os.replace(npz_tmp_actual, npz_path)
        elif os.path.exists(npz_tmp_base):
            # In case numpy didn't append .npz (shouldn't happen, but be safe)
            os.replace(npz_tmp_base, npz_path)

        # ── Atomic write: JSON metadata ──
        meta = {
            "ids": self.ids,
            "metadata": self.metadata,
            "n_clusters": self.n_clusters,
            "n_probe": self.n_probe,
            "max_n_probe": self.max_n_probe,
            "cosine_threshold": self.cosine_threshold,
            "dimension": self.dimension,
            "entity_count": len(self.ids),
            "is_indexed": self._is_indexed,
            "built_at": self._built_at,
            "random_state": self.random_state,
            "batch_size": self.batch_size,
        }
        meta_tmp = meta_path + ".tmp"
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(meta_tmp, meta_path)

        logger.info(
            f"[IVF] Index saved: {len(self.ids)} vectors, "
            f"indexed={self._is_indexed} → {self.working_dir}"
        )

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        npz_path = os.path.join(self.working_dir, "ivf_entities.npz")
        meta_path = os.path.join(self.working_dir, "ivf_entities_meta.json")

        if not os.path.exists(npz_path) or not os.path.exists(meta_path):
            logger.info("[IVF] No saved index found on disk.")
            return False

        try:
            # Load arrays
            data = np.load(npz_path, allow_pickle=False)
            self.vectors = data["vectors"]
            self.centroids = data.get("centroids") if "centroids" in data else None
            self.assignments = data.get("assignments") if "assignments" in data else None

            # Load metadata
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.ids = meta["ids"]
            self.metadata = meta["metadata"]
            self.n_clusters = meta.get("n_clusters", self.n_clusters)
            self.n_probe = meta.get("n_probe", self.n_probe)
            self.max_n_probe = meta.get("max_n_probe", self.max_n_probe)
            self.cosine_threshold = meta.get("cosine_threshold", self.cosine_threshold)
            self.dimension = meta.get("dimension", 0)
            self._is_indexed = meta.get("is_indexed", False)
            self._built_at = meta.get("built_at", "")
            self._entity_count_at_build = meta.get("entity_count", 0)

            # Rebuild derived structures
            self.id_to_index = {vid: i for i, vid in enumerate(self.ids)}
            if self._is_indexed and self.assignments is not None:
                self._rebuild_bucket_cache()

            logger.info(
                f"[IVF] Index loaded: {len(self.ids)} vectors, "
                f"indexed={self._is_indexed}, built_at={self._built_at}"
            )
            return True

        except Exception as e:
            logger.error(f"[IVF] Failed to load index: {e}")
            return False

    def check_stale(self, current_entity_count: int) -> None:
        """Log a warning if the index might be stale."""
        if self._entity_count_at_build == 0:
            return
        if current_entity_count != self._entity_count_at_build:
            logger.warning(
                f"[IVF] Index may be stale: "
                f"built with {self._entity_count_at_build} entities, "
                f"current count is {current_entity_count}. "
                f"Consider calling `rebuild` to update the index."
            )

    @property
    def is_ready(self) -> bool:
        """Whether the index has data loaded (exact or IVF)."""
        return self.vectors is not None and len(self.ids) > 0

    @property
    def is_ivf_active(self) -> bool:
        """Whether IVF clustering is active (vs exact fallback)."""
        return self._is_indexed and self.centroids is not None
