"""
Unit tests for IVFEntityIndex.

Tests cover:
- Index build with L2-normalization and KMeans clustering
- Exact fallback when dataset is too small
- Query accuracy and result format
- Persistence (save/load)
- Stale index detection
"""

import asyncio
import os
import shutil
import tempfile

import numpy as np
import pytest

from lightrag.kg.ivf_entity_index import IVFEntityIndex


# ── Fixtures ──


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test artifacts."""
    d = tempfile.mkdtemp(prefix="ivf_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_fake_data(n: int, dim: int = 128, seed: int = 42):
    """Generate fake vectors, ids, and metadata."""
    rng = np.random.RandomState(seed)
    vectors = rng.randn(n, dim).astype(np.float32)
    ids = [f"ent-{i:05d}" for i in range(n)]
    metadata = {
        vid: {"entity_name": f"Entity_{i}", "created_at": 1700000000 + i}
        for i, vid in enumerate(ids)
    }
    return ids, vectors, metadata


# ── Tests ──


class TestIVFBuildIndex:
    def test_build_with_clustering(self, tmp_dir):
        """When N >= min_vectors_for_index, IVF should build clusters."""
        ids, vectors, metadata = _make_fake_data(2000, dim=64)

        idx = IVFEntityIndex(
            working_dir=tmp_dir,
            n_clusters=32,
            n_probe=3,
            min_vectors_for_index=500,
            random_state=42,
        )
        idx._build_index(ids, vectors, metadata)

        assert idx.is_ready
        assert idx.is_ivf_active
        assert idx.centroids is not None
        assert idx.centroids.shape == (32, 64)
        assert idx.assignments is not None
        assert len(idx.cluster_to_indices) == 32

        # Verify all vectors are assigned
        total_assigned = sum(len(v) for v in idx.cluster_to_indices.values())
        assert total_assigned == 2000

    def test_fallback_exact_when_small(self, tmp_dir):
        """When N < min_vectors_for_index, should fallback to exact mode."""
        ids, vectors, metadata = _make_fake_data(100, dim=64)

        idx = IVFEntityIndex(
            working_dir=tmp_dir,
            n_clusters=32,
            min_vectors_for_index=500,
        )
        idx._build_index(ids, vectors, metadata)

        assert idx.is_ready
        assert not idx.is_ivf_active
        assert idx.centroids is None

    def test_auto_n_clusters(self, tmp_dir):
        """When n_clusters=0, it should auto-compute sqrt(N)."""
        ids, vectors, metadata = _make_fake_data(10000, dim=32)

        idx = IVFEntityIndex(
            working_dir=tmp_dir,
            n_clusters=0,
            min_vectors_for_index=500,
        )
        idx._build_index(ids, vectors, metadata)

        # sqrt(10000) = 100, capped at 256
        assert idx.n_clusters == 100
        assert idx.is_ivf_active


class TestIVFQuery:
    def _build_index(self, tmp_dir, n=2000, dim=64, n_clusters=32):
        ids, vectors, metadata = _make_fake_data(n, dim)
        idx = IVFEntityIndex(
            working_dir=tmp_dir,
            n_clusters=n_clusters,
            n_probe=5,
            min_vectors_for_index=500,
            cosine_threshold=0.0,  # Accept all for testing
        )
        idx._build_index(ids, vectors, metadata)
        return idx, vectors

    def test_ivf_query_returns_correct_format(self, tmp_dir):
        """Query results must have id, entity_name, distance, created_at."""
        idx, vectors = self._build_index(tmp_dir)

        # Use first vector as query
        results = asyncio.get_event_loop().run_until_complete(
            idx.query_anchors(query_embedding=vectors[0], top_k=5)
        )

        assert len(results) > 0
        assert len(results) <= 5
        for r in results:
            assert "id" in r
            assert "entity_name" in r
            assert "distance" in r
            assert "created_at" in r

    def test_ivf_query_top1_matches_exact(self, tmp_dir):
        """Top-1 result of IVF should match exact search for the same query."""
        idx, vectors = self._build_index(tmp_dir, n=2000, n_clusters=32)

        query_vec = vectors[42]

        # IVF search
        ivf_results = asyncio.get_event_loop().run_until_complete(
            idx.query_anchors(query_embedding=query_vec, top_k=1)
        )

        # Exact search
        exact_results = idx._exact_search(
            query_vec / np.linalg.norm(query_vec), top_k=1
        )

        assert ivf_results[0]["id"] == exact_results[0]["id"]

    def test_exact_fallback_query(self, tmp_dir):
        """When index falls back to exact, query should still work."""
        ids, vectors, metadata = _make_fake_data(100, dim=64)

        idx = IVFEntityIndex(
            working_dir=tmp_dir,
            n_clusters=32,
            min_vectors_for_index=500,
            cosine_threshold=0.0,
        )
        idx._build_index(ids, vectors, metadata)

        assert not idx.is_ivf_active

        results = asyncio.get_event_loop().run_until_complete(
            idx.query_anchors(query_embedding=vectors[0], top_k=3)
        )
        assert len(results) == 3

    def test_cosine_threshold_filters(self, tmp_dir):
        """Results below cosine_threshold should be excluded."""
        idx, vectors = self._build_index(tmp_dir)

        # Set very high threshold
        idx.cosine_threshold = 0.99

        results = asyncio.get_event_loop().run_until_complete(
            idx.query_anchors(query_embedding=vectors[0], top_k=5)
        )

        for r in results:
            assert r["distance"] >= 0.99


class TestIVFPersistence:
    def test_save_and_load(self, tmp_dir):
        """Index should survive save/load cycle."""
        ids, vectors, metadata = _make_fake_data(2000, dim=64)

        idx1 = IVFEntityIndex(
            working_dir=tmp_dir,
            n_clusters=32,
            n_probe=5,
            min_vectors_for_index=500,
        )
        idx1._build_index(ids, vectors, metadata)
        idx1.save()

        # Load into new instance
        idx2 = IVFEntityIndex(working_dir=tmp_dir)
        assert idx2.load()
        assert idx2.is_ready
        assert idx2.is_ivf_active
        assert len(idx2.ids) == 2000
        assert idx2.centroids.shape == (32, 64)
        assert len(idx2.cluster_to_indices) == 32

    def test_query_after_reload(self, tmp_dir):
        """Query results should be identical before and after reload."""
        ids, vectors, metadata = _make_fake_data(2000, dim=64)

        idx1 = IVFEntityIndex(
            working_dir=tmp_dir,
            n_clusters=32,
            n_probe=5,
            min_vectors_for_index=500,
            cosine_threshold=0.0,
        )
        idx1._build_index(ids, vectors, metadata)

        results_before = asyncio.get_event_loop().run_until_complete(
            idx1.query_anchors(query_embedding=vectors[0], top_k=5)
        )

        idx1.save()

        idx2 = IVFEntityIndex(working_dir=tmp_dir, cosine_threshold=0.0)
        idx2.load()

        results_after = asyncio.get_event_loop().run_until_complete(
            idx2.query_anchors(query_embedding=vectors[0], top_k=5)
        )

        assert [r["id"] for r in results_before] == [r["id"] for r in results_after]

    def test_load_nonexistent(self, tmp_dir):
        """Loading from empty directory should return False."""
        idx = IVFEntityIndex(working_dir=tmp_dir)
        assert not idx.load()
        assert not idx.is_ready


class TestIVFStaleDetection:
    def test_stale_warning(self, tmp_dir, caplog):
        """check_stale should log warning when counts differ."""
        import logging

        ids, vectors, metadata = _make_fake_data(2000, dim=64)

        idx = IVFEntityIndex(
            working_dir=tmp_dir,
            n_clusters=32,
            min_vectors_for_index=500,
        )
        idx._build_index(ids, vectors, metadata)

        # lightrag logger has propagate=False, enable it temporarily for caplog
        from lightrag.utils import logger as lr_logger
        lr_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="lightrag"):
                idx.check_stale(2500)
            assert "stale" in caplog.text.lower()
        finally:
            lr_logger.propagate = False

    def test_no_warning_when_match(self, tmp_dir, capsys):
        """check_stale should NOT warn when counts match."""
        ids, vectors, metadata = _make_fake_data(2000, dim=64)

        idx = IVFEntityIndex(
            working_dir=tmp_dir,
            n_clusters=32,
            min_vectors_for_index=500,
        )
        idx._build_index(ids, vectors, metadata)

        idx.check_stale(2000)

        captured = capsys.readouterr()
        assert "stale" not in captured.err.lower()
