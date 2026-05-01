"""
Build BM25 Index — One-time script for Beam Search Hybrid Retrieval.

Reads entity and chunk data from the LightRAG working directory,
builds BM25Okapi indices, and saves them as pickle files:
  - bm25_entities.pkl
  - bm25_chunks.pkl

Usage:
  python build_bm25_index.py [--working-dir ./medical_rag/medical_rag_v2]

Typical runtime: < 5 seconds.
Does NOT modify any LightRAG data files (insert/upsert logic untouched).

KLTN — Graduation Thesis
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index for Beam Search Hybrid Retrieval")
    parser.add_argument(
        "--working-dir",
        default="./medical_rag_v6",
        help="Path to the LightRAG working directory (default: ./medical_rag_v6)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Build BM25 Index -- KLTN Beam Search Hybrid Retrieval")
    print("=" * 60)
    print(f"Working dir: {args.working_dir}")

    from lightrag.bm25_storage import BM25Storage

    storage = BM25Storage.build_and_save(args.working_dir)

    print(f"\nDone!")
    print(f"   Entities: {len(storage.entity_ids)}")
    print(f"   Chunks:   {len(storage.chunk_ids)}")
    print(f"   Files:    bm25_entities.pkl, bm25_chunks.pkl")


if __name__ == "__main__":
    main()
