# -*- coding: utf-8 -*-
"""
Evaluate Semantic Markdown Chunks — Phase 6
=============================================

Compares chunk quality between the old fixed-token chunking (medical_rag_v3)
and the new semantic chunking (medical_rag_v4).

Metrics:
  1. Entity completeness:   avg edges per drug/disease node
  2. Orphan chunk rate:     % of chunks that produced 0 entities
  3. Token distribution:    histogram of chunk sizes
  4. LLM call efficiency:   total extraction calls from cache

Usage:
  python scripts/evaluate_chunks.py --old-dir ./medical_rag_v3 --new-dir ./medical_rag_v4
  python scripts/evaluate_chunks.py --new-dir ./medical_rag_v4   # evaluate new only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Ensure Vietnamese output is handled correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_json(filepath: Path) -> dict:
    """Load a JSON file with error handling."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  ⚠ Could not load {filepath.name}: {e}")
        return {}


class RAGStorageAnalyzer:
    """Analyze a LightRAG working directory."""

    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.name = self.working_dir.name

        # Load all storage files
        self.text_chunks = load_json(self.working_dir / "kv_store_text_chunks.json")
        self.full_docs = load_json(self.working_dir / "kv_store_full_docs.json")
        self.entity_chunks = load_json(self.working_dir / "kv_store_entity_chunks.json")
        self.relation_chunks = load_json(
            self.working_dir / "kv_store_relation_chunks.json"
        )
        self.full_entities = load_json(
            self.working_dir / "kv_store_full_entities.json"
        )
        self.full_relations = load_json(
            self.working_dir / "kv_store_full_relations.json"
        )
        self.llm_cache = load_json(
            self.working_dir / "kv_store_llm_response_cache.json"
        )

        # Try loading graph
        self.graph = self._load_graph()

    def _load_graph(self) -> Any:
        """Load networkx graph if available."""
        graphml_path = self.working_dir / "graph_chunk_entity_relation.graphml"
        if not graphml_path.exists():
            return None
        try:
            import networkx as nx
            return nx.read_graphml(graphml_path)
        except ImportError:
            print("  ⚠ networkx not installed, skipping graph analysis")
            return None
        except Exception as e:
            print(f"  ⚠ Could not load graph: {e}")
            return None

    def analyze(self) -> dict:
        """Run all analysis metrics."""
        results = {
            "name": self.name,
            "num_docs": len(self.full_docs),
            "num_chunks": len(self.text_chunks),
        }

        # 1. Token distribution
        results["token_stats"] = self._token_distribution()

        # 2. Orphan chunk rate
        results["orphan_stats"] = self._orphan_chunk_analysis()

        # 3. Entity completeness (graph-based)
        results["entity_stats"] = self._entity_completeness()

        # 4. LLM call count
        results["llm_stats"] = self._llm_call_stats()

        # 5. Context header analysis (new chunking specific)
        results["context_stats"] = self._context_header_analysis()

        return results

    def _token_distribution(self) -> dict:
        """Analyze token count distribution across chunks."""
        if not self.text_chunks:
            return {"count": 0}

        token_counts = []
        for chunk_data in self.text_chunks.values():
            tokens = chunk_data.get("tokens", 0)
            token_counts.append(tokens)

        if not token_counts:
            return {"count": 0}

        token_counts.sort()
        n = len(token_counts)

        # Build histogram buckets
        buckets = [0, 50, 100, 200, 500, 800, 1000, 1500, 2000, 3000]
        histogram = {}
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i + 1]
            count = sum(1 for t in token_counts if lo <= t < hi)
            if count > 0:
                histogram[f"{lo}-{hi}"] = count
        # >= last bucket
        count = sum(1 for t in token_counts if t >= buckets[-1])
        if count > 0:
            histogram[f"{buckets[-1]}+"] = count

        return {
            "count": n,
            "min": token_counts[0],
            "max": token_counts[-1],
            "avg": sum(token_counts) / n,
            "median": token_counts[n // 2],
            "p10": token_counts[int(n * 0.1)],
            "p90": token_counts[int(n * 0.9)],
            "histogram": histogram,
        }

    def _orphan_chunk_analysis(self) -> dict:
        """
        Identify chunks that produced 0 entity extractions.
        An 'orphan' chunk is one whose chunk_id does not appear in
        any entity's source_id list.
        """
        if not self.text_chunks:
            return {"total_chunks": 0, "orphan_chunks": 0, "orphan_rate": 0.0}

        # Collect all chunk IDs that appear in entity source_ids
        chunk_ids_with_entities = set()

        # Check entity_chunks storage (maps entity_name -> list of chunk_ids)
        for entity_name, chunk_ids in self.entity_chunks.items():
            if isinstance(chunk_ids, list):
                chunk_ids_with_entities.update(chunk_ids)
            elif isinstance(chunk_ids, dict):
                cids = chunk_ids.get("chunk_ids", [])
                if isinstance(cids, list):
                    chunk_ids_with_entities.update(cids)

        # Also check LLM cache for extraction results
        for cache_key, cache_data in self.llm_cache.items():
            if "extract" in str(cache_key):
                chunk_ids_with_entities.add(cache_key)

        total = len(self.text_chunks)
        orphan_ids = []
        for chunk_id in self.text_chunks:
            # Check if this chunk has any LLM cache entries
            chunk_data = self.text_chunks[chunk_id]
            llm_cache_list = chunk_data.get("llm_cache_list", [])
            if not llm_cache_list:
                orphan_ids.append(chunk_id)

        orphan_count = len(orphan_ids)

        # Sample orphan content for inspection
        orphan_samples = []
        for oid in orphan_ids[:3]:
            content = self.text_chunks[oid].get("content", "")[:150]
            orphan_samples.append(content)

        return {
            "total_chunks": total,
            "orphan_chunks": orphan_count,
            "orphan_rate": orphan_count / total if total > 0 else 0.0,
            "orphan_samples": orphan_samples,
        }

    def _entity_completeness(self) -> dict:
        """Analyze entity/relation density from graph storage."""
        if self.graph is None:
            # Fallback: use entity_chunks storage
            num_entities = len(self.entity_chunks) if self.entity_chunks else 0
            num_relations = len(self.relation_chunks) if self.relation_chunks else 0

            return {
                "num_entities": num_entities,
                "num_relations": num_relations,
                "graph_available": False,
            }

        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()

        # Degree distribution
        degrees = [d for _, d in self.graph.degree()]
        degrees.sort()

        # Entity type distribution (if available)
        type_counter = Counter()
        for node, data in self.graph.nodes(data=True):
            entity_type = data.get("entity_type", "unknown")
            type_counter[entity_type] = type_counter.get(entity_type, 0) + 1

        # Find drug/disease nodes with high degree (these are the anchor entities)
        high_degree_nodes = []
        for node, degree in sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)[:20]:
            data = self.graph.nodes[node]
            high_degree_nodes.append({
                "name": node,
                "degree": degree,
                "type": data.get("entity_type", "unknown"),
            })

        return {
            "num_entities": num_nodes,
            "num_relations": num_edges,
            "graph_available": True,
            "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "median_degree": degrees[len(degrees) // 2] if degrees else 0,
            "entity_types": dict(type_counter.most_common(15)),
            "top_nodes": high_degree_nodes,
        }

    def _llm_call_stats(self) -> dict:
        """Count LLM extraction calls."""
        if not self.llm_cache:
            return {"total_cache_entries": 0}

        extract_calls = 0
        summary_calls = 0
        other_calls = 0

        for key in self.llm_cache:
            key_str = str(key)
            if "extract" in key_str:
                extract_calls += 1
            elif "summary" in key_str or "summarize" in key_str:
                summary_calls += 1
            else:
                other_calls += 1

        return {
            "total_cache_entries": len(self.llm_cache),
            "extract_calls": extract_calls,
            "summary_calls": summary_calls,
            "other_calls": other_calls,
        }

    def _context_header_analysis(self) -> dict:
        """Check how many chunks have context headers (semantic chunking indicator)."""
        if not self.text_chunks:
            return {"chunks_with_context": 0}

        with_context = 0
        without_context = 0

        for chunk_data in self.text_chunks.values():
            content = chunk_data.get("content", "")
            if content.startswith("[Chủ đề:") or content.startswith("[Chu de:"):
                with_context += 1
            else:
                without_context += 1

        return {
            "chunks_with_context": with_context,
            "chunks_without_context": without_context,
            "context_rate": with_context / (with_context + without_context)
            if (with_context + without_context) > 0
            else 0.0,
        }


def print_comparison(old_results: dict | None, new_results: dict) -> None:
    """Print a side-by-side comparison table."""

    def _col(val, width=15):
        return str(val)[:width].ljust(width)

    header = "=" * 70
    print(f"\n{header}")
    print("  CHUNK QUALITY EVALUATION REPORT")
    print(header)

    if old_results:
        print(f"\n{'Metric':<35} {'Old (' + old_results['name'] + ')':<17} {'New (' + new_results['name'] + ')':<17}")
        print("-" * 70)

        # Basic stats
        print(f"{'Documents':<35} {_col(old_results['num_docs'])} {_col(new_results['num_docs'])}")
        print(f"{'Total chunks':<35} {_col(old_results['num_chunks'])} {_col(new_results['num_chunks'])}")

        # Token stats
        ot = old_results["token_stats"]
        nt = new_results["token_stats"]
        if ot.get("count", 0) > 0 and nt.get("count", 0) > 0:
            ot_avg = f"{ot['avg']:.0f}"
            nt_avg = f"{nt['avg']:.0f}"
            ot_range = f"{ot['min']}-{ot['max']}"
            nt_range = f"{nt['min']}-{nt['max']}"
            print(f"{'Avg tokens/chunk':<35} {_col(ot_avg)} {_col(nt_avg)}")
            print(f"{'Median tokens/chunk':<35} {_col(ot['median'])} {_col(nt['median'])}")
            print(f"{'Token range':<35} {_col(ot_range)} {_col(nt_range)}")

        # Orphan stats
        oo = old_results["orphan_stats"]
        no = new_results["orphan_stats"]
        print(f"{'Orphan chunks':<35} {_col(oo['orphan_chunks'])} {_col(no['orphan_chunks'])}")
        oo_rate = f"{oo['orphan_rate']:.1%}"
        no_rate = f"{no['orphan_rate']:.1%}"
        print(f"{'Orphan rate':<35} {_col(oo_rate)} {_col(no_rate)}")

        # Entity stats
        oe = old_results["entity_stats"]
        ne = new_results["entity_stats"]
        print(f"{'Entities in graph':<35} {_col(oe['num_entities'])} {_col(ne['num_entities'])}")
        print(f"{'Relations in graph':<35} {_col(oe['num_relations'])} {_col(ne['num_relations'])}")
        if oe.get("graph_available") and ne.get("graph_available"):
            oe_deg = f"{oe['avg_degree']:.1f}"
            ne_deg = f"{ne['avg_degree']:.1f}"
            print(f"{'Avg node degree':<35} {_col(oe_deg)} {_col(ne_deg)}")
            print(f"{'Max node degree':<35} {_col(oe['max_degree'])} {_col(ne['max_degree'])}")

        # Context headers
        oc = old_results["context_stats"]
        nc = new_results["context_stats"]
        print(f"{'Chunks with context header':<35} {_col(oc['chunks_with_context'])} {_col(nc['chunks_with_context'])}")

        # LLM calls
        ol = old_results["llm_stats"]
        nl = new_results["llm_stats"]
        print(f"{'LLM extract calls':<35} {_col(ol['extract_calls'])} {_col(nl['extract_calls'])}")
    else:
        # Single-side report
        print(f"\n{'Metric':<35} {'Value':<20}")
        print("-" * 55)
        print(f"{'Documents':<35} {new_results['num_docs']}")
        print(f"{'Total chunks':<35} {new_results['num_chunks']}")

        nt = new_results["token_stats"]
        if nt.get("count", 0) > 0:
            print(f"{'Avg tokens/chunk':<35} {nt['avg']:.0f}")
            print(f"{'Median tokens/chunk':<35} {nt['median']}")
            print(f"{'Token range':<35} {nt['min']}–{nt['max']}")
            print(f"{'P10/P90 tokens':<35} {nt.get('p10', 'N/A')} / {nt.get('p90', 'N/A')}")

        no = new_results["orphan_stats"]
        print(f"{'Orphan chunks':<35} {no['orphan_chunks']}")
        print(f"{'Orphan rate':<35} {no['orphan_rate']:.1%}")

        ne = new_results["entity_stats"]
        print(f"{'Entities in graph':<35} {ne['num_entities']}")
        print(f"{'Relations in graph':<35} {ne['num_relations']}")
        if ne.get("graph_available"):
            print(f"{'Avg node degree':<35} {ne['avg_degree']:.1f}")
            print(f"{'Max node degree':<35} {ne['max_degree']}")

        nc = new_results["context_stats"]
        print(f"{'Chunks with context header':<35} {nc['chunks_with_context']}")
        print(f"{'Context header rate':<35} {nc['context_rate']:.1%}")

    # Top entities (from new)
    ne = new_results["entity_stats"]
    if ne.get("top_nodes"):
        print(f"\n{'Top Entities by Degree (New):':}")
        print(f"  {'Name':<40} {'Degree':<10} {'Type':<20}")
        print("  " + "-" * 65)
        for node in ne["top_nodes"][:10]:
            print(f"  {node['name'][:40]:<40} {node['degree']:<10} {node['type']:<20}")

    # Token histogram (from new)
    nt = new_results["token_stats"]
    if nt.get("histogram"):
        print(f"\nToken Distribution (New):")
        for bucket, count in nt["histogram"].items():
            bar = "█" * min(count, 50)
            print(f"  {bucket:>10}: {count:>4} {bar}")

    print(f"\n{header}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate chunk quality for LightRAG medical data"
    )
    parser.add_argument(
        "--old-dir",
        type=str,
        default=None,
        help="Working directory of old fixed-token chunking (e.g., ./medical_rag_v3)",
    )
    parser.add_argument(
        "--new-dir",
        type=str,
        required=True,
        help="Working directory of new semantic chunking (e.g., ./medical_rag_v4)",
    )
    args = parser.parse_args()

    old_results = None
    if args.old_dir:
        old_dir = Path(args.old_dir)
        if old_dir.exists():
            print(f"Analyzing old storage: {old_dir}")
            analyzer = RAGStorageAnalyzer(str(old_dir))
            old_results = analyzer.analyze()
        else:
            print(f"⚠ Old directory not found: {old_dir}")

    new_dir = Path(args.new_dir)
    if not new_dir.exists():
        print(f"ERROR: New directory not found: {new_dir}")
        sys.exit(1)

    print(f"Analyzing new storage: {new_dir}")
    new_analyzer = RAGStorageAnalyzer(str(new_dir))
    new_results = new_analyzer.analyze()

    print_comparison(old_results, new_results)


if __name__ == "__main__":
    main()
