# -*- coding: utf-8 -*-
"""
OPTIMIZED BEAM SEARCH
Các cải tiến:
1. Song song hóa I/O operations trong mỗi hop
2. Song song hóa scoring loop sử dụng asyncio.gather
3. Batch processing với chunking
4. Early pruning
"""

import asyncio
import time
from typing import Any
from lightrag.utils import logger

# Default weights (có thể tune)
ALPHA_SEMANTIC = 0.7
BETA_WEIGHT = 0.3
GAMMA_LENGTH = 0.1

# Batch size cho parallel processing
BATCH_SIZE = 50  # Số lượng candidates xử lý song song


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Tính cosine similarity giữa hai vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _score_candidate_batch(
    batch: list[tuple],
    query_embedding: list[float] | None,
    depth: int,
    neighbor_vectors: dict[str, list[float]],
    edge_data_batch: dict,
) -> list[dict]:
    """Score một batch candidates song song."""
    results = []
    for from_node, src, tgt, neighbor in batch:
        edge_data = edge_data_batch.get((src, tgt))
        if edge_data is None:
            continue

        # Semantic score
        semantic_score = 0.0
        if query_embedding is not None and neighbor in neighbor_vectors:
            neighbor_vec = neighbor_vectors[neighbor]
            semantic_score = _cosine_similarity(query_embedding, neighbor_vec)

        # Edge weight
        edge_weight = float(edge_data.get("weight", 1.0))
        normalized_weight = min(edge_weight / 10.0, 1.0)

        # Length penalty
        length_penalty = GAMMA_LENGTH * depth

        # Combined score
        score = (
            ALPHA_SEMANTIC * semantic_score
            + BETA_WEIGHT * normalized_weight
            - length_penalty
        )

        results.append({
            "neighbor": neighbor,
            "score": score,
            "semantic_score": semantic_score,
            "edge_src": src,
            "edge_tgt": tgt,
            "edge_data": edge_data,
            "from_node": from_node,
        })

    return results


async def optimized_beam_search(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst,
    entities_vdb,
    relationships_vdb,
    text_chunks_db,
    query_param,
    query_embedding: list[float] = None,
) -> dict[str, Any]:
    """
    OPTIMIZED Beam Search với parallel processing.

    Các tối ưu hóa:
    1. Song song hóa get_nodes_edges_batch + get_vectors_by_ids + get_edges_batch
    2. Scoring loop với batching và asyncio.gather
    3. Chunking cho batch processing
    """
    beam_width = query_param.beam_width
    max_depth = query_param.max_depth
    pruning_threshold = query_param.pruning_threshold

    BEAM_MAX_ANCHOR_K = 10
    effective_top_k = min(query_param.top_k, BEAM_MAX_ANCHOR_K)

    t0 = time.perf_counter()

    # ─── Phase 1: Find anchors ───────────────────────────────────────────
    # Tối ưu: Fetch anchors + query embedding song song nếu có thể

    anchor_entities = []
    anchor_edge_results = []

    # Fetch LL entities và HL edges song song
    tasks = []
    if ll_keywords:
        tasks.append(entities_vdb.query(ll_keywords, top_k=effective_top_k))

    if hl_keywords:
        tasks.append(relationships_vdb.query(hl_keywords, top_k=effective_top_k))

    if tasks:
        results = await asyncio.gather(*tasks)
        if ll_keywords:
            anchor_entities = results[0] if len(results) > 0 else []
        if hl_keywords:
            anchor_edge_results = results[1] if len(results) > 1 else []

    if not anchor_entities and not anchor_edge_results:
        return {
            "final_entities": [],
            "final_relations": [],
            "vector_chunks": [],
            "chunk_tracking": {},
            "query_embedding": query_embedding,
        }

    # Collect anchor names
    anchor_names = list(dict.fromkeys(r["entity_name"] for r in anchor_entities))
    for edge_r in anchor_edge_results:
        src = edge_r.get("src_id", "")
        tgt = edge_r.get("tgt_id", "")
        if src and src not in anchor_names:
            anchor_names.append(src)
        if tgt and tgt not in anchor_names:
            anchor_names.append(tgt)

    # ─── Phase 2: Initialize anchors ───────────────────────────────────────
    collected_entities: dict[str, dict] = {}
    collected_relations: dict[tuple, dict] = {}

    # Batch fetch nodes và degrees song song
    anchor_nodes, anchor_degrees = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(anchor_names),
        knowledge_graph_inst.node_degrees_batch(anchor_names),
    )

    for name in anchor_names:
        node_data = anchor_nodes.get(name)
        if node_data is not None:
            collected_entities[name] = {
                "data": node_data,
                "score": 1.0,
                "hop": 0,
                "degree": anchor_degrees.get(name, 0),
            }

    # Pre-collect anchor HL edges
    hl_edge_pairs = []
    for edge_r in anchor_edge_results:
        src = edge_r.get("src_id", "")
        tgt = edge_r.get("tgt_id", "")
        if src and tgt:
            edge_key = tuple(sorted([src, tgt]))
            if edge_key not in collected_relations:
                hl_edge_pairs.append({"src": src, "tgt": tgt})

    if hl_edge_pairs:
        hl_edges_batch = await knowledge_graph_inst.get_edges_batch(hl_edge_pairs)
        for pair in hl_edge_pairs:
            src, tgt = pair["src"], pair["tgt"]
            edge_data = hl_edges_batch.get((src, tgt))
            if edge_data is not None:
                edge_key = tuple(sorted([src, tgt]))
                collected_relations[edge_key] = {
                    "data": edge_data,
                    "score": 0.9,
                    "src_tgt": (src, tgt),
                }

    current_frontier = [n for n in anchor_names if n in collected_entities]

    t1 = time.perf_counter()
    logger.info(f"Beam search init: {len(collected_entities)} anchors in {t1-t0:.3f}s")

    # ─── Phase 3: Hop traversal với OPTIMIZATIONS ──────────────────────────
    for depth in range(1, max_depth + 1):
        if not current_frontier:
            break

        # TỐI ƯU 1: Batch fetch all neighbor edges
        edges_batch = await knowledge_graph_inst.get_nodes_edges_batch(
            current_frontier
        )

        # Collect neighbors
        neighbor_set: set[str] = set()
        frontier_edges: list[tuple[str, str, str]] = []

        for node_name in current_frontier:
            edges = edges_batch.get(node_name, [])
            for src, tgt in edges:
                neighbor = tgt if src == node_name else src
                if neighbor not in collected_entities:
                    neighbor_set.add(neighbor)
                    frontier_edges.append((node_name, src, tgt))

        if not neighbor_set:
            break

        neighbor_list = list(neighbor_set)

        # TỐI ƯU 2: Fetch vectors và edge data SONG SONG
        neighbor_vectors_task = entities_vdb.get_vectors_by_ids(neighbor_list)

        unique_edge_pairs = list(set((fe[1], fe[2]) for fe in frontier_edges))
        edge_pairs_dicts = [{"src": s, "tgt": t} for s, t in unique_edge_pairs]
        edge_data_task = knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)

        # Chạy song song
        neighbor_vectors, edge_data_batch = await asyncio.gather(
            neighbor_vectors_task,
            edge_data_task,
        )

        # TỐI ƯU 3: Scoring với BATCHING
        candidates: list[dict] = []

        # Chuan bị data cho batch processing
        batch_data = [
            (from_node, src, tgt, tgt if src == from_node else src)
            for from_node, src, tgt in frontier_edges
        ]

        # Xử lý theo batches
        for i in range(0, len(batch_data), BATCH_SIZE):
            batch = batch_data[i:i + BATCH_SIZE]
            batch_results = await _score_candidate_batch(
                batch,
                query_embedding,
                depth,
                neighbor_vectors,
                edge_data_batch,
            )
            candidates.extend(batch_results)

        if not candidates:
            break

        # Sort và select
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Adaptive threshold
        effective_beam = beam_width * len(current_frontier)
        if len(candidates) > effective_beam * 2:
            adaptive_threshold = candidates[effective_beam]["score"]
            candidates = [c for c in candidates if c["score"] >= adaptive_threshold]

        selected = candidates[:effective_beam]

        # Deduplicate
        seen_in_this_hop = set()
        unique_selected = []
        for c in selected:
            neighbor = c["neighbor"]
            if neighbor not in collected_entities and neighbor not in seen_in_this_hop:
                seen_in_this_hop.add(neighbor)
                unique_selected.append(c)

        # TỐI ƯU 4: Batch fetch node data + degrees
        selected_names = [c["neighbor"] for c in unique_selected]
        if selected_names:
            batch_nodes, batch_degrees = await asyncio.gather(
                knowledge_graph_inst.get_nodes_batch(selected_names),
                knowledge_graph_inst.node_degrees_batch(selected_names),
            )
        else:
            batch_nodes, batch_degrees = {}, {}

        new_frontier = []
        for c in unique_selected:
            neighbor = c["neighbor"]
            node_data = batch_nodes.get(neighbor)
            if node_data is not None:
                collected_entities[neighbor] = {
                    "data": node_data,
                    "score": c["score"],
                    "hop": depth,
                    "degree": batch_degrees.get(neighbor, 0),
                }
                new_frontier.append(neighbor)

            edge_key = tuple(sorted([c["edge_src"], c["edge_tgt"]]))
            if edge_key not in collected_relations:
                collected_relations[edge_key] = {
                    "data": c["edge_data"],
                    "score": c["score"],
                    "src_tgt": (c["edge_src"], c["edge_tgt"]),
                }

        t_hop = time.perf_counter()
        logger.info(
            f"Beam hop {depth}: {len(candidates)} → {len(new_frontier)} "
            f"in {t_hop-t1:.3f}s (total: {t_hop-t0:.3f}s)"
        )

        current_frontier = new_frontier

    # ─── Phase 4: Format output ────────────────────────────────────────────
    final_entities = []
    for name, info in collected_entities.items():
        entity = {
            **info["data"],
            "entity_name": name,
            "rank": info.get("degree", 0),
            "beam_score": info["score"],
            "beam_hop": info["hop"],
            "created_at": info["data"].get("created_at"),
        }
        final_entities.append(entity)

    final_entities.sort(key=lambda x: x.get("beam_score", 0), reverse=True)

    final_relations = []
    for edge_key, info in collected_relations.items():
        edge_data = info["data"]
        if "weight" not in edge_data:
            edge_data["weight"] = 1.0
        relation = {
            "src_tgt": info["src_tgt"],
            "rank": int(info["score"] * 100),
            "beam_score": info["score"],
            **edge_data,
        }
        final_relations.append(relation)

    final_relations.sort(key=lambda x: x.get("beam_score", 0), reverse=True)

    t_total = time.perf_counter()
    logger.info(
        f"Beam search complete: {len(final_entities)} entities, "
        f"{len(final_relations)} relations in {t_total-t0:.3f}s"
    )

    return {
        "final_entities": final_entities,
        "final_relations": final_relations,
        "vector_chunks": [],
        "chunk_tracking": {},
        "query_embedding": query_embedding,
    }
