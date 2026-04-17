# -*- coding: utf-8 -*-
"""
Benchmark: Original vs Optimized Beam Search
Test trên 5 câu hỏi với beam_width=5, max_depth=2
"""

import os
import sys
import asyncio
import time
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv(dotenv_path=r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\.env")

WORKING_DIR = "./medical_rag/medical_rag_v2"

# ================================
# EMBEDDING FUNCTION (Ollama)
# ================================
async def ollama_embed_func(texts: list[str]) -> np.ndarray:
    import httpx
    embed_url = "http://localhost:11434/api/embeddings"
    embeddings = []
    for text in texts:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                embed_url,
                json={"model": "embeddinggemma:300m", "prompt": text}
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            embeddings.append(np.array(embedding, dtype=np.float32))
    return np.array(embeddings, dtype=np.float32)

# ================================
# LLM function
# ================================
async def llm_func(prompt: str, **kwargs) -> str:
    return await gemini_model_complete(
        prompt=prompt,
        model_name="gemini-2.5-flash-lite",
        **kwargs
    )

# ================================
# TEST QUESTIONS
# ================================
TEST_QUESTIONS = [
    "Thuốc AlphaDHG có thể gây ra những tác dụng phụ nào?",
    "Các vỉ thuốc Amlor 5 Pfizer có bao nhiêu viên thuốc?",
    "Sử dụng chung Davita Bone Sugar free với thuốc lợi tiểu nhóm thiazid có thể gây ra tác dụng phụ gì?",
    "Phụ nữ nào cần lưu ý khi sử dụng thuốc Coldko?",
    "Tôi nên làm gì nếu muốn bắt đầu, ngừng hoặc thay đổi bất kỳ loại thuốc nào trong khi đang sử dụng Acnequidt 20 ml?",
]

# ================================
# ORIGINAL BEAM SEARCH (từ operate.py)
# ================================
def _cosine_similarity_original(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# Copy hàm gốc từ operate.py
async def original_beam_search(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst,
    entities_vdb,
    relationships_vdb,
    query_param,
    query_embedding: list[float] = None,
):
    """Original beam search từ lightrag/operate.py"""
    from lightrag.operate import _beam_search_graph
    return await _beam_search_graph(
        query=query,
        ll_keywords=ll_keywords,
        hl_keywords=hl_keywords,
        knowledge_graph_inst=knowledge_graph_inst,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_db=None,
        query_param=query_param,
        query_embedding=query_embedding,
    )

# ================================
# OPTIMIZED BEAM SEARCH
# ================================
ALPHA_SEMANTIC = 0.7
BETA_WEIGHT = 0.3
GAMMA_LENGTH = 0.1
BATCH_SIZE = 50

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

async def _score_batch_async(
    batch: list,
    query_embedding,
    depth: int,
    neighbor_vectors: dict,
    edge_data_batch: dict,
    pruning_threshold: float = 0.0,
):
    """Score batch với một chút async delay để yield control."""
    results = []
    for from_node, src, tgt, neighbor in batch:
        edge_data = edge_data_batch.get((src, tgt))
        if edge_data is None:
            continue

        semantic_score = 0.0
        if query_embedding is not None and neighbor in neighbor_vectors:
            neighbor_vec = neighbor_vectors[neighbor]
            semantic_score = _cosine_similarity(query_embedding, neighbor_vec)

        edge_weight = float(edge_data.get("weight", 1.0))
        normalized_weight = min(edge_weight / 10.0, 1.0)
        length_penalty = GAMMA_LENGTH * depth

        score = (
            ALPHA_SEMANTIC * semantic_score
            + BETA_WEIGHT * normalized_weight
            - length_penalty
        )

        # FIX: Apply pruning threshold
        if score < pruning_threshold:
            continue

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
    query_param,
    query_embedding: list[float] = None,
):
    """
    OPTIMIZED beam search với parallel processing.
    """
    beam_width = query_param.beam_width
    max_depth = query_param.max_depth
    pruning_threshold = query_param.pruning_threshold

    BEAM_MAX_ANCHOR_K = 10
    effective_top_k = min(query_param.top_k, BEAM_MAX_ANCHOR_K)

    t0 = time.perf_counter()

    # Phase 1: Find anchors
    anchor_entities = []
    anchor_edge_results = []

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
        return {"final_entities": [], "final_relations": [], "vector_chunks": [], "chunk_tracking": {}, "query_embedding": query_embedding}

    anchor_names = list(dict.fromkeys(r["entity_name"] for r in anchor_entities))
    for edge_r in anchor_edge_results:
        src = edge_r.get("src_id", "")
        tgt = edge_r.get("tgt_id", "")
        if src and src not in anchor_names:
            anchor_names.append(src)
        if tgt and tgt not in anchor_names:
            anchor_names.append(tgt)

    # Phase 2: Initialize
    collected_entities = {}
    collected_relations = {}

    # OPTIMIZED: Batch fetch song song
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

    # Phase 3: Hop traversal
    for depth in range(1, max_depth + 1):
        if not current_frontier:
            break

        t_hop_start = time.perf_counter()

        edges_batch = await knowledge_graph_inst.get_nodes_edges_batch(current_frontier)

        neighbor_set = set()
        frontier_edges = []

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

        # OPTIMIZED: Fetch vectors và edges SONG SONG
        neighbor_vectors_task = entities_vdb.get_vectors_by_ids(neighbor_list)
        unique_edge_pairs = list(set((fe[1], fe[2]) for fe in frontier_edges))
        edge_pairs_dicts = [{"src": s, "tgt": t} for s, t in unique_edge_pairs]
        edge_data_task = knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)

        neighbor_vectors, edge_data_batch = await asyncio.gather(
            neighbor_vectors_task, edge_data_task
        )

        # OPTIMIZED: Batch scoring
        candidates = []
        batch_data = [
            (from_node, src, tgt, tgt if src == from_node else src)
            for from_node, src, tgt in frontier_edges
        ]

        for i in range(0, len(batch_data), BATCH_SIZE):
            batch = batch_data[i:i + BATCH_SIZE]
            batch_results = await _score_batch_async(
                batch, query_embedding, depth, neighbor_vectors, edge_data_batch, pruning_threshold
            )
            candidates.extend(batch_results)
            # Yield control để không block event loop
            await asyncio.sleep(0)

        if not candidates:
            break

        candidates.sort(key=lambda x: x["score"], reverse=True)

        effective_beam = beam_width * len(current_frontier)
        if len(candidates) > effective_beam * 2:
            adaptive_threshold = candidates[effective_beam]["score"]
            candidates = [c for c in candidates if c["score"] >= adaptive_threshold]

        selected = candidates[:effective_beam]

        seen_in_this_hop = set()
        unique_selected = []
        for c in selected:
            neighbor = c["neighbor"]
            if neighbor not in collected_entities and neighbor not in seen_in_this_hop:
                seen_in_this_hop.add(neighbor)
                unique_selected.append(c)

        # OPTIMIZED: Batch fetch nodes + degrees
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

        t_hop_end = time.perf_counter()
        print(f"    Hop {depth}: {len(candidates)} -> {len(new_frontier)} in {t_hop_end - t_hop_start:.3f}s")

        current_frontier = new_frontier

    # Phase 4: Format
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
    print(f"    Total: {len(final_entities)} entities, {len(final_relations)} relations in {t_total - t0:.3f}s")

    return {
        "final_entities": final_entities,
        "final_relations": final_relations,
        "vector_chunks": [],
        "chunk_tracking": {},
        "query_embedding": query_embedding,
        "elapsed": t_total - t0,
    }

# ================================
# HELPER: Extract keywords
# ================================
async def extract_keywords_simple(rag, query):
    """Extract keywords đơn giản (tái sử dụng cache nếu có)."""
    # Đơn giản: split query thành keywords
    keywords = query.replace("?", " ").replace(",", " ").split()
    return keywords, keywords[:3]  # ll_keywords, hl_keywords

# ================================
# MAIN TEST
# ================================
async def main():
    print("="*80)
    print("BENCHMARK: ORIGINAL vs OPTIMIZED BEAM SEARCH")
    print("Config: beam_width=5, max_depth=2")
    print("="*80)

    # Initialize RAG
    embedding_func = EmbeddingFunc(
        embedding_dim=768,
        max_token_size=512,
        func=ollama_embed_func
    )

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_func,
        llm_model_name="gemini-2.5-flash-lite",
        embedding_func=embedding_func,
    )

    await rag.initialize_storages()

    # Get storages
    entities_vdb = rag.entities_vdb
    relationships_vdb = rag.relationships_vdb
    knowledge_graph_inst = rag.chunk_entity_relation_graph

    param = QueryParam(mode="beam", beam_width=5, max_depth=2)

    results = {"original": [], "optimized": []}

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*80}")
        print(f"CÂU {i}: {question}")
        print(f"{'='*80}")

        # Extract keywords (dùng simple method để tránh LLM)
        ll_keywords = question.replace("?", " ").replace(",", " ")
        hl_keywords = ll_keywords

        # Query embedding
        emb_start = time.time()
        query_embedding_arr = await embedding_func([question])
        query_embedding = query_embedding_arr[0].tolist() if isinstance(query_embedding_arr, np.ndarray) else query_embedding_arr[0]
        emb_time = time.time() - emb_start

        # === ORIGINAL ===
        print(f"\n[ORIGINAL BEAM SEARCH]")
        orig_start = time.time()

        orig_result = await original_beam_search(
            query=question,
            ll_keywords=ll_keywords,
            hl_keywords=hl_keywords,
            knowledge_graph_inst=knowledge_graph_inst,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            query_param=param,
            query_embedding=query_embedding,
        )

        orig_time = time.time() - orig_start
        orig_entities = len(orig_result.get("final_entities", []))
        orig_relations = len(orig_result.get("final_relations", []))

        print(f"  -> Entities: {orig_entities}, Relations: {orig_relations}")
        print(f"  -> Time: {orig_time:.3f}s (embedding: {emb_time:.3f}s)")

        results["original"].append({
            "question": question,
            "entities": orig_entities,
            "relations": orig_relations,
            "time": orig_time,
            "embedding_time": emb_time,
        })

        # === OPTIMIZED ===
        print(f"\n[OPTIMIZED BEAM SEARCH]")
        opt_start = time.time()

        opt_result = await optimized_beam_search(
            query=question,
            ll_keywords=ll_keywords,
            hl_keywords=hl_keywords,
            knowledge_graph_inst=knowledge_graph_inst,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            query_param=param,
            query_embedding=query_embedding,
        )

        opt_time = time.time() - opt_start
        opt_entities = len(opt_result.get("final_entities", []))
        opt_relations = len(opt_result.get("final_relations", []))

        print(f"  -> Entities: {opt_entities}, Relations: {opt_relations}")
        print(f"  -> Time: {opt_time:.3f}s (embedding: {emb_time:.3f}s)")

        results["optimized"].append({
            "question": question,
            "entities": opt_entities,
            "relations": opt_relations,
            "time": opt_time,
            "embedding_time": emb_time,
        })

        # === COMPARISON ===
        speedup = orig_time / opt_time if opt_time > 0 else 0
        print(f"\n[SO SÁNH]")
        print(f"  Original: {orig_time:.3f}s")
        print(f"  Optimized: {opt_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Entities match: {orig_entities == opt_entities}")
        print(f"  Relations match: {orig_relations == opt_relations}")

    # === TONG KET ===
    print(f"\n{'='*80}")
    print("TỔNG KẾT")
    print(f"{'='*80}")

    orig_total = sum(r["time"] for r in results["original"])
    opt_total = sum(r["time"] for r in results["optimized"])
    orig_entities_total = sum(r["entities"] for r in results["original"])
    opt_entities_total = sum(r["entities"] for r in results["optimized"])
    orig_relations_total = sum(r["relations"] for r in results["original"])
    opt_relations_total = sum(r["relations"] for r in results["optimized"])

    print(f"\n{'Metric':<25} {'Original':<15} {'Optimized':<15} {'Change'}")
    print("-" * 70)
    print(f"{'Total Time':<25} {orig_total:.3f}s{'':<8} {opt_total:.3f}s{'':<8} {((opt_total-orig_total)/orig_total*100):+.1f}%")
    print(f"{'Total Entities':<25} {orig_entities_total:<15} {opt_entities_total:<15} {'✓ Match' if orig_entities_total == opt_entities_total else '✗ Diff'}")
    print(f"{'Total Relations':<25} {orig_relations_total:<15} {opt_relations_total:<15} {'✓ Match' if orig_relations_total == opt_relations_total else '✗ Diff'}")

    print(f"\n{'Câu':<5} {'Original (s)':<15} {'Optimized (s)':<15} {'Speedup':<10} {'Entities Match'}")
    print("-" * 60)
    for i, (orig, opt) in enumerate(zip(results["original"], results["optimized"]), 1):
        speedup = orig["time"] / opt["time"] if opt["time"] > 0 else 0
        match = "✓" if orig["entities"] == opt["entities"] and orig["relations"] == opt["relations"] else "✗"
        print(f"{i:<5} {orig['time']:<15.3f} {opt['time']:<15.3f} {speedup:<10.2f}x {match}")

    avg_speedup = orig_total / opt_total if opt_total > 0 else 0
    print(f"\n{'='*40}")
    print(f"AVERAGE SPEEDUP: {avg_speedup:.2f}x")
    print(f"{'='*40}")

    await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
