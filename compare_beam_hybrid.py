# -*- coding: utf-8 -*-
"""
Compare: Optimized Beam Search vs Hybrid Search
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

        if score < pruning_threshold:
            continue

        results.append({
            "neighbor": neighbor,
            "score": score,
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
    """Optimized beam search với parallel processing."""
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
        return {"final_entities": [], "final_relations": [], "time": time.perf_counter() - t0}

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

        # Parallel fetch
        neighbor_vectors_task = entities_vdb.get_vectors_by_ids(neighbor_list)
        unique_edge_pairs = list(set((fe[1], fe[2]) for fe in frontier_edges))
        edge_pairs_dicts = [{"src": s, "tgt": t} for s, t in unique_edge_pairs]
        edge_data_task = knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)

        neighbor_vectors, edge_data_batch = await asyncio.gather(
            neighbor_vectors_task, edge_data_task
        )

        # Batch scoring
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

    elapsed = time.perf_counter() - t0

    return {
        "final_entities": final_entities,
        "final_relations": final_relations,
        "time": elapsed,
    }

# ================================
# MAIN TEST
# ================================
async def main():
    print("="*80)
    print("SO SÁNH: OPTIMIZED BEAM SEARCH vs HYBRID SEARCH")
    print("Config: beam_width=5, max_depth=2 | 5 câu hỏi")
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

    entities_vdb = rag.entities_vdb
    relationships_vdb = rag.relationships_vdb
    knowledge_graph_inst = rag.chunk_entity_relation_graph

    results = {"beam": [], "hybrid": []}

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*80}")
        print(f"CÂU {i}: {question}")
        print(f"{'='*80}")

        ll_keywords = question.replace("?", " ").replace(",", " ")
        hl_keywords = ll_keywords

        # Query embedding
        emb_start = time.time()
        query_embedding_arr = await embedding_func([question])
        query_embedding = query_embedding_arr[0].tolist() if isinstance(query_embedding_arr, np.ndarray) else query_embedding_arr[0]
        emb_time = time.time() - emb_start

        # === OPTIMIZED BEAM SEARCH ===
        print(f"\n[OPTIMIZED BEAM SEARCH]")
        beam_param = QueryParam(mode="beam", beam_width=5, max_depth=2, top_k=10)

        beam_start = time.time()
        beam_result = await optimized_beam_search(
            query=question,
            ll_keywords=ll_keywords,
            hl_keywords=hl_keywords,
            knowledge_graph_inst=knowledge_graph_inst,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            query_param=beam_param,
            query_embedding=query_embedding,
        )
        beam_time = time.time() - beam_start

        beam_entities = len(beam_result["final_entities"])
        beam_relations = len(beam_result["final_relations"])
        beam_entity_names = {e["entity_name"] for e in beam_result["final_entities"]}
        beam_relation_pairs = {r["src_tgt"] for r in beam_result["final_relations"]}

        print(f"  -> Entities: {beam_entities}, Relations: {beam_relations}")
        print(f"  -> Time: {beam_time:.3f}s (embedding: {emb_time:.3f}s)")

        results["beam"].append({
            "question": question,
            "entities": beam_entities,
            "relations": beam_relations,
            "time": beam_time,
            "entity_names": beam_entity_names,
            "relation_pairs": beam_relation_pairs,
        })

        # === HYBRID SEARCH (sử dụng aquery_data) ===
        print(f"\n[HYBRID SEARCH]")
        hybrid_param = QueryParam(mode="hybrid", top_k=10)

        hybrid_start = time.time()
        hybrid_result = await rag.aquery_data(
            question,
            param=hybrid_param
        )
        hybrid_time = time.time() - hybrid_start

        # Extract data từ hybrid result
        hybrid_data = hybrid_result.get("data", {})
        hybrid_entities_list = hybrid_data.get("entities", [])
        hybrid_relations_list = hybrid_data.get("relationships", [])

        hybrid_entities = len(hybrid_entities_list)
        hybrid_relations = len(hybrid_relations_list)
        hybrid_entity_names = {e["entity_name"] for e in hybrid_entities_list}
        hybrid_relation_pairs = {tuple(sorted(r["src_tgt"])) for r in hybrid_relations_list if "src_tgt" in r}

        print(f"  -> Entities: {hybrid_entities}, Relations: {hybrid_relations}")
        print(f"  -> Time: {hybrid_time:.3f}s")

        results["hybrid"].append({
            "question": question,
            "entities": hybrid_entities,
            "relations": hybrid_relations,
            "time": hybrid_time,
            "entity_names": hybrid_entity_names,
            "relation_pairs": hybrid_relation_pairs,
        })

        # === SO SÁNH ===
        print(f"\n[SO SÁNH]")
        common_entities = beam_entity_names & hybrid_entity_names
        all_entities = beam_entity_names | hybrid_entity_names
        entity_overlap = len(common_entities) / len(all_entities) * 100 if all_entities else 0

        common_relations = beam_relation_pairs & hybrid_relation_pairs
        all_relations = beam_relation_pairs | hybrid_relation_pairs
        relation_overlap = len(common_relations) / len(all_relations) * 100 if all_relations else 0

        print(f"  Beam entities: {beam_entities}")
        print(f"  Hybrid entities: {hybrid_entities}")
        print(f"  Common entities: {len(common_entities)} ({entity_overlap:.1f}% overlap)")
        print(f"  Only in Beam: {len(beam_entity_names - hybrid_entity_names)}")
        print(f"  Only in Hybrid: {len(hybrid_entity_names - beam_entity_names)}")
        print(f"")
        print(f"  Beam relations: {beam_relations}")
        print(f"  Hybrid relations: {hybrid_relations}")
        print(f"  Common relations: {len(common_relations)} ({relation_overlap:.1f}% overlap)")

        # Chiến lược retrieve chunks
        if hybrid_data:
            beam_chunks = hybrid_data.get("beam_context_chunks", [])
            vector_chunks = hybrid_data.get("vector_context_chunks", [])
            print(f"")
            print(f"  Hybrid chunks: {len(beam_chunks) + len(vector_chunks)}")
            print(f"    - Beam chunks: {len(beam_chunks)}")
            print(f"    - Vector chunks: {len(vector_chunks)}")

    # === TONG KET ===
    print(f"\n{'='*80}")
    print("TỔNG KẾT")
    print(f"{'='*80}")

    beam_total_time = sum(r["time"] for r in results["beam"])
    hybrid_total_time = sum(r["time"] for r in results["hybrid"])

    beam_total_entities = sum(r["entities"] for r in results["beam"])
    hybrid_total_entities = sum(r["entities"] for r in results["hybrid"])

    beam_total_relations = sum(r["relations"] for r in results["beam"])
    hybrid_total_relations = sum(r["relations"] for r in results["hybrid"])

    # Tính overlap tổng
    total_common_entities = 0
    total_all_entities = set()
    for beam, hybrid in zip(results["beam"], results["hybrid"]):
        total_common_entities += len(beam["entity_names"] & hybrid["entity_names"])
        total_all_entities |= beam["entity_names"] | hybrid["entity_names"]

    total_entity_overlap = total_common_entities / len(total_all_entities) * 100 if total_all_entities else 0

    total_common_relations = 0
    total_all_relations = set()
    for beam, hybrid in zip(results["beam"], results["hybrid"]):
        total_common_relations += len(beam["relation_pairs"] & hybrid["relation_pairs"])
        total_all_relations |= beam["relation_pairs"] | hybrid["relation_pairs"]

    total_relation_overlap = total_common_relations / len(total_all_relations) * 100 if total_all_relations else 0

    print(f"\n{'Metric':<30} {'Beam':<15} {'Hybrid':<15} {'Difference'}")
    print("-" * 75)
    print(f"{'Total Time':<30} {beam_total_time:.3f}s{'':<8} {hybrid_total_time:.3f}s{'':<8} {(hybrid_total_time - beam_total_time):+.3f}s")
    print(f"{'Total Entities':<30} {beam_total_entities:<15} {hybrid_total_entities:<15} {hybrid_total_entities - beam_total_entities:+d}")
    print(f"{'Total Relations':<30} {beam_total_relations:<15} {hybrid_total_relations:<15} {hybrid_total_relations - beam_total_relations:+d}")
    print(f"{'Entity Overlap':<30} {'':<15} {'':<15} {total_entity_overlap:.1f}%")
    print(f"{'Relation Overlap':<30} {'':<15} {'':<15} {total_relation_overlap:.1f}%")

    print(f"\n{'Câu':<5} {'Beam Time':<12} {'Hybrid Time':<12} {'Beam E':<10} {'Hybrid E':<10} {'Overlap %':<10}")
    print("-" * 60)
    for i, (beam, hybrid) in enumerate(zip(results["beam"], results["hybrid"]), 1):
        common_e = len(beam["entity_names"] & hybrid["entity_names"])
        all_e = len(beam["entity_names"] | hybrid["entity_names"])
        overlap = common_e / all_e * 100 if all_e else 0
        print(f"{i:<5} {beam['time']:<12.3f} {hybrid['time']:<12.3f} {beam['entities']:<10} {hybrid['entities']:<10} {overlap:<10.1f}")

    print(f"\n{'='*40}")
    print(f"Speedup (Beam vs Hybrid): {hybrid_total_time / beam_total_time:.2f}x")
    print(f"{'='*40}")

    # Chi tiết overlap
    print(f"\n{'='*80}")
    print("CHI TIET OVERLAP THEO CAU HOI")
    print(f"{'='*80}")
    for i, (beam, hybrid) in enumerate(zip(results["beam"], results["hybrid"]), 1):
        common_e = beam["entity_names"] & hybrid["entity_names"]
        only_beam_e = beam["entity_names"] - hybrid["entity_names"]
        only_hybrid_e = hybrid["entity_names"] - beam["entity_names"]

        common_r = beam["relation_pairs"] & hybrid["relation_pairs"]
        only_beam_r = beam["relation_pairs"] - hybrid["relation_pairs"]
        only_hybrid_r = hybrid["relation_pairs"] - beam["relation_pairs"]

        print(f"\nCâu {i}: {results['beam'][i-1]['question'][:50]}...")
        print(f"  Entities: Beam={beam['entities']}, Hybrid={hybrid['entities']}")
        print(f"    - Chung: {len(common_e)}")
        print(f"    - Chi Beam: {len(only_beam_e)}")
        print(f"    - Chi Hybrid: {len(only_hybrid_e)}")
        print(f"  Relations: Beam={beam['relations']}, Hybrid={hybrid['relations']}")
        print(f"    - Chung: {len(common_r)}")
        print(f"    - Chi Beam: {len(only_beam_r)}")
        print(f"    - Chi Hybrid: {len(only_hybrid_r)}")

    await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
