"""
Đánh giá chất lượng Knowledge Graph - LightRAG Medical KG
=========================================================

Metrics:
1. fCorrectness (Schema Compliance Rate) - Schema Aware KG Completions
2. Average Clustering Coefficient - Newman (2010)

References:
- Seo et al. "Structural Quality Metrics to Evaluate Knowledge Graphs" (2022)
- Zaveri et al. "Quality Assessment for Linked Data: A Survey" (2016)
- Newman, M. "Networks: An Introduction", Oxford University Press (2010)
"""

import networkx as nx
import numpy as np
from collections import Counter
import json
import time
import os

# ============================================================
# CẤU HÌNH
# ============================================================

# 10 entity types khớp .env / LightRAG ENTITY_TYPES (phiên bản medical KG)
ENTITY_TYPES = [
    "Disease",
    "Symptom",
    "Drug",
    "Chemical compound",
    "Protein",
    "Anatomy",
    "Biological process",
    "Exposure",
    "Diagnostic test",
    "Treatment method",
]


def _normalize_entity_type_label(label: str) -> str:
    """LightRAG lưu entity_type trong GraphML: lowercase, bỏ khoảng trắng."""
    return "".join(label.lower().split())


# Tập type chuẩn dùng cho fCorrectness / ICR (đồng bộ với ENTITY_TYPES ở trên)
PREDEFINED_SCHEMA_TYPES = {_normalize_entity_type_label(t) for t in ENTITY_TYPES}

# Đường dẫn GraphML mặc định: KG trong medical_rag_v2. Ghi đè bằng KG_EVAL_GRAPHML_PATH nếu cần.
_GRAPHML_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "medical_rag",
    "medical_rag_v2",
    "graph_chunk_entity_relation.graphml",
)
GRAPHML_PATH = os.environ.get("KG_EVAL_GRAPHML_PATH", _GRAPHML_DEFAULT)

# Mapping: entity types tiếng Việt → type chuẩn (để phân tích hallucination)
VIETNAMESE_TYPE_MAPPING = {
    "khác": "other",
    "bệnh": "disease",
    "bệnhlý": "disease",
    "triệuchứng": "symptom",
    "thuốc": "drug",
    "hợpchấthóahọc": "chemicalcompound",
    "giảiphẫu": "anatomy",
    "quátrìnhsinhhọc": "biologicalprocess",
    "phơinhiễm": "exposure",
    "xétnghiệmchẩnđoán": "diagnostictest",
    "phươngphápđiềutrị": "treatmentmethod",
    "biểuhiệnlâmsàng": "symptom",
    "yếutốphơinhiễm": "exposure",
}


def load_graph(path: str) -> nx.Graph:
    """Load Knowledge Graph từ GraphML file."""
    print(f"Loading graph from: {path}")
    start = time.time()
    G = nx.read_graphml(path)
    elapsed = time.time() - start
    print(f"  Loaded in {elapsed:.1f}s — {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges\n")
    return G


def get_entity_type(G: nx.Graph, node: str) -> str:
    """Lấy entity_type của 1 node, trả về lowercase."""
    return G.nodes[node].get("entity_type", "MISSING").lower().strip()


# ============================================================
# METRIC 1: fCorrectness (Schema Compliance Rate)
# ============================================================

def evaluate_f_correctness(G: nx.Graph) -> dict:
    """
    fCorrectness = # schema-correct triples / # total triples

    Một triple (edge) được coi là "schema-correct" khi:
    - CẢ HAI entities (source + target) có entity_type thuộc 10 predefined types
    - Cả 2 entity names không rỗng và hợp lệ (>= 2 ký tự)

    Reference: "Schema Aware Knowledge Graph Completions" methodology
    Adapted: Seo et al. 2022 (ICR concept) + Zaveri et al. 2016 (Accuracy dimension)
    """
    print("=" * 70)
    print("  METRIC 1: fCorrectness (Schema Compliance Rate)")
    print("=" * 70)

    total_triples = G.number_of_edges()
    schema_correct = 0
    both_valid_type = 0
    one_invalid = 0
    both_invalid = 0

    # Chi tiết vi phạm
    violation_reasons = Counter()

    for src, tgt, data in G.edges(data=True):
        src_type = get_entity_type(G, src)
        tgt_type = get_entity_type(G, tgt)

        src_valid = src_type in PREDEFINED_SCHEMA_TYPES
        tgt_valid = tgt_type in PREDEFINED_SCHEMA_TYPES

        # Kiểm tra tên hợp lệ
        src_name_valid = len(src.strip()) >= 2
        tgt_name_valid = len(tgt.strip()) >= 2

        if src_valid and tgt_valid and src_name_valid and tgt_name_valid:
            schema_correct += 1
            both_valid_type += 1
        elif src_valid and tgt_valid:
            # Types đúng nhưng tên không hợp lệ
            violation_reasons["invalid_name"] += 1
        elif not src_valid and not tgt_valid:
            both_invalid += 1
            violation_reasons[f"both_invalid({src_type},{tgt_type})"] += 1
        else:
            one_invalid += 1
            invalid_type = src_type if not src_valid else tgt_type
            violation_reasons[f"one_invalid({invalid_type})"] += 1

    f_correctness = schema_correct / total_triples if total_triples > 0 else 0

    # --- Entity-level analysis ---
    total_entities = G.number_of_nodes()
    entity_types = Counter(get_entity_type(G, n) for n in G.nodes())

    # Phân loại entities
    correct_type_count = sum(
        c for t, c in entity_types.items() if t in PREDEFINED_SCHEMA_TYPES
    )
    other_count = entity_types.get("other", 0)
    unknown_count = entity_types.get("unknown", 0)
    vn_hallucination_count = sum(
        entity_types.get(t, 0) for t in VIETNAMESE_TYPE_MAPPING
    )
    # Mọi entity có type không thuộc 10 loại ENTITY_TYPES (không cần danh sách tay)
    non_schema_count = sum(
        c for t, c in entity_types.items() if t not in PREDEFINED_SCHEMA_TYPES
    )
    combined_type_count = sum(
        c for t, c in entity_types.items() if "," in t
    )

    # ICR (Instantiated Class Ratio)
    used_predefined = sum(
        1 for t in PREDEFINED_SCHEMA_TYPES if entity_types.get(t, 0) > 0
    )
    icr = used_predefined / len(PREDEFINED_SCHEMA_TYPES)

    # --- Output ---
    print(f"\n{'─' * 50}")
    print(f"  TRIPLE-LEVEL ANALYSIS")
    print(f"{'─' * 50}")
    print(f"  Total triples (edges):         {total_triples:>10,}")
    print(f"  Schema-correct triples:        {schema_correct:>10,}")
    print(f"  One entity invalid type:       {one_invalid:>10,}")
    print(f"  Both entities invalid type:    {both_invalid:>10,}")
    print()
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  fCorrectness = {f_correctness:.4f} ({f_correctness*100:.1f}%)             │")
    print(f"  │  ({schema_correct:,} / {total_triples:,} triples)       │")
    print(f"  └─────────────────────────────────────────────┘")

    print(f"\n{'─' * 50}")
    print(f"  ENTITY-LEVEL ANALYSIS")
    print(f"{'─' * 50}")
    print(f"  Total entities (nodes):        {total_entities:>10,}")
    print(f"  Đúng 10 predefined types:      {correct_type_count:>10,}  ({correct_type_count/total_entities*100:.1f}%)")
    print(f"  'other' (không phân loại được): {other_count:>10,}  ({other_count/total_entities*100:.1f}%)")
    print(f"  'UNKNOWN':                     {unknown_count:>10,}  ({unknown_count/total_entities*100:.1f}%)")
    print(f"  Vietnamese hallucination:      {vn_hallucination_count:>10,}  ({vn_hallucination_count/total_entities*100:.4f}%)")
    print(f"  Ngoài 10 loại schema (tổng):   {non_schema_count:>10,}  ({non_schema_count/total_entities*100:.1f}%)")
    print(f"    (gồm other, unknown, type do LLM tự tạo — không dùng whitelist tay)")
    print(f"  Combined types (format error): {combined_type_count:>10,}  ({combined_type_count/total_entities*100:.4f}%)")

    print(f"\n{'─' * 50}")
    print(f"  ICR (Instantiated Class Ratio)")
    print(f"  Seo et al. 2022")
    print(f"{'─' * 50}")
    print(f"  Predefined types:              {len(PREDEFINED_SCHEMA_TYPES)}")
    print(f"  Used predefined types:         {used_predefined}")
    print(f"  ICR = {used_predefined}/{len(PREDEFINED_SCHEMA_TYPES)} = {icr:.2f} ({icr*100:.0f}%)")

    print(f"\n{'─' * 50}")
    print(f"  ENTITY TYPE DISTRIBUTION")
    print(f"{'─' * 50}")
    print(f"  {'Type':<35} {'Count':>8}  {'%':>7}  {'Status'}")
    print(f"  {'─'*35} {'─'*8}  {'─'*7}  {'─'*15}")
    for t, c in entity_types.most_common():
        pct = c / total_entities * 100
        if t in PREDEFINED_SCHEMA_TYPES:
            status = "✅ Schema"
        elif t == "other":
            status = "⚠️  Unclassified"
        elif t == "unknown":
            status = "⚠️  Unknown"
        elif t in VIETNAMESE_TYPE_MAPPING:
            status = f"❌ VN→{VIETNAMESE_TYPE_MAPPING[t]}"
        elif "," in t:
            status = "❌ Combined"
        else:
            status = "❌ Ngoài schema"
        print(f"  {t:<35} {c:>8,}  {pct:>6.1f}%  {status}")

    # Top violation reasons
    print(f"\n{'─' * 50}")
    print(f"  TOP SCHEMA VIOLATIONS (at triple level)")
    print(f"{'─' * 50}")
    for reason, count in violation_reasons.most_common(10):
        print(f"  {reason:<45} {count:>8,}")

    return {
        "total_triples": total_triples,
        "schema_correct_triples": schema_correct,
        "f_correctness": round(f_correctness, 4),
        "total_entities": total_entities,
        "correct_type_entities": correct_type_count,
        "entity_schema_compliance": round(correct_type_count / total_entities, 4),
        "icr": round(icr, 2),
        "other_rate": round(other_count / total_entities, 4),
        "vn_hallucination_count": vn_hallucination_count,
        "non_schema_entities": non_schema_count,
        "non_schema_rate": round(non_schema_count / total_entities, 4),
    }


# ============================================================
# METRIC 2: Average Clustering Coefficient
# ============================================================

def evaluate_clustering_coefficient(G: nx.Graph) -> dict:
    """
    Average Clustering Coefficient = (1/n) * Σ C_v

    Trong đó C_v = 2*e_v / (d_v * (d_v - 1))
    - d_v: bậc (degree) của node v
    - e_v: số cạnh thực sự giữa các láng giềng của v

    Reference: Newman, M. "Networks: An Introduction" (2010)
    """
    print("\n" + "=" * 70)
    print("  METRIC 2: Average Clustering Coefficient")
    print("  Newman (2010)")
    print("=" * 70)

    start = time.time()

    # Chuyển sang undirected graph nếu cần (clustering coefficient cần undirected)
    if G.is_directed():
        G_undirected = G.to_undirected()
    else:
        G_undirected = G

    # Tính average clustering coefficient
    print("\n  Computing average clustering coefficient...")
    print("  (Có thể mất vài phút cho đồ thị lớn...)")

    avg_cc = nx.average_clustering(G_undirected)
    elapsed = time.time() - start

    # Transitivity (global clustering coefficient — metric bổ sung)
    print("  Computing transitivity (global clustering coefficient)...")
    transitivity = nx.transitivity(G_undirected)

    # Phân tích clustering theo entity type
    print("  Computing clustering by entity type...")
    type_clustering = {}
    nodes_by_type = {}
    for n in G_undirected.nodes():
        t = get_entity_type(G_undirected, n)
        if t in PREDEFINED_SCHEMA_TYPES:
            if t not in nodes_by_type:
                nodes_by_type[t] = []
            nodes_by_type[t].append(n)

    for t, nodes in nodes_by_type.items():
        cc_values = [nx.clustering(G_undirected, n) for n in nodes]
        type_clustering[t] = {
            "count": len(nodes),
            "avg_cc": round(np.mean(cc_values), 4) if cc_values else 0,
            "median_cc": round(np.median(cc_values), 4) if cc_values else 0,
        }

    # --- Output ---
    print(f"\n{'─' * 50}")
    print(f"  CLUSTERING COEFFICIENT RESULTS")
    print(f"  (Computed in {elapsed:.1f}s)")
    print(f"{'─' * 50}")
    print()
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  Average Clustering Coefficient:            │")
    print(f"  │  ⟨C⟩ = {avg_cc:.4f}                              │")
    print(f"  │                                             │")
    print(f"  │  Global Clustering (Transitivity):          │")
    print(f"  │  C_global = {transitivity:.4f}                         │")
    print(f"  └─────────────────────────────────────────────┘")

    # Diễn giải
    print(f"\n{'─' * 50}")
    print(f"  INTERPRETATION")
    print(f"{'─' * 50}")
    if avg_cc > 0.3:
        interpretation = "CAO — KG có community structure rõ ràng, entities cùng chủ đề liên kết chặt"
    elif avg_cc > 0.1:
        interpretation = "TRUNG BÌNH — phù hợp với đa số KG thực tế, có clustering vừa phải"
    elif avg_cc > 0.01:
        interpretation = "THẤP — KG khá phân tán, ít nhóm chặt chẽ"
    else:
        interpretation = "RẤT THẤP — KG gần như không có community structure"

    print(f"  ⟨C⟩ = {avg_cc:.4f} → {interpretation}")
    print()
    print(f"  Tham chiếu KG lớn:")
    print(f"  • Wikidata:     ⟨C⟩ ≈ 0.11 – 0.17")
    print(f"  • DBpedia:      ⟨C⟩ ≈ 0.15 – 0.25")
    print(f"  • Social nets:  ⟨C⟩ ≈ 0.30 – 0.60")
    print(f"  • Random graph: ⟨C⟩ ≈ {1/G_undirected.number_of_nodes():.6f} (rất thấp)")

    # Clustering theo entity type
    print(f"\n{'─' * 50}")
    print(f"  CLUSTERING BY ENTITY TYPE")
    print(f"{'─' * 50}")
    print(f"  {'Type':<25} {'Count':>8}  {'Avg CC':>8}  {'Median CC':>10}")
    print(f"  {'─'*25} {'─'*8}  {'─'*8}  {'─'*10}")
    for t in sorted(type_clustering.keys(), key=lambda x: type_clustering[x]["avg_cc"], reverse=True):
        info = type_clustering[t]
        print(f"  {t:<25} {info['count']:>8,}  {info['avg_cc']:>8.4f}  {info['median_cc']:>10.4f}")

    return {
        "avg_clustering_coefficient": round(avg_cc, 4),
        "transitivity": round(transitivity, 4),
        "clustering_by_type": type_clustering,
        "interpretation": interpretation,
    }


# ============================================================
# BỔ SUNG: Graph Topology Metrics
# ============================================================

def evaluate_graph_topology(G: nx.Graph) -> dict:
    """
    Các chỉ số topology bổ trợ.
    Reference: Newman (2010), Zaveri et al. (2016)
    """
    print("\n" + "=" * 70)
    print("  SUPPLEMENTARY: Graph Topology Metrics")
    print("  Newman (2010), Zaveri et al. (2016)")
    print("=" * 70)

    if G.is_directed():
        G_und = G.to_undirected()
    else:
        G_und = G

    n_nodes = G_und.number_of_nodes()
    n_edges = G_und.number_of_edges()

    # Degree statistics
    degrees = [d for _, d in G_und.degree()]
    avg_degree = np.mean(degrees)
    median_degree = np.median(degrees)
    max_degree = max(degrees)

    # Density
    density = nx.density(G_und)

    # Connected components
    components = list(nx.connected_components(G_und))
    n_components = len(components)
    largest_cc = max(components, key=len)
    largest_cc_ratio = len(largest_cc) / n_nodes

    # Isolated nodes (degree = 0)
    isolated = sum(1 for d in degrees if d == 0)
    isolated_ratio = isolated / n_nodes

    # Top-10 hub entities
    degree_dict = dict(G_und.degree())
    top_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:15]

    # Entity type balance (Shannon Entropy)
    type_counts = Counter(get_entity_type(G_und, n) for n in G_und.nodes())
    total = sum(type_counts.values())
    probs = [c / total for c in type_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(len(type_counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # --- Output ---
    print(f"\n{'─' * 50}")
    print(f"  BASIC STATISTICS")
    print(f"{'─' * 50}")
    print(f"  Nodes (Entities):     {n_nodes:>12,}")
    print(f"  Edges (Relations):    {n_edges:>12,}")
    print(f"  Avg Degree:           {avg_degree:>12.2f}")
    print(f"  Median Degree:        {median_degree:>12.0f}")
    print(f"  Max Degree:           {max_degree:>12,}")
    print(f"  Graph Density:        {density:>12.6f}")

    print(f"\n{'─' * 50}")
    print(f"  CONNECTIVITY")
    print(f"{'─' * 50}")
    print(f"  Connected Components: {n_components:>12,}")
    print(f"  Largest Component:    {len(largest_cc):>12,}  ({largest_cc_ratio*100:.1f}%)")
    print(f"  Isolated Nodes:       {isolated:>12,}  ({isolated_ratio*100:.2f}%)")

    print(f"\n{'─' * 50}")
    print(f"  ENTITY TYPE BALANCE (Shannon Entropy)")
    print(f"  Seo et al. 2022 — adapted CI metric")
    print(f"{'─' * 50}")
    print(f"  Shannon Entropy:      {entropy:.4f}")
    print(f"  Max Entropy:          {max_entropy:.4f}  (if perfectly balanced)")
    print(f"  Normalized Entropy:   {normalized_entropy:.4f}  (1.0 = perfect balance)")

    print(f"\n{'─' * 50}")
    print(f"  TOP-15 HUB ENTITIES (highest degree)")
    print(f"{'─' * 50}")
    print(f"  {'#':<4} {'Entity':<40} {'Type':<20} {'Degree':>8}")
    print(f"  {'─'*4} {'─'*40} {'─'*20} {'─'*8}")
    for i, (node, deg) in enumerate(top_hubs, 1):
        etype = get_entity_type(G_und, node)
        name = node[:38] if len(node) > 38 else node
        print(f"  {i:<4} {name:<40} {etype:<20} {deg:>8,}")

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": round(avg_degree, 2),
        "median_degree": int(median_degree),
        "max_degree": max_degree,
        "density": round(density, 6),
        "n_components": n_components,
        "largest_component_ratio": round(largest_cc_ratio, 4),
        "isolated_nodes": isolated,
        "isolated_ratio": round(isolated_ratio, 4),
        "normalized_entropy": round(normalized_entropy, 4),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "█" * 70)
    print("  KNOWLEDGE GRAPH QUALITY EVALUATION")
    print("  LightRAG Medical KG (medical_rag_v2 schema)")
    print("█" * 70)
    print()

    # Load graph
    G = load_graph(GRAPHML_PATH)

    # Run evaluations
    results_correctness = evaluate_f_correctness(G)
    results_topology = evaluate_graph_topology(G)
    results_clustering = evaluate_clustering_coefficient(G)

    # Summary
    print("\n" + "█" * 70)
    print("  SUMMARY — KG QUALITY METRICS")
    print("█" * 70)
    print()
    print(f"  ┌───────────────────────────────────────────────────────┐")
    print(f"  │  METRIC                          │     VALUE         │")
    print(f"  ├───────────────────────────────────┼───────────────────┤")
    print(f"  │  fCorrectness (Schema Compliance) │  {results_correctness['f_correctness']:.4f} ({results_correctness['f_correctness']*100:.1f}%)    │")
    print(f"  │  ICR (Instantiated Class Ratio)   │  {results_correctness['icr']:.2f} ({results_correctness['icr']*100:.0f}%)       │")
    print(f"  │  Entity Schema Compliance         │  {results_correctness['entity_schema_compliance']:.4f} ({results_correctness['entity_schema_compliance']*100:.1f}%)    │")
    print(f"  │  Avg Clustering Coefficient ⟨C⟩   │  {results_clustering['avg_clustering_coefficient']:.4f}            │")
    print(f"  │  Transitivity (Global CC)         │  {results_clustering['transitivity']:.4f}            │")
    print(f"  │  Avg Degree                       │  {results_topology['avg_degree']:.2f}             │")
    print(f"  │  Largest Component Ratio          │  {results_topology['largest_component_ratio']:.4f} ({results_topology['largest_component_ratio']*100:.1f}%)    │")
    print(f"  │  Entity Type Balance (Entropy)    │  {results_topology['normalized_entropy']:.4f}            │")
    print(f"  └───────────────────────────────────┴───────────────────┘")

    # Save results to JSON
    all_results = {
        "f_correctness": results_correctness,
        "topology": results_topology,
        "clustering": results_clustering,
    }
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "kg_quality_evaluation_results.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
