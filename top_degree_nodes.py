import networkx as nx

GRAPHML_PATH = r"medical_rag\medical_rag_v2\graph_chunk_entity_relation.graphml"

print("Loading graph...")
G = nx.read_graphml(GRAPHML_PATH)
print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Get degree for all nodes and sort (undirected total degree)
degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)

print(f"\n{'='*80}")
print(f"  TOP 30 NODES BY DEGREE (highest connectivity)")
print(f"{'='*80}")
print(f"{'Rank':<5} {'Node':<50} {'Degree':<8} {'Entity Type'}")
print(f"{'-'*80}")
for i, (node, deg) in enumerate(degrees[:30], 1):
    etype = G.nodes[node].get("entity_type", "N/A")
    label = node[:48] if len(node) > 48 else node
    print(f"{i:<5} {label:<50} {deg:<8} {etype}")

print(f"\n{'='*80}")
print(f"  DEGREE STATISTICS")
print(f"{'='*80}")
all_degrees = [d for _, d in G.degree()]
print(f"  Max degree    : {max(all_degrees)}")
print(f"  Min degree    : {min(all_degrees)}")
print(f"  Avg degree    : {sum(all_degrees)/len(all_degrees):.2f}")

# Count nodes with degree >= 100
high = sum(1 for d in all_degrees if d >= 100)
print(f"  Nodes deg>=100: {high}")
high50 = sum(1 for d in all_degrees if d >= 50)
print(f"  Nodes deg>=50 : {high50}")
