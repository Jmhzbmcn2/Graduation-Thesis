"""
Visualize Knowledge Graph từ LightRAG
Tạo file HTML interactive để xem đồ thị
"""

import pipmaster as pm

# Cài đặt thư viện cần thiết
if not pm.is_installed("pyvis"):
    pm.install("pyvis")
if not pm.is_installed("networkx"):
    pm.install("networkx")

import networkx as nx
from pyvis.network import Network
import random
import os

# Configuration
WORKING_DIR = "./medical_rag"  # Thư mục chứa dữ liệu
GRAPH_FILE = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
OUTPUT_HTML = os.path.join(WORKING_DIR, "knowledge_graph.html")

def visualize_graph():
    # Kiểm tra file tồn tại
    if not os.path.exists(GRAPH_FILE):
        print(f"❌ Không tìm thấy file: {GRAPH_FILE}")
        print("   Hãy chạy run_acepron_megallm.py trước để tạo Knowledge Graph!")
        return
    
    print(f"📊 Loading graph from: {GRAPH_FILE}")
    
    # Load GraphML file
    G = nx.read_graphml(GRAPH_FILE)
    
    print(f"   ✅ Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Tạo Pyvis network
    net = Network(
        height="100vh", 
        width="100%",
        bgcolor="#222222",
        font_color="white",
        notebook=False,
        directed=True
    )
    
    # Cấu hình physics để đồ thị đẹp hơn
    net.set_options("""
    {
        "nodes": {
            "font": {"size": 14, "color": "white"},
            "scaling": {"min": 10, "max": 30}
        },
        "edges": {
            "color": {"inherit": true},
            "smooth": {"type": "continuous"}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        }
    }
    """)
    
    # Màu theo loại entity (nếu có)
    entity_colors = {
        "person": "#FF6B6B",
        "location": "#4ECDC4",
        "organization": "#45B7D1",
        "event": "#96CEB4",
        "concept": "#FFEAA7",
        "method": "#DDA0DD",
        "drug": "#FF7F50",
        "disease": "#87CEEB",
        "symptom": "#98D8C8",
        "dosage": "#F7DC6F",
        "default": "#BB86FC"
    }
    
    # Convert NetworkX graph to Pyvis
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        
        # Lấy entity type nếu có
        entity_type = node_data.get("entity_type", "default").lower()
        color = entity_colors.get(entity_type, entity_colors["default"])
        
        # Tạo tooltip với description
        description = node_data.get("description", "No description")
        title = f"<b>{node_id}</b><br>Type: {entity_type}<br><br>{description[:500]}..."
        
        # Thêm node
        net.add_node(
            node_id,
            label=node_id[:30] + "..." if len(node_id) > 30 else node_id,
            title=title,
            color=color,
            size=20
        )
    
    # Thêm edges với labels
    for source, target, edge_data in G.edges(data=True):
        description = edge_data.get("description", "")
        keywords = edge_data.get("keywords", "relates to")
        weight = float(edge_data.get("weight", 1.0))
        
        # Lấy keyword đầu tiên làm label (ngắn gọn)
        label = keywords.split(",")[0].strip() if keywords else "relates to"
        
        title = f"<b>{source} → {target}</b><br><br><b>Keywords:</b> {keywords}<br><br><b>Description:</b><br>{description[:500]}..."
        
        net.add_edge(
            source, 
            target, 
            title=title,
            label=label,  # Hiển thị label trên edge
            font={"size": 10, "color": "#AAAAAA", "align": "middle"},
            width=max(1, weight * 2),
            arrows={"to": {"enabled": True, "scaleFactor": 0.5}}
        )
    
    # Lưu file HTML
    net.save_graph(OUTPUT_HTML)
    
    print(f"\n✅ Knowledge Graph đã được lưu tại: {OUTPUT_HTML}")
    print(f"   Mở file này trong trình duyệt để xem đồ thị!")
    
    # Tự động mở trong trình duyệt
    abs_path = os.path.abspath(OUTPUT_HTML)
    print(f"\n🌐 Đang mở trong trình duyệt...")
    os.startfile(abs_path)


def print_graph_stats():
    """In thống kê về Knowledge Graph"""
    if not os.path.exists(GRAPH_FILE):
        return
    
    G = nx.read_graphml(GRAPH_FILE)
    
    print("\n" + "=" * 50)
    print("📊 KNOWLEDGE GRAPH STATISTICS")
    print("=" * 50)
    print(f"   Total Nodes (Entities): {G.number_of_nodes()}")
    print(f"   Total Edges (Relations): {G.number_of_edges()}")
    
    # Đếm theo entity type
    entity_types = {}
    for node_id in G.nodes():
        entity_type = G.nodes[node_id].get("entity_type", "unknown")
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    print("\n   Entity Types:")
    for etype, count in sorted(entity_types.items(), key=lambda x: -x[1]):
        print(f"      - {etype}: {count}")
    
    # Top entities by degree
    print("\n   Top 10 Entities (by connections):")
    degrees = sorted(G.degree(), key=lambda x: -x[1])[:10]
    for node, degree in degrees:
        print(f"      - {node}: {degree} connections")


if __name__ == "__main__":
    print_graph_stats()
    print()
    visualize_graph()
