"""
Phân tích phân bố Entity Types từ Knowledge Graph (medical_rag_v2).
Đọc từ file graphml và vẽ đồ thị bar chart.
"""

import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ==================== PARSE GRAPHML ====================
_ROOT = os.path.dirname(os.path.abspath(__file__))
GRAPHML_PATH = os.path.join(
    _ROOT,
    "medical_rag",
    "medical_rag_v2",
    "graph_chunk_entity_relation.graphml",
)
OUTPUT_PATH = "entity_type_distribution.png"

tree = ET.parse(GRAPHML_PATH)
root = tree.getroot()
ns = '{http://graphml.graphdrawing.org/xmlns}'

nodes = root.findall(f'.//{ns}node')
print(f"📊 Tổng số entity nodes: {len(nodes)}")

# Extract entity types (key d1)
entity_types = []
for node in nodes:
    for data in node.findall(f'{ns}data'):
        if data.attrib.get('key') == 'd1' and data.text:
            entity_types.append(data.text.strip().lower())

print(f"📊 Tổng số entity có type: {len(entity_types)}")

# Count types
type_counts = Counter(entity_types)
sorted_types = type_counts.most_common()

print(f"\n📋 Phân bố Entity Types ({len(sorted_types)} loại):")
print("-" * 50)
for t, c in sorted_types:
    pct = c / len(entity_types) * 100
    print(f"  {t:30s} : {c:5d}  ({pct:.1f}%)")
print("-" * 50)
print(f"  {'TỔNG':30s} : {len(entity_types):5d}")

# ==================== VISUALIZATION ====================
labels = [t for t, _ in sorted_types]
counts = [c for _, c in sorted_types]

fig, ax = plt.subplots(figsize=(14, 7))

# Color palette
colors = plt.cm.viridis([i / len(labels) for i in range(len(labels))])

bars = ax.barh(range(len(labels)), counts, color=colors, edgecolor='white', linewidth=0.5)

# Labels
ax.set_yticks(range(len(labels)))
ax.set_yticklabels([f"{l}  ({c})" for l, c in zip(labels, counts)], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Count', fontsize=12, fontweight='bold')
ax.set_title(f'Entity Type Distribution in Medical KG\n(Total: {len(entity_types)} entities, {len(sorted_types)} types)',
             fontsize=14, fontweight='bold', pad=15)

# Percentage labels on bars
for i, (bar, count) in enumerate(zip(bars, counts)):
    pct = count / len(entity_types) * 100
    ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%', va='center', fontsize=9, color='#333')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f"\n✅ Chart saved to: {OUTPUT_PATH}")
plt.show()

