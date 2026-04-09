"""Extract all unique entity types from the Knowledge Graph."""
import json
from collections import Counter

# 1. From graphml file
print("=" * 60)
print("  ENTITY TYPES IN KNOWLEDGE GRAPH")
print("=" * 60)

# Parse graphml for entity types
import xml.etree.ElementTree as ET

tree = ET.parse(r"/home/linhvd/Graduation-Thesis/medical_rag/medical_rag_ollama/graph_chunk_entity_relation.graphml")
root = tree.getroot()

ns = {'g': 'http://graphml.graphstruct.org/graphml'}
# Try to find namespace
for elem in root.iter():
    if elem.tag.startswith('{'):
        ns_uri = elem.tag.split('}')[0] + '}'
        ns = {'g': ns_uri.strip('{}')}
        break

# Find entity_type key definition
entity_type_key = None
for key in root.findall('.//{%s}key' % ns['g']):
    if key.get('attr.name') == 'entity_type':
        entity_type_key = key.get('id')
        break

if not entity_type_key:
    # Try without namespace
    for key in root.iter('key'):
        if key.get('attr.name') == 'entity_type':
            entity_type_key = key.get('id')
            break

print(f"\nEntity type key ID: {entity_type_key}")

# Count entity types from nodes
type_counter = Counter()
total_nodes = 0

for node in root.iter('{%s}node' % ns['g']):
    total_nodes += 1
    for data in node:
        if data.get('key') == entity_type_key:
            entity_type = data.text
            if entity_type:
                type_counter[entity_type] += 1

# Also try without namespace
if total_nodes == 0:
    for node in root.iter('node'):
        total_nodes += 1
        for data in node:
            if data.get('key') == entity_type_key:
                entity_type = data.text
                if entity_type:
                    type_counter[entity_type] += 1

print(f"Total nodes: {total_nodes}")
print(f"Unique entity types: {len(type_counter)}")

print(f"\n{'Entity Type':<40} {'Count':>8} {'%':>8}")
print("-" * 58)
for etype, count in type_counter.most_common():
    pct = count / total_nodes * 100
    print(f"  {etype:<38} {count:>8} {pct:>7.1f}%")

# 2. Also check .env for ENTITY_TYPES config
print(f"\n{'=' * 60}")
print("  ENTITY_TYPES CONFIG IN .env")
print("=" * 60)
try:
    with open(r"/home/linhvd/Graduation-Thesis/.env", "r", encoding="utf-8") as f:
        for line in f:
            if "ENTITY_TYPE" in line.upper():
                print(f"  {line.strip()}")
except:
    print("  Could not read .env")

print()
