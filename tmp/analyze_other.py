"""Analyze 'other' type entities to understand what they are."""
import xml.etree.ElementTree as ET
from collections import Counter

tree = ET.parse(r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\medical_rag_ollama\graph_chunk_entity_relation.graphml")
root = tree.getroot()

# Find namespace
ns_uri = ""
for elem in root.iter():
    if elem.tag.startswith('{'):
        ns_uri = elem.tag.split('}')[0] + '}'
        break

# Find key IDs
keys = {}
for key in root.iter(f'{ns_uri}key'):
    attr_name = key.get('attr.name')
    if attr_name in ('entity_type', 'entity_id', 'description'):
        keys[attr_name] = key.get('id')

print(f"Keys: {keys}")

# Collect 'other' entities
other_entities = []
for node in root.iter(f'{ns_uri}node'):
    node_data = {}
    for data in node:
        key_id = data.get('key')
        for attr_name, mapped_key in keys.items():
            if key_id == mapped_key:
                node_data[attr_name] = data.text
    
    if node_data.get('entity_type', '').lower() == 'other':
        other_entities.append({
            'name': node_data.get('entity_id', node.get('id', 'N/A')),
            'desc': (node_data.get('description', '')[:120] + '...') if len(node_data.get('description', '')) > 120 else node_data.get('description', ''),
        })

print(f"\nTotal 'other' entities: {len(other_entities)}")
print(f"\nSample 'other' entities (first 50):")
print("=" * 90)

for i, e in enumerate(other_entities[:50]):
    name = e['name'][:35]
    desc = e['desc'][:80]
    print(f"  {i+1:3d}. {name:<37} | {desc}")

# Try to categorize what 'other' entities should be
print(f"\n{'=' * 90}")
print("Phân tích: Các entity 'other' có thể thuộc category nào?")
print("=" * 90)

# Simple keyword matching to suggest better types
suggestions = Counter()
medical_keywords = {
    'Drug/Treatment': ['thuốc', 'liều', 'mg', 'viên', 'uống', 'tiêm', 'điều trị', 'liệu pháp', 'phẫu thuật'],
    'Disease/Condition': ['bệnh', 'hội chứng', 'viêm', 'ung thư', 'nhiễm', 'suy', 'rối loạn'],
    'Anatomy': ['gan', 'thận', 'tim', 'phổi', 'não', 'mạch', 'xương', 'cơ', 'da', 'mắt', 'tai'],
    'Symptom/Phenotype': ['đau', 'sốt', 'buồn nôn', 'mệt', 'triệu chứng', 'sưng', 'ngứa'],
    'Lab/Diagnostic': ['xét nghiệm', 'chẩn đoán', 'siêu âm', 'chụp', 'máu', 'nước tiểu'],
    'Person/Role': ['bác sĩ', 'bệnh nhân', 'người bệnh', 'trẻ em', 'phụ nữ', 'thai'],
    'Guideline/Instruction': ['hướng dẫn', 'lưu ý', 'cách dùng', 'bảo quản', 'chống chỉ định'],
}

for e in other_entities:
    name_lower = e['name'].lower()
    desc_lower = e['desc'].lower()
    text = name_lower + " " + desc_lower
    
    matched = False
    for category, keywords in medical_keywords.items():
        if any(kw in text for kw in keywords):
            suggestions[category] += 1
            matched = True
            break
    if not matched:
        suggestions['Truly Other'] += 1

print(f"\nSuggested re-classification of 'other' entities:")
for cat, count in suggestions.most_common():
    pct = count / len(other_entities) * 100
    print(f"  {cat:<25} {count:4d} ({pct:.1f}%)")
