"""Analyze 'other' entities from graphml - robust parser."""
import json
from collections import Counter
import re

# Read graphml as text and parse with regex (more robust than XML parser for large/incomplete files)
with open(r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\medical_rag_ollama\graph_chunk_entity_relation.graphml", "r", encoding="utf-8") as f:
    content = f.read()

# Find all nodes with their data
# Pattern: <node id="...">...<data key="d0">entity_id</data>...<data key="d1">entity_type</data>...<data key="d2">description</data>...</node>
node_pattern = re.compile(r'<node id="([^"]*)">(.*?)</node>', re.DOTALL)
data_pattern = re.compile(r'<data key="(d\d+)">(.*?)</data>', re.DOTALL)

other_entities = []
all_types = Counter()

for node_match in node_pattern.finditer(content):
    node_content = node_match.group(2)
    data_fields = {}
    for data_match in data_pattern.finditer(node_content):
        data_fields[data_match.group(1)] = data_match.group(2).strip()
    
    etype = data_fields.get('d1', '').lower().strip()
    all_types[etype] += 1
    
    if etype == 'other':
        name = data_fields.get('d0', node_match.group(1))
        desc = data_fields.get('d2', '')[:120]
        other_entities.append({'name': name, 'desc': desc})

print(f"Total 'other' entities found: {len(other_entities)}")

# Keyword-based classification  
medical_keywords = {
    'Drug/Medication': ['thuốc', ' mg', 'viên', 'hoạt chất', 'biệt dược', 'dược', 'liều dùng', 'dạng bào chế'],
    'Treatment/Procedure': ['điều trị', 'liệu pháp', 'phẫu thuật', 'phương pháp', 'chữa', 'can thiệp', 'trị liệu', 'phòng ngừa', 'chăm sóc'],
    'Symptom/Sign': ['đau', 'sốt', 'buồn nôn', 'mệt', 'triệu chứng', 'sưng', 'ngứa', 'chóng mặt', 'ho ', 'nôn', 'khó thở'],
    'Anatomy/Body part': ['gan', 'thận', 'tim', 'phổi', 'não', 'mạch', 'xương', 'cơ ', 'da ', 'mắt', 'tai', 'dạ dày', 'ruột', 'máu', 'tế bào', 'cơ thể'],
    'Disease/Condition': ['bệnh', 'hội chứng', 'viêm', 'ung thư', 'nhiễm', 'suy', 'rối loạn', 'tình trạng'],
    'Person/Patient role': ['bác sĩ', 'bệnh nhân', 'người bệnh', 'trẻ em', 'phụ nữ', 'thai', 'người lớn', 'trẻ sơ sinh', 'nam giới', 'nữ giới', 'người dùng'],
    'Lab/Diagnostic test': ['xét nghiệm', 'chẩn đoán', 'siêu âm', 'chụp', 'nồng độ', 'chỉ số', 'kết quả', 'kiểm tra'],
    'Food/Nutrition': ['thực phẩm', 'vitamin', 'dinh dưỡng', 'chế độ ăn', 'thức ăn', 'rau', 'sữa', 'bổ sung'],
    'Guideline/Warning': ['hướng dẫn', 'lưu ý', 'cách dùng', 'bảo quản', 'chống chỉ định', 'cảnh báo', 'thận trọng', 'tương tác', 'tác dụng phụ'],
    'Medical device/Form': ['dung dịch', 'ống', 'bơm', 'kim', 'máy', 'thiết bị', 'băng', 'gạc'],
}

category_counts = Counter()
category_examples = {k: [] for k in list(medical_keywords.keys()) + ['Truly Other']}

for e in other_entities:
    text = (e['name'] + " " + e['desc']).lower()
    matched = False
    for category, keywords in medical_keywords.items():
        if any(kw in text for kw in keywords):
            category_counts[category] += 1
            if len(category_examples[category]) < 5:
                category_examples[category].append(e['name'][:30])
            matched = True
            break
    if not matched:
        category_counts['Truly Other'] += 1
        if len(category_examples['Truly Other']) < 10:
            category_examples['Truly Other'].append(e['name'][:30])

print(f"\n{'Category':<25} {'Count':>6} {'%':>7}")
print("-" * 80)
for cat, count in category_counts.most_common():
    pct = count / len(other_entities) * 100 if other_entities else 0
    examples = ", ".join(category_examples.get(cat, [])[:3])
    print(f"  {cat:<23} {count:>6} {pct:>6.1f}%  e.g. {examples}")

print(f"\n{'='*60}")
print("Sample 'Truly Other' entities:")
print("=" * 60)
truly_other_examples = category_examples.get('Truly Other', [])
for name in truly_other_examples:
    print(f"  - {name}")
