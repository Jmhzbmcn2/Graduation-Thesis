"""
Phân tích chi tiết các entity có type = 'other' trong Knowledge Graph.
Ghi kết quả ra file text.
"""
import networkx as nx
import sys, io
from collections import Counter
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

GRAPHML_PATH = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\medical_rag_ollama\graph_chunk_entity_relation.graphml"
OUTPUT_FILE = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\tmp\other_analysis_result.txt"

print("Loading graph...")
G = nx.read_graphml(GRAPHML_PATH)
print(f"Loaded: {G.number_of_nodes()} nodes")

# Lấy tất cả node có type = 'other'
other_nodes = []
for node, data in G.nodes(data=True):
    etype = data.get("entity_type", "").lower().strip()
    if etype == "other":
        other_nodes.append({
            "name": node,
            "description": data.get("description", ""),
        })

print(f"Total 'other' nodes: {len(other_nodes)}")

# Phân loại theo các từ khóa y tế phổ biến
keyword_categories = {
    "Triệu chứng / Symptom": [
        "đau", "sốt", "ho", "mệt", "buồn nôn", "nôn", "chóng mặt",
        "nhức", "ngứa", "sưng", "viêm", "chảy máu", "khó thở",
        "tiêu chảy", "táo bón", "phù", "ban", "mẩn", "tê",
        "triệu chứng", "symptom", "pain", "fever", "cough",
        "nausea", "fatigue", "headache", "vomiting", "diarrhea",
        "swelling", "rash", "bleeding", "dyspnea", "edema",
    ],
    "Xét nghiệm / Chỉ số lâm sàng": [
        "xét nghiệm", "chỉ số", "nồng độ", "hàm lượng", "test",
        "glucose", "cholesterol", "hba1c", "creatinin", "alt", "ast",
        "bilirubin", "hemoglobin", "hematocrit", "platelet", "wbc",
        "crp", "esr", "bun", "gfr", "ldl", "hdl", "triglycerid",
        "albumin", "ferritin", "tsh", "psa",
        "siêu âm", "x-quang", "ct scan", "mri", "chụp", "điện tim",
        "sinh thiết", "nội soi", "xạ hình", "chẩn đoán",
        "xquang", "chụp cắt", "diagnostic", "imaging",
    ],
    "Phương pháp điều trị / Treatment": [
        "điều trị", "phẫu thuật", "liệu pháp", "trị liệu",
        "hóa trị", "xạ trị", "phục hồi", "ghép", "cấy",
        "treatment", "therapy", "surgery", "transplant",
        "chemotherapy", "radiation", "rehabilitation",
        "phương pháp", "can thiệp", "phác đồ",
    ],
    "Dược chất / Hợp chất hóa học": [
        "acid", "enzyme", "receptor", "hormone", "vitamin",
        "kháng thể", "kháng sinh", "insulin", "steroid",
        "inhibitor", "agonist", "antagonist",
        "gene", "gen ", "dna", "rna", "mrna",
        "cytokine", "interleukin", "interferon",
        "chất", "hợp chất", "hoạt chất", "dược chất",
    ],
    "Cơ quan / Bộ phận cơ thể": [
        "gan", "thận", "phổi", "tim", "não", "dạ dày", "ruột",
        "xương", "khớp", "da ", "mắt", "tai ", "mũi", "họng",
        "tử cung", "buồng trứng", "tuyến", "mạch máu", "động mạch",
        "tĩnh mạch", "liver", "kidney", "lung", "heart",
        "brain", "stomach", "intestine", "bone",
    ],
    "Đối tượng bệnh nhân / Population": [
        "bệnh nhân", "người bệnh", "trẻ em", "người cao tuổi",
        "phụ nữ mang thai", "thai nhi", "trẻ sơ sinh", "nam giới",
        "nữ giới", "người lớn", "thanh thiếu niên",
        "patient", "children", "elderly", "pregnant",
        "nhóm tuổi", "giới tính", "dân số", "cộng đồng",
    ],
    "Thực phẩm / Dinh dưỡng": [
        "thực phẩm", "dinh dưỡng", "chế độ ăn",
        "khoáng chất", "chất béo", "carbohydrate",
        "chất xơ", "calori", "food", "nutrition", "diet",
        "rau", "trái cây", "sữa", "thịt", "cá ",
    ],
    "Liều lượng / Đơn vị đo": [
        " mg", " ml", "mcg", " iu ", "mmol",
        "liều", "dose", "lần/ngày", "viên", "ống",
        "mg/kg", "mg/dl", "mmhg", "ng/ml",
    ],
}

lines = []
def log(text=""):
    lines.append(text)
    print(text)

log("=" * 70)
log("  PHÂN TÍCH CHI TIẾT ENTITY TYPE = 'OTHER'")
log(f"  Tổng: {len(other_nodes)} nodes / {G.number_of_nodes()} tổng")
log("=" * 70)

category_counts = Counter()
category_examples = {}
uncategorized = []

for node in other_nodes:
    name_lower = node["name"].lower()
    desc_lower = node["description"].lower() if node["description"] else ""
    combined = name_lower + " " + desc_lower

    matched = False
    for category, keywords in keyword_categories.items():
        for kw in keywords:
            if kw in combined:
                category_counts[category] += 1
                if category not in category_examples:
                    category_examples[category] = []
                if len(category_examples[category]) < 10:
                    category_examples[category].append(node["name"])
                matched = True
                break
        if matched:
            break

    if not matched:
        uncategorized.append(node["name"])

log(f"\n{'Category':<45} {'Count':>8}  {'%':>7}")
log(f"{'─'*45} {'─'*8}  {'─'*7}")

total_other = len(other_nodes)
for cat, count in category_counts.most_common():
    pct = count / total_other * 100
    log(f"  {cat:<43} {count:>8}  {pct:>6.1f}%")

uncategorized_count = len(uncategorized)
pct = uncategorized_count / total_other * 100
log(f"  {'(Không phân loại được)':<43} {uncategorized_count:>8}  {pct:>6.1f}%")

# In ví dụ cho từng category
log(f"\n{'=' * 70}")
log("📋 VÍ DỤ CÁC ENTITY TIÊU BIỂU TRONG MỖI NHÓM")
log(f"{'=' * 70}")
for cat, count in category_counts.most_common():
    examples = category_examples.get(cat, [])
    log(f"\n🔹 {cat} ({count} entities):")
    for ex in examples[:10]:
        log(f"     • {ex}")

# In sample các entity không phân loại được
log(f"\n🔹 Không phân loại được ({uncategorized_count} entities):")
random.seed(42)
samples = random.sample(uncategorized, min(50, len(uncategorized)))
for ex in samples:
    log(f"     • {ex}")

# Save to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n✅ Results saved to: {OUTPUT_FILE}")
