"""
Script to extract the first question from test_case.xlsx
and create a RAGAS-compatible JSON dataset
"""
import pandas as pd
import json

INPUT_FILE = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\testcase\test_case.xlsx"
OUTPUT_FILE = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\testcase\ragas_test_1question.json"

# Read Excel
df = pd.read_excel(INPUT_FILE)
print(f"Total test cases: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Get first row
first = df.iloc[0]
print(f"\n--- First Question ---")
print(f"Question: {first.get('question', 'N/A')}")
print(f"Answer (ground_truth): {first.get('answer', 'N/A')[:100]}...")
print(f"Context: {str(first.get('context', 'N/A'))[:100]}...")

# Create RAGAS format
ragas_data = {
    "test_cases": [
        {
            "question": str(first.get('question', '')),
            "ground_truth": str(first.get('answer', '')),
            "project": "medical_rag_test"
        }
    ]
}

# Save JSON
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(ragas_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved to: {OUTPUT_FILE}")
print(f"\nJSON content:")
print(json.dumps(ragas_data, ensure_ascii=False, indent=2))
