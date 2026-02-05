"""
Script to run test cases through LightRAG with naive and mix modes
Reads questions from test_case.xlsx and saves results to Excel file
"""
import pandas as pd
import requests
import time
from datetime import datetime

# Configuration
LIGHTRAG_URL = "http://localhost:9621"
INPUT_FILE = "test_case.xlsx"
OUTPUT_FILE = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\testcase\test_case.xlsx"

# Number of test cases to run (set to None to run all)
TEST_LIMIT = 130  # Start with 5 for testing


def query_lightrag(question: str, mode: str, retries: int = 3) -> str:
    """Query LightRAG with specified mode"""
    url = f"{LIGHTRAG_URL}/query"
    payload = {
        "query": question,
        "mode": mode,
        "stream": False,
        "only_need_context": False,
        "top_k": 10,
        "chunk_top_k": 10
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", str(result))
        except requests.exceptions.Timeout:
            print(f"  ⚠️ Timeout (attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    
    return "Error: Failed to get response"


def main():
    print("=" * 60)
    print("🩺 LightRAG Test Case Runner")
    print("=" * 60)
    
    # Read test cases
    print(f"\n📂 Reading test cases from: {INPUT_FILE}")
    try:
        df = pd.read_excel(INPUT_FILE)
        print(f"   Found {len(df)} test cases")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return
    
    # Apply test limit
    if TEST_LIMIT:
        df = df.head(TEST_LIMIT)
        print(f"   Running first {TEST_LIMIT} test cases...")
    
    # Prepare results columns
    naive_responses = []
    mix_responses = []
    
    # Process each question
    print(f"\n🔄 Starting queries...")
    total = len(df)
    
    for idx, row in df.iterrows():
        question = row.get('question', row.get('Question', ''))
        if not question or pd.isna(question):
            print(f"  [{idx+1}/{total}] ⚠️ Skipping empty question")
            naive_responses.append("")
            mix_responses.append("")
            continue
        
        print(f"\n[{idx+1}/{total}] 📝 Question: {question[:80]}...")
        
        # Query naive mode
        print(f"  🔹 Querying NAIVE mode...")
        start_time = time.time()
        naive_result = query_lightrag(question, "naive")
        naive_time = time.time() - start_time
        naive_responses.append(naive_result)
        print(f"     Done in {naive_time:.1f}s")
        
        # Query mix mode
        print(f"  🔹 Querying MIX mode...")
        start_time = time.time()
        mix_result = query_lightrag(question, "mix")
        mix_time = time.time() - start_time
        mix_responses.append(mix_result)
        print(f"     Done in {mix_time:.1f}s")
    
    # Add results to dataframe
    df['naive_response'] = naive_responses
    df['mix_response'] = mix_responses
    
    # Keep only required columns
    columns_to_keep = ['question', 'answer', 'context', 'naive_response', 'mix_response', 'title', 'article_url']
    df = df[[col for col in columns_to_keep if col in df.columns]]
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results_{timestamp}.xlsx"
    
    print(f"\n💾 Saving results to: {output_file}")
    df.to_excel(output_file, index=False)
    print(f"✅ Done! Results saved successfully.")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   Total questions processed: {total}")
    print(f"   Output file: {output_file}")


if __name__ == "__main__":
    main()
