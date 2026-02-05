"""
Compare LightRAG Query Modes: naive vs mix
Sends the same question to both modes and displays results side-by-side
"""
import requests
import json

# Configuration
LIGHTRAG_URL = "http://localhost:9621"
# LIGHTRAG_URL = "http://localhost:9621"  # Change if different

def query_lightrag(question: str, mode: str) -> dict:
    """Query LightRAG with specified mode"""
    url = f"{LIGHTRAG_URL}/query"
    payload = {
        "query": question,
        "mode": mode,
        "stream": False,
        "only_need_context": False,
        "top_k": 10,           # KG entities/relations retrieval
        "chunk_top_k": 10      # Text chunks retrieval
    }
    
    response = requests.post(url, json=payload)
    return response.json()


def compare_modes(question: str):
    """Compare naive and mix modes for the same question"""
    print("=" * 80)
    print(f"📝 QUESTION: {question}")
    print("=" * 80)
    
    modes = ["naive", "mix"]
    results = {}
    
    for mode in modes:
        print(f"\n🔄 Querying with mode: {mode}...")
        try:
            result = query_lightrag(question, mode)
            results[mode] = result.get("response", str(result))
        except Exception as e:
            results[mode] = f"Error: {e}"
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 COMPARISON RESULTS")
    print("=" * 80)
    
    for mode in modes:
        print(f"\n{'─' * 40}")
        print(f"🔹 MODE: {mode.upper()}")
        print(f"{'─' * 40}")
        print(results[mode])
        print()
    
    # Save to file for easier comparison
    output_file = "comparison_result.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Question: {question}\n")
        f.write("=" * 80 + "\n\n")
        for mode in modes:
            f.write(f"=== MODE: {mode.upper()} ===\n")
            f.write(results[mode] + "\n\n")
    
    print(f"📁 Results saved to: {output_file}")


if __name__ == "__main__":
    # Interactive mode
    print("🩺 LightRAG Mode Comparison Tool")
    print("=" * 40)
    
    while True:
        question = input("\n❓ Enter your question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if question:
            compare_modes(question)
        else:
            print("⚠️ Please enter a valid question.")
