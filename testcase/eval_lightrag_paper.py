"""
RAG Evaluation Script - Group 1: LightRAG Paper Metrics
Evaluates naive_response vs mix_response using LLM-as-Judge (Gemini)
on 3 criteria from the LightRAG paper:
- Comprehensiveness
- Diversity  
- Empowerment

Reference: "LightRAG: Simple and Fast Retrieval-Augmented Generation" (Guo et al., 2024)
Prompt taken from: reproduce/batch_eval.py (original LightRAG evaluation)
"""
import os
import sys
import random
import pandas as pd
import json
import time
import re
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)

# ============================================================
# Configuration
# ============================================================
GEMINI_API_KEY = os.getenv("LLM_BINDING_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"  # Judge model
TEST_LIMIT = 130  # All questions

INPUT_FILE = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\testcase\test_results_20260204_014454.xlsx"

if not GEMINI_API_KEY:
    print("❌ LLM_BINDING_API_KEY not found in .env file!")
    sys.exit(1)

# ============================================================
# Prompt from LightRAG paper (reproduce/batch_eval.py) - EXACT COPY
# No labels like "Naive Mode" or "Mix Mode" to avoid bias
# ============================================================
SYS_PROMPT = """
---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
"""

EVAL_PROMPT_TEMPLATE = """
You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Diversity": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }}
}}"""


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response"""
    text = text.strip()
    
    # Try to find JSON block in markdown
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
    
    # Clean control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"    [DEBUG] JSON parse error: {e}")
            return None


def evaluate_pair(model, question: str, response_a: str, response_b: str, idx: int) -> dict:
    """Evaluate two responses using Gemini - no labels, just Answer 1 / Answer 2"""
    
    # Truncate very long responses to avoid token limits
    max_len = 3000
    if len(response_a) > max_len:
        response_a = response_a[:max_len] + "..."
    if len(response_b) > max_len:
        response_b = response_b[:max_len] + "..."
    
    prompt = EVAL_PROMPT_TEMPLATE.format(
        query=question,
        answer1=response_a,
        answer2=response_b
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                SYS_PROMPT + "\n\n" + prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=1000,
                )
            )
            
            result = extract_json(response.text)
            if result:
                return result
            else:
                print(f"    ⚠️ Q{idx}: Failed to parse JSON (attempt {attempt+1}), retrying...")
                time.sleep(2)
                
        except Exception as e:
            error_msg = str(e)[:120]
            print(f"    ❌ Q{idx}: Error (attempt {attempt+1}): {error_msg}")
            if "429" in error_msg or "quota" in error_msg.lower():
                wait_time = 30 * (attempt + 1)
                print(f"    ⏳ Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
    
    return None


def map_winner(raw_winner: str, swapped: bool) -> str:
    """
    Map the LLM's 'Answer 1'/'Answer 2' back to 'Naive'/'Mix',
    accounting for whether the positions were swapped.
    
    If swapped=False: Answer 1 = Naive, Answer 2 = Mix
    If swapped=True:  Answer 1 = Mix,   Answer 2 = Naive
    """
    is_answer1 = "1" in raw_winner
    is_answer2 = "2" in raw_winner
    
    if not swapped:
        if is_answer1:
            return "Naive"
        elif is_answer2:
            return "Mix"
    else:
        if is_answer1:
            return "Mix"
        elif is_answer2:
            return "Naive"
    
    return "Unclear"


def run_evaluation():
    """Run LightRAG paper evaluation on test results"""
    
    print("=" * 70)
    print("📊 RAG Evaluation - Group 1: LightRAG Paper Metrics")
    print("   (Comprehensiveness, Diversity, Empowerment)")
    print("   LLM Judge: Gemini 2.0 Flash")
    print("   Anti-bias: Random position swap + No mode labels")
    print("=" * 70)
    
    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    # Read data
    print(f"\n📂 Reading: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)
    df = df.head(TEST_LIMIT)
    print(f"   Evaluating {len(df)} questions")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Evaluate each question
    results = []
    start_time = time.time()
    swap_count = 0
    
    for idx, row in df.iterrows():
        question = str(row['question'])
        naive = str(row['naive_response'])
        mix = str(row['mix_response'])
        
        # Random swap to eliminate position bias
        swapped = random.random() < 0.5
        if swapped:
            answer1, answer2 = mix, naive
            swap_count += 1
        else:
            answer1, answer2 = naive, mix
        
        print(f"\n🔍 [{idx+1}/{len(df)}] {question[:60]}... {'[SWAPPED]' if swapped else ''}")
        
        eval_result = evaluate_pair(model, question, answer1, answer2, idx+1)
        
        if eval_result:
            # Map raw winners back to Naive/Mix (accounting for swap)
            comp_raw = eval_result.get("Comprehensiveness", {}).get("Winner", "N/A")
            div_raw = eval_result.get("Diversity", {}).get("Winner", "N/A")
            emp_raw = eval_result.get("Empowerment", {}).get("Winner", "N/A")
            overall_raw = eval_result.get("Overall Winner", {}).get("Winner", "N/A")
            
            comp_winner = map_winner(comp_raw, swapped)
            div_winner = map_winner(div_raw, swapped)
            emp_winner = map_winner(emp_raw, swapped)
            overall_winner = map_winner(overall_raw, swapped)
            
            results.append({
                "question": question,
                "swapped": swapped,
                "comp_raw": comp_raw, "comp_winner": comp_winner,
                "comp_explanation": eval_result.get("Comprehensiveness", {}).get("Explanation", ""),
                "div_raw": div_raw, "div_winner": div_winner,
                "div_explanation": eval_result.get("Diversity", {}).get("Explanation", ""),
                "emp_raw": emp_raw, "emp_winner": emp_winner,
                "emp_explanation": eval_result.get("Empowerment", {}).get("Explanation", ""),
                "overall_raw": overall_raw, "overall_winner": overall_winner,
                "overall_explanation": eval_result.get("Overall Winner", {}).get("Explanation", ""),
            })
            
            print(f"   ✅ Comp: {comp_winner} | Div: {div_winner} | Emp: {emp_winner} | Overall: {overall_winner}")
        else:
            results.append({
                "question": question, "swapped": swapped,
                "comp_raw": "Error", "comp_winner": "Error",
                "comp_explanation": "Failed to evaluate",
                "div_raw": "Error", "div_winner": "Error",
                "div_explanation": "Failed to evaluate",
                "emp_raw": "Error", "emp_winner": "Error",
                "emp_explanation": "Failed to evaluate",
                "overall_raw": "Error", "overall_winner": "Error",
                "overall_explanation": "Failed to evaluate",
            })
            print(f"   ❌ Failed to evaluate")
        
        # Rate limiting
        time.sleep(8)
    
    elapsed = time.time() - start_time
    
    # ============================================================
    # Compute stats
    # ============================================================
    results_df = pd.DataFrame(results)
    
    print(f"\n{'=' * 70}")
    print(f"📊 EVALUATION RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total evaluated: {len(results_df)}")
    print(f"Swapped positions: {swap_count}/{len(results_df)} ({swap_count/len(results_df)*100:.0f}%)")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/len(results_df):.1f}s per question)")
    
    print(f"\n{'Metric':<20} | {'Naive Wins':<14} | {'Mix Wins':<14} | {'Unclear':<10}")
    print("-" * 65)
    
    metrics = [
        ("comp_winner", "Comprehensiveness"),
        ("div_winner", "Diversity"),
        ("emp_winner", "Empowerment"),
        ("overall_winner", "Overall"),
    ]
    
    summary_rows = []
    for col, name in metrics:
        naive_wins = int((results_df[col] == "Naive").sum())
        mix_wins = int((results_df[col] == "Mix").sum())
        unclear = int(len(results_df) - naive_wins - mix_wins)
        total_valid = naive_wins + mix_wins
        mix_rate = round(mix_wins / total_valid * 100, 1) if total_valid > 0 else 0
        
        print(f"{name:<20} | {naive_wins:<14} | {mix_wins:<14} | {unclear:<10}")
        summary_rows.append({
            "Metric": name,
            "Naive_Wins": naive_wins,
            "Mix_Wins": mix_wins,
            "Unclear": unclear,
            "Mix_Win_Rate_%": mix_rate
        })
    
    print(f"\n{'=' * 70}")
    
    # ============================================================
    # Save results
    # ============================================================
    output_file = rf"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\testcase\eval_130.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: Summary
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Detail with evaluation per question
        results_df.to_excel(writer, sheet_name='Detail', index=False)
        
        # Sheet 3: Original data for reference
        df[['question', 'answer', 'naive_response', 'mix_response']].to_excel(
            writer, sheet_name='Original_Data', index=False
        )
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"   • Sheet 'Summary': Win counts per metric")
    print(f"   • Sheet 'Detail': Per-question evaluation with explanations")
    print(f"   • Sheet 'Original_Data': Original questions and responses")
    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    run_evaluation()
