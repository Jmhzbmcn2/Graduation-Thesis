import os
import sys
import random
import pandas as pd
import json
import time
import re
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

# Load .env
load_dotenv()

# ============================================================
# Configuration — Local LLM (vLLM)
# ============================================================
JUDGE_MODEL    = os.environ.get("EVAL_LLM_MODEL",        "Qwen/Qwen2.5-14B-Instruct-AWQ")
JUDGE_BASE_URL = os.environ.get("EVAL_LLM_BINDING_HOST", "http://localhost:8000/v1")
JUDGE_API_KEY  = os.environ.get("EVAL_LLM_BINDING_API_KEY", "sk-123456")

TEST_LIMIT = 300  # Limit to 20 cases per comparison

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE   = os.path.join(_SCRIPT_DIR, "focused_hyper_param.xlsx")
OUTPUT_FILE  = os.path.join(_SCRIPT_DIR, "llm_as_judge_focused_vs_hybrid_mix.xlsx")

# Comparisons to run: (sheet_A, label_A, sheet_B, label_B)
COMPARISONS = [
    ("Focused", "Focused", "Hybrid", "Hybrid"),
    ("Focused", "Focused", "Mix",    "Mix"),
]

# Prompt
EVAL_PROMPT_TEMPLATE = """
You will evaluate two answers to the same question based on four criteria: **Comprehensiveness**, **Diversity**, **Empowerment**, and **Directness**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?
- **Directness**:  How specifically and clearly does the answer address the question?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these four categories.

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the four criteria listed above and provide detailed explanations for each criterion.

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
    "Directness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the four criteria]"
    }}
}}"""

SYS_PROMPT = """
---Role---
You are an expert tasked with evaluating two answers to the same question based on four criteria: **Comprehensiveness**, **Diversity**, **Empowerment**, and **Directness**.
"""

def extract_json(text: str) -> dict:
    text = text.strip()
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

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

def evaluate_pair(client: OpenAI, question: str, response_a: str, response_b: str, idx: int) -> dict:
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
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYS_PROMPT.strip()},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw_text = response.choices[0].message.content or ""
            result = extract_json(raw_text)
            if result:
                return result
            else:
                print(f"    ⚠️ Q{idx}: Failed to parse JSON (attempt {attempt+1}), retrying...")
                time.sleep(2)
        except Exception as e:
            error_msg = str(e)[:120]
            print(f"    ❌ Q{idx}: Error (attempt {attempt+1}): {error_msg}")
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
    return None

def map_winner(raw_winner: str, swapped: bool, label_a: str, label_b: str) -> str:
    """
    When NOT swapped: Answer 1 = label_a, Answer 2 = label_b
    When swapped:     Answer 1 = label_b, Answer 2 = label_a
    """
    is_answer1 = "1" in raw_winner
    is_answer2 = "2" in raw_winner

    if not swapped:
        if is_answer1:
            return label_a
        elif is_answer2:
            return label_b
    else:
        if is_answer1:
            return label_b
        elif is_answer2:
            return label_a

    return "Unclear"

def run_single_comparison(
    client: OpenAI,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> tuple[list, list]:
    """Run one comparison between df_a (label_a) and df_b (label_b).

    Returns (results, summary_rows).
    """
    n = min(len(df_a), len(df_b), TEST_LIMIT)
    print(f"\n{'─' * 70}")
    print(f"🔬 Comparison: {label_a} vs {label_b}  ({n} questions)")
    print(f"{'─' * 70}")

    random.seed(42)
    results = []
    swap_count = 0
    start_time = time.time()

    for idx in range(n):
        question  = str(df_a.iloc[idx].get("user_input", ""))
        resp_a    = str(df_a.iloc[idx].get("response",   ""))
        resp_b    = str(df_b.iloc[idx].get("response",   ""))

        if not question or question == "nan":
            continue

        swapped = random.random() < 0.5
        if swapped:
            answer1, answer2 = resp_b, resp_a
            swap_count += 1
        else:
            answer1, answer2 = resp_a, resp_b

        print(f"\n🔍 [{idx+1}/{n}] {question[:60]}... {'[SWAPPED]' if swapped else ''}")

        eval_result = evaluate_pair(client, question, answer1, answer2, idx + 1)

        if eval_result:
            comp_raw    = eval_result.get("Comprehensiveness", {}).get("Winner", "N/A")
            div_raw     = eval_result.get("Diversity",         {}).get("Winner", "N/A")
            emp_raw     = eval_result.get("Empowerment",       {}).get("Winner", "N/A")
            dir_raw     = eval_result.get("Directness",        {}).get("Winner", "N/A")
            overall_raw = eval_result.get("Overall Winner",    {}).get("Winner", "N/A")

            comp_winner    = map_winner(comp_raw,    swapped, label_a, label_b)
            div_winner     = map_winner(div_raw,     swapped, label_a, label_b)
            emp_winner     = map_winner(emp_raw,     swapped, label_a, label_b)
            dir_winner     = map_winner(dir_raw,     swapped, label_a, label_b)
            overall_winner = map_winner(overall_raw, swapped, label_a, label_b)

            results.append({
                "question":            question,
                "swapped":             swapped,
                "comp_raw":            comp_raw,    "comp_winner":    comp_winner,
                "comp_explanation":    eval_result.get("Comprehensiveness", {}).get("Explanation", ""),
                "div_raw":             div_raw,     "div_winner":     div_winner,
                "div_explanation":     eval_result.get("Diversity",         {}).get("Explanation", ""),
                "emp_raw":             emp_raw,     "emp_winner":     emp_winner,
                "emp_explanation":     eval_result.get("Empowerment",       {}).get("Explanation", ""),
                "dir_raw":             dir_raw,     "dir_winner":     dir_winner,
                "dir_explanation":     eval_result.get("Directness",        {}).get("Explanation", ""),
                "overall_raw":         overall_raw, "overall_winner": overall_winner,
                "overall_explanation": eval_result.get("Overall Winner",    {}).get("Explanation", ""),
            })

            print(f"   ✅ Comp: {comp_winner} | Div: {div_winner} | Emp: {emp_winner} | Dir: {dir_winner} | Overall: {overall_winner}")
        else:
            results.append({
                "question": question, "swapped": swapped,
                "comp_raw":    "Error", "comp_winner":    "Error", "comp_explanation":    "Failed to evaluate",
                "div_raw":     "Error", "div_winner":     "Error", "div_explanation":     "Failed to evaluate",
                "emp_raw":     "Error", "emp_winner":     "Error", "emp_explanation":     "Failed to evaluate",
                "dir_raw":     "Error", "dir_winner":     "Error", "dir_explanation":     "Failed to evaluate",
                "overall_raw": "Error", "overall_winner": "Error", "overall_explanation": "Failed to evaluate",
            })
            print(f"   ❌ Failed to evaluate")

        time.sleep(1)

    elapsed = time.time() - start_time
    results_df = pd.DataFrame(results)

    # ── Print per-comparison summary ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"📊 SUMMARY: {label_a} vs {label_b}")
    print(f"{'=' * 70}")
    print(f"Total evaluated : {len(results_df)}")
    print(f"Swapped         : {swap_count}/{len(results_df)} ({swap_count/max(len(results_df),1)*100:.0f}%)")
    print(f"Elapsed         : {elapsed:.1f}s ({elapsed/max(len(results_df),1):.1f}s/question)")
    print(f"\n{'Metric':<20} | {label_a+' Wins':<16} | {label_b+' Wins':<16} | {'Unclear':<10}")
    print("-" * 70)

    metrics = [
        ("comp_winner",    "Comprehensiveness"),
        ("div_winner",     "Diversity"),
        ("emp_winner",     "Empowerment"),
        ("dir_winner",     "Directness"),
        ("overall_winner", "Overall"),
    ]

    summary_rows = []
    for col, name in metrics:
        a_wins      = int((results_df[col] == label_a).sum())
        b_wins      = int((results_df[col] == label_b).sum())
        unclear     = int(len(results_df) - a_wins - b_wins)
        total_valid = a_wins + b_wins
        a_rate      = round(a_wins / total_valid * 100, 1) if total_valid > 0 else 0

        print(f"{name:<20} | {a_wins:<16} | {b_wins:<16} | {unclear:<10}")
        summary_rows.append({
            "Metric":              name,
            f"{label_a}_Wins":    a_wins,
            f"{label_b}_Wins":    b_wins,
            "Unclear":            unclear,
            f"{label_a}_Win_Rate_%": a_rate,
        })

    print(f"\n{'=' * 70}")
    return results, summary_rows


def run_evaluation():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 70)
    print("📊 RAG Evaluation - LightRAG Paper Metrics")
    print("   Focused vs Hybrid  |  Focused vs Mix")
    print(f"   LLM Judge: {JUDGE_MODEL} @ {JUDGE_BASE_URL}")
    print("=" * 70)

    if not JUDGE_API_KEY:
        print("Vui lòng thiết lập EVAL_LLM_BINDING_API_KEY trong file .env")
        return

    print(f"\n📂 Reading data from: {INPUT_FILE}")
    try:
        all_sheets = pd.read_excel(INPUT_FILE, sheet_name=None)
    except Exception as e:
        print(f"Lỗi đọc file {INPUT_FILE}: {e}")
        return

    required = {"Hybrid", "Mix", "Focused"}
    missing  = required - set(all_sheets.keys())
    if missing:
        print(f"Lỗi: Không tìm thấy sheet(s): {missing}")
        return

    df_hybrid  = all_sheets["Hybrid"].head(TEST_LIMIT).copy()
    df_mix     = all_sheets["Mix"].head(TEST_LIMIT).copy()
    df_focused = all_sheets["Focused"].head(TEST_LIMIT).copy()

    print(f"   Focused : {len(df_focused)} rows")
    print(f"   Hybrid  : {len(df_hybrid)} rows")
    print(f"   Mix     : {len(df_mix)} rows")

    client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    # ── Run both comparisons ─────────────────────────────────────────────
    fh_results, fh_summary = run_single_comparison(
        client, df_focused, df_hybrid, "Focused", "Hybrid"
    )
    fm_results, fm_summary = run_single_comparison(
        client, df_focused, df_mix, "Focused", "Mix"
    )

    # ── Save results ─────────────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # Focused vs Hybrid
        pd.DataFrame(fh_summary).to_excel(writer, sheet_name='Summary_Focused_vs_Hybrid', index=False)
        pd.DataFrame(fh_results).to_excel(writer, sheet_name='Detail_Focused_vs_Hybrid',  index=False)

        # Focused vs Mix
        pd.DataFrame(fm_summary).to_excel(writer, sheet_name='Summary_Focused_vs_Mix',    index=False)
        pd.DataFrame(fm_results).to_excel(writer, sheet_name='Detail_Focused_vs_Mix',     index=False)

        # Original data side-by-side
        n = min(len(df_focused), len(df_hybrid), len(df_mixed := df_mix), TEST_LIMIT)
        df_orig = pd.DataFrame({
            "question":        df_focused["user_input"].values[:n],
            "focused_response": df_focused["response"].values[:n],
            "hybrid_response":  df_hybrid["response"].values[:n],
            "mix_response":     df_mix["response"].values[:n],
        })
        df_orig.to_excel(writer, sheet_name='Original_Data', index=False)

    print(f"\n💾 Results saved to: {OUTPUT_FILE}")
    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    run_evaluation()
