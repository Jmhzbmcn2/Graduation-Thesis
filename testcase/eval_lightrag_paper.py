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
# Configuration
# ============================================================
JUDGE_MODEL    = "qwen/qwen3-30b-a3b-instruct-2507"
JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")

TEST_LIMIT = 300  # Limit to 20 cases

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE   = os.path.join(_SCRIPT_DIR, "eval_ragas_beam.xlsx")
OUTPUT_FILE  = os.path.join(_SCRIPT_DIR, "eval_lightrag_paper_results_mix_vs_beam.xlsx")

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
                extra_headers={"HTTP-Referer": "https://github.com/vuduylinh", "X-Title": "LightRAG-Eval"}
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

def map_winner(raw_winner: str, swapped: bool) -> str:
    is_answer1 = "1" in raw_winner
    is_answer2 = "2" in raw_winner

    if not swapped:
        if is_answer1:
            return "Mix"
        elif is_answer2:
            return "Beam"
    else:
        if is_answer1:
            return "Beam"
        elif is_answer2:
            return "Mix"

    return "Unclear"

def run_evaluation():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 70)
    print("📊 RAG Evaluation - Group 1: LightRAG Paper Metrics")
    print("   Comparing Mix Mode vs Beam Mode")
    print(f"   LLM Judge: {JUDGE_MODEL} @ {JUDGE_BASE_URL}")
    print("=" * 70)

    if not JUDGE_API_KEY:
        print("Vui lòng thiết lập OPENROUTER_API_KEY trong file .env")
        return

    print(f"\n📂 Reading data from: {INPUT_FILE}")
    try:
        all_sheets = pd.read_excel(INPUT_FILE, sheet_name=None)
    except Exception as e:
        print(f"Lỗi đọc file {INPUT_FILE}: {e}")
        return

    if "Mix" not in all_sheets or "Beam" not in all_sheets:
        print("Lỗi: Không tìm thấy sheet 'Mix' hoặc 'Beam' trong file Excel.")
        return

    df_mix = all_sheets["Mix"].copy()
    df_beam = all_sheets["Beam"].copy()

    # Get up to 20 cases
    df_mix = df_mix.head(TEST_LIMIT)
    df_beam = df_beam.head(TEST_LIMIT)

    print(f"   Tìm thấy {len(df_mix)} câu trong sheet Mix và {len(df_beam)} câu trong sheet Beam")

    client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
    random.seed(42)

    results = []
    start_time = time.time()
    swap_count = 0

    for idx in range(min(len(df_mix), len(df_beam))):
        question = str(df_mix.iloc[idx].get("user_input", ""))
        mix_resp = str(df_mix.iloc[idx].get("response", ""))
        beam_resp = str(df_beam.iloc[idx].get("response", ""))

        if not question or question == "nan":
            continue

        swapped = random.random() < 0.5
        if swapped:
            answer1, answer2 = beam_resp, mix_resp
            swap_count += 1
        else:
            answer1, answer2 = mix_resp, beam_resp

        print(f"\n🔍 [{idx+1}/{TEST_LIMIT}] {question[:60]}... {'[SWAPPED]' if swapped else ''}")

        eval_result = evaluate_pair(client, question, answer1, answer2, idx+1)

        if eval_result:
            comp_raw    = eval_result.get("Comprehensiveness", {}).get("Winner", "N/A")
            div_raw     = eval_result.get("Diversity",         {}).get("Winner", "N/A")
            emp_raw     = eval_result.get("Empowerment",       {}).get("Winner", "N/A")
            dir_raw     = eval_result.get("Directness",        {}).get("Winner", "N/A")
            overall_raw = eval_result.get("Overall Winner",    {}).get("Winner", "N/A")

            comp_winner    = map_winner(comp_raw,    swapped)
            div_winner     = map_winner(div_raw,     swapped)
            emp_winner     = map_winner(emp_raw,     swapped)
            dir_winner     = map_winner(dir_raw,     swapped)
            overall_winner = map_winner(overall_raw, swapped)

            results.append({
                "question":           question,
                "swapped":            swapped,
                "comp_raw":           comp_raw,    "comp_winner":    comp_winner,
                "comp_explanation":   eval_result.get("Comprehensiveness", {}).get("Explanation", ""),
                "div_raw":            div_raw,     "div_winner":     div_winner,
                "div_explanation":    eval_result.get("Diversity",         {}).get("Explanation", ""),
                "emp_raw":            emp_raw,     "emp_winner":     emp_winner,
                "emp_explanation":    eval_result.get("Empowerment",       {}).get("Explanation", ""),
                "dir_raw":            dir_raw,     "dir_winner":     dir_winner,
                "dir_explanation":    eval_result.get("Directness",        {}).get("Explanation", ""),
                "overall_raw":        overall_raw, "overall_winner": overall_winner,
                "overall_explanation":eval_result.get("Overall Winner",    {}).get("Explanation", ""),
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

    print(f"\n{'=' * 70}")
    print(f"📊 EVALUATION RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total evaluated: {len(results_df)}")
    print(f"Swapped positions: {swap_count}/{len(results_df)} ({swap_count/max(len(results_df),1)*100:.0f}%)")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/max(len(results_df),1):.1f}s per question)")

    print(f"\n{'Metric':<20} | {'Mix Wins':<14} | {'Beam Wins':<14} | {'Unclear':<10}")
    print("-" * 65)

    metrics = [
        ("comp_winner",    "Comprehensiveness"),
        ("div_winner",     "Diversity"),
        ("emp_winner",     "Empowerment"),
        ("dir_winner",     "Directness"),
        ("overall_winner", "Overall"),
    ]

    summary_rows = []
    for col, name in metrics:
        mix_wins    = int((results_df[col] == "Mix").sum())
        beam_wins   = int((results_df[col] == "Beam").sum())
        unclear     = int(len(results_df) - mix_wins - beam_wins)
        total_valid = mix_wins + beam_wins
        beam_rate    = round(beam_wins / total_valid * 100, 1) if total_valid > 0 else 0

        print(f"{name:<20} | {mix_wins:<14} | {beam_wins:<14} | {unclear:<10}")
        summary_rows.append({
            "Metric":        name,
            "Mix_Wins":      mix_wins,
            "Beam_Wins":     beam_wins,
            "Unclear":       unclear,
            "Beam_Win_Rate_%": beam_rate,
        })

    print(f"\n{'=' * 70}")

    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)
        results_df.to_excel(writer, sheet_name='Detail', index=False)
        
        # Thêm dữ liệu gốc
        df_combined = pd.DataFrame({
            "question": df_mix["user_input"].values[:len(results_df)],
            "mix_response": df_mix["response"].values[:len(results_df)],
            "beam_response": df_beam["response"].values[:len(results_df)],
        })
        df_combined.to_excel(writer, sheet_name='Original_Data', index=False)

    print(f"\n💾 Results saved to: {OUTPUT_FILE}")
    print(f"\n✅ Evaluation complete!")

if __name__ == "__main__":
    run_evaluation()
