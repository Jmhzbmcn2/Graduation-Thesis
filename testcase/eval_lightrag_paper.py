"""
RAG Evaluation Script - Group 1: LightRAG Paper Metrics
Evaluates naive_response vs mix_response using LLM-as-Judge (vLLM local)
on 3 criteria from the LightRAG paper:
- Comprehensiveness
- Diversity
- Empowerment

Cấu hình local server:
  - LLM Judge : vLLM (Qwen2.5-14B-Instruct-AWQ) tại http://localhost:8000
  - LightRAG  : http://localhost:9621
  - Input     : testcase/500_cases.csv  (cột: question, answer, ...)
  - Output    : testcase/eval_lightrag_paper_results.xlsx

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
import requests
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)

# ============================================================
# Configuration
# ============================================================

# LightRAG server (để query naive / mix response)
LIGHTRAG_URL = os.getenv("LIGHTRAG_URL", "http://localhost:9621")

# LLM Judge — vLLM local (OpenAI-compatible)
# Có thể override qua environment variables
JUDGE_MODEL    = os.getenv("EVAL_LLM_MODEL",    "Qwen/Qwen2.5-14B-Instruct-AWQ")
JUDGE_API_KEY  = os.getenv("EVAL_LLM_API_KEY",  "EMPTY")   # vLLM không cần key thật
JUDGE_BASE_URL = os.getenv("EVAL_LLM_BASE_URL", "http://localhost:8000/v1")

TEST_LIMIT = 50  # None = tất cả, số nguyên = giới hạn (ví dụ: 50)

# Đường dẫn file (server — relative to this script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE   = os.path.join(_SCRIPT_DIR, "500_cases.csv")
# Cache file: lưu responses đã query để không phải query lại
CACHE_FILE   = os.path.join(_SCRIPT_DIR, "eval_lightrag_paper_responses.xlsx")
OUTPUT_FILE  = os.path.join(_SCRIPT_DIR, "eval_lightrag_paper_results_500cases.xlsx")

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


# ============================================================
# Query LightRAG
# ============================================================

def clean_answer(answer: str) -> str:
    """Loại bỏ phần ### References khỏi answer"""
    return re.split(r"\n*###\s*References", answer, maxsplit=1)[0].strip()


def query_lightrag(question: str, mode: str, retries: int = 3) -> str:
    """
    Query LightRAG và trả về answer (string).
    mode: 'naive' hoặc 'mix'
    """
    base = {"query": question, "mode": mode, "stream": False, "top_k": 10}

    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{LIGHTRAG_URL}/query",
                json={**base, "only_need_context": False},
                timeout=180,
            )
            resp.raise_for_status()
            answer = clean_answer(resp.json().get("response", ""))
            return answer
        except Exception as e:
            print(f"    ⚠️ LightRAG query [{mode}] lần {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(3)

    return "Error: Không lấy được response"


def build_responses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Query LightRAG để lấy naive_response và mix_response cho từng câu hỏi.
    Nếu CACHE_FILE tồn tại, load từ cache (bỏ qua những câu đã có).
    Trả về DataFrame với cột: question, answer, naive_response, mix_response.
    """
    total = len(df)

    # --- Load cache nếu có ---
    if os.path.exists(CACHE_FILE):
        print(f"📂 Tìm thấy cache '{CACHE_FILE}', đang load...")
        cache_df = pd.read_excel(CACHE_FILE)
        cached_questions = set(cache_df["question"].tolist())
    else:
        cache_df = pd.DataFrame(columns=["question", "answer", "naive_response", "mix_response"])
        cached_questions = set()

    new_rows = []
    need_query = df[~df["question"].isin(cached_questions)]
    skip_count = len(df) - len(need_query)
    if skip_count > 0:
        print(f"   ↩️  Bỏ qua {skip_count} câu đã có trong cache")

    if len(need_query) > 0:
        print(f"\n🔄 Querying LightRAG cho {len(need_query)} câu mới...")
        for i, (_, row) in enumerate(need_query.iterrows(), 1):
            question = str(row.get("question", row.get("Question", ""))).strip()
            ground_truth = str(row.get("answer", row.get("Answer", ""))).strip()
            if not question or question == "nan":
                continue

            print(f"  [{i}/{len(need_query)}] {question[:65]}...")

            naive = query_lightrag(question, "naive")
            time.sleep(0.5)
            mix   = query_lightrag(question, "mix")
            time.sleep(0.5)

            new_rows.append({
                "question":       question,
                "answer":         ground_truth,
                "naive_response": naive,
                "mix_response":   mix,
            })

        # Lưu cache
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([cache_df, new_df], ignore_index=True)
        combined.to_excel(CACHE_FILE, index=False)
        print(f"💾 Cache đã lưu: {CACHE_FILE}")
    else:
        combined = cache_df

    # Giữ đúng thứ tự như df gốc
    result = df[["question"]].merge(
        combined[["question", "answer", "naive_response", "mix_response"]],
        on="question",
        how="left",
    )
    return result


# ============================================================
# LLM-as-Judge
# ============================================================

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


def evaluate_pair(client: OpenAI, question: str, response_a: str, response_b: str, idx: int) -> dict:
    """Evaluate two responses using local vLLM - no labels, just Answer 1 / Answer 2"""

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


# ============================================================
# Main
# ============================================================

def run_evaluation():
    """Run LightRAG paper evaluation on test results"""

    print("=" * 70)
    print("📊 RAG Evaluation - Group 1: LightRAG Paper Metrics")
    print("   (Comprehensiveness, Diversity, Empowerment)")
    print(f"   LLM Judge: {JUDGE_MODEL} @ {JUDGE_BASE_URL}")
    print(f"   LightRAG : {LIGHTRAG_URL}")
    print("   Anti-bias: Random position swap + No mode labels")
    print("=" * 70)

    # ── 1. Đọc test cases ──────────────────────────────────────────────
    print(f"\n📂 Reading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    if TEST_LIMIT is not None:
        df = df.head(TEST_LIMIT)
    print(f"   Tìm thấy {len(df)} test cases")

    # ── 2. Query LightRAG (hoặc load cache) ───────────────────────────
    df_with_responses = build_responses(df)

    # Lọc bỏ hàng lỗi
    df_with_responses = df_with_responses.dropna(subset=["naive_response", "mix_response"])
    df_with_responses = df_with_responses[
        ~df_with_responses["naive_response"].str.startswith("Error:")
        & ~df_with_responses["mix_response"].str.startswith("Error:")
    ]
    print(f"\n   Đánh giá {len(df_with_responses)} câu hỏi (sau khi lọc lỗi)")

    # ── 3. LLM-as-Judge ───────────────────────────────────────────────
    client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    random.seed(42)

    results = []
    start_time = time.time()
    swap_count = 0

    for idx, row in enumerate(df_with_responses.itertuples(), 1):
        question = str(row.question)
        naive    = str(row.naive_response)
        mix      = str(row.mix_response)

        # Random swap to eliminate position bias
        swapped = random.random() < 0.5
        if swapped:
            answer1, answer2 = mix, naive
            swap_count += 1
        else:
            answer1, answer2 = naive, mix

        print(f"\n🔍 [{idx}/{len(df_with_responses)}] {question[:60]}... {'[SWAPPED]' if swapped else ''}")

        eval_result = evaluate_pair(client, question, answer1, answer2, idx)

        if eval_result:
            comp_raw    = eval_result.get("Comprehensiveness", {}).get("Winner", "N/A")
            div_raw     = eval_result.get("Diversity",         {}).get("Winner", "N/A")
            emp_raw     = eval_result.get("Empowerment",       {}).get("Winner", "N/A")
            overall_raw = eval_result.get("Overall Winner",    {}).get("Winner", "N/A")

            comp_winner    = map_winner(comp_raw,    swapped)
            div_winner     = map_winner(div_raw,     swapped)
            emp_winner     = map_winner(emp_raw,     swapped)
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
                "overall_raw":        overall_raw, "overall_winner": overall_winner,
                "overall_explanation":eval_result.get("Overall Winner",    {}).get("Explanation", ""),
            })

            print(f"   ✅ Comp: {comp_winner} | Div: {div_winner} | Emp: {emp_winner} | Overall: {overall_winner}")
        else:
            results.append({
                "question": question, "swapped": swapped,
                "comp_raw":    "Error", "comp_winner":    "Error", "comp_explanation":    "Failed to evaluate",
                "div_raw":     "Error", "div_winner":     "Error", "div_explanation":     "Failed to evaluate",
                "emp_raw":     "Error", "emp_winner":     "Error", "emp_explanation":     "Failed to evaluate",
                "overall_raw": "Error", "overall_winner": "Error", "overall_explanation": "Failed to evaluate",
            })
            print(f"   ❌ Failed to evaluate")

        # Nhỏ delay để không overload vLLM
        time.sleep(1)

    elapsed = time.time() - start_time

    # ── 4. Tính stats ─────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    print(f"\n{'=' * 70}")
    print(f"📊 EVALUATION RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total evaluated: {len(results_df)}")
    print(f"Swapped positions: {swap_count}/{len(results_df)} ({swap_count/max(len(results_df),1)*100:.0f}%)")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/max(len(results_df),1):.1f}s per question)")

    print(f"\n{'Metric':<20} | {'Naive Wins':<14} | {'Mix Wins':<14} | {'Unclear':<10}")
    print("-" * 65)

    metrics = [
        ("comp_winner",    "Comprehensiveness"),
        ("div_winner",     "Diversity"),
        ("emp_winner",     "Empowerment"),
        ("overall_winner", "Overall"),
    ]

    summary_rows = []
    for col, name in metrics:
        naive_wins  = int((results_df[col] == "Naive").sum())
        mix_wins    = int((results_df[col] == "Mix").sum())
        unclear     = int(len(results_df) - naive_wins - mix_wins)
        total_valid = naive_wins + mix_wins
        mix_rate    = round(mix_wins / total_valid * 100, 1) if total_valid > 0 else 0

        print(f"{name:<20} | {naive_wins:<14} | {mix_wins:<14} | {unclear:<10}")
        summary_rows.append({
            "Metric":        name,
            "Naive_Wins":    naive_wins,
            "Mix_Wins":      mix_wins,
            "Unclear":       unclear,
            "Mix_Win_Rate_%": mix_rate,
        })

    print(f"\n{'=' * 70}")

    # ── 5. Lưu kết quả ────────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # Sheet 1: Summary
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 2: Detail with evaluation per question
        results_df.to_excel(writer, sheet_name='Detail', index=False)

        # Sheet 3: Original data (responses) for reference
        df_with_responses[["question", "answer", "naive_response", "mix_response"]].to_excel(
            writer, sheet_name='Original_Data', index=False
        )

    print(f"\n💾 Results saved to: {OUTPUT_FILE}")
    print(f"   • Sheet 'Summary'       : Win counts per metric")
    print(f"   • Sheet 'Detail'        : Per-question evaluation with explanations")
    print(f"   • Sheet 'Original_Data' : Questions + LightRAG responses")
    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    run_evaluation()
