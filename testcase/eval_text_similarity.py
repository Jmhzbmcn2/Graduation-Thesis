"""
RAG Evaluation Script - Group 2: Automatic Text Similarity Metrics
Evaluates naive_response and mix_response against ground truth (answer)
using ROUGE-L, BLEU, BERTScore, and F1 metrics.

Based on standard NLP evaluation papers:
- ROUGE-L (Lin, 2004)
- BLEU (Papineni et al., 2002) 
- BERTScore (Zhang et al., 2020)
- Token F1 Score
"""
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
from collections import Counter

# ============================================================
# Metric Implementations
# ============================================================

def tokenize(text: str) -> list:
    """Simple whitespace + punctuation tokenizer for Vietnamese text"""
    if not text or pd.isna(text):
        return []
    # Lowercase and split on whitespace/punctuation
    text = text.lower().strip()
    tokens = re.findall(r'\w+', text)
    return tokens


def compute_f1(prediction: str, ground_truth: str) -> dict:
    """
    Compute token-level F1 score between prediction and ground truth.
    Returns precision, recall, and F1.
    """
    pred_tokens = tokenize(prediction)
    truth_tokens = tokenize(ground_truth)
    
    if not pred_tokens or not truth_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)
    
    # Common tokens
    common = sum((pred_counter & truth_counter).values())
    
    precision = common / len(pred_tokens) if pred_tokens else 0
    recall = common / len(truth_tokens) if truth_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def compute_rouge_l(prediction: str, ground_truth: str) -> dict:
    """Compute ROUGE-L using rouge-score library"""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = scorer.score(ground_truth, prediction)
        return {
            "precision": round(scores['rougeL'].precision, 4),
            "recall": round(scores['rougeL'].recall, 4),
            "f1": round(scores['rougeL'].fmeasure, 4)
        }
    except ImportError:
        print("⚠️ rouge-score not installed. Run: pip install rouge-score")
        return {"precision": 0, "recall": 0, "f1": 0}


def compute_bleu(prediction: str, ground_truth: str) -> float:
    """Compute BLEU score using NLTK"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        pred_tokens = tokenize(prediction)
        truth_tokens = tokenize(ground_truth)
        
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        # Use smoothing to avoid 0 scores for short sentences
        smoothie = SmoothingFunction().method1
        score = sentence_bleu([truth_tokens], pred_tokens, smoothing_function=smoothie)
        return round(score, 4)
    except ImportError:
        print("⚠️ nltk not installed. Run: pip install nltk")
        return 0.0


def compute_bert_score_batch(predictions: list, ground_truths: list) -> dict:
    """Compute BERTScore for a batch of predictions"""
    try:
        from bert_score import score as bert_score
        
        print("  🔄 Computing BERTScore (this may take a moment)...")
        P, R, F1 = bert_score(
            predictions, 
            ground_truths, 
            lang="vi",  # Vietnamese
            verbose=False,
            batch_size=16
        )
        return {
            "precision": [round(p.item(), 4) for p in P],
            "recall": [round(r.item(), 4) for r in R],
            "f1": [round(f.item(), 4) for f in F1]
        }
    except ImportError:
        print("⚠️ bert-score not installed. Run: pip install bert-score")
        n = len(predictions)
        return {"precision": [0]*n, "recall": [0]*n, "f1": [0]*n}


# ============================================================
# Main Evaluation
# ============================================================

def evaluate_responses(input_file: str, output_file: str = None):
    """Run all text similarity evaluations on test results"""
    
    print("=" * 70)
    print("📊 RAG Evaluation - Group 2: Automatic Text Similarity Metrics")
    print("=" * 70)
    
    # Read data
    print(f"\n📂 Reading: {input_file}")
    df = pd.read_excel(input_file)
    print(f"   Total test cases: {len(df)}")
    
    # Filter out rows with empty responses
    valid_mask = df['naive_response'].notna() & df['mix_response'].notna() & df['answer'].notna()
    df_valid = df[valid_mask].copy()
    print(f"   Valid test cases: {len(df_valid)}")
    
    ground_truths = df_valid['answer'].astype(str).tolist()
    naive_responses = df_valid['naive_response'].astype(str).tolist()
    mix_responses = df_valid['mix_response'].astype(str).tolist()
    
    # ---- 1. Token F1 Score ----
    print(f"\n🔹 Computing Token F1 Score...")
    naive_f1_scores = []
    mix_f1_scores = []
    for pred_n, pred_m, truth in zip(naive_responses, mix_responses, ground_truths):
        naive_f1_scores.append(compute_f1(pred_n, truth))
        mix_f1_scores.append(compute_f1(pred_m, truth))
    
    df_valid['naive_f1'] = [s['f1'] for s in naive_f1_scores]
    df_valid['mix_f1'] = [s['f1'] for s in mix_f1_scores]
    print(f"   ✅ Done")
    
    # ---- 2. ROUGE-L ----
    print(f"\n🔹 Computing ROUGE-L...")
    naive_rouge_scores = []
    mix_rouge_scores = []
    for pred_n, pred_m, truth in zip(naive_responses, mix_responses, ground_truths):
        naive_rouge_scores.append(compute_rouge_l(pred_n, truth))
        mix_rouge_scores.append(compute_rouge_l(pred_m, truth))
    
    df_valid['naive_rouge_l'] = [s['f1'] for s in naive_rouge_scores]
    df_valid['mix_rouge_l'] = [s['f1'] for s in mix_rouge_scores]
    print(f"   ✅ Done")
    
    # ---- 3. BLEU ----
    print(f"\n🔹 Computing BLEU...")
    naive_bleu_scores = []
    mix_bleu_scores = []
    for pred_n, pred_m, truth in zip(naive_responses, mix_responses, ground_truths):
        naive_bleu_scores.append(compute_bleu(pred_n, truth))
        mix_bleu_scores.append(compute_bleu(pred_m, truth))
    
    df_valid['naive_bleu'] = naive_bleu_scores
    df_valid['mix_bleu'] = mix_bleu_scores
    print(f"   ✅ Done")
    
    # ---- 4. BERTScore ----
    print(f"\n🔹 Computing BERTScore...")
    print(f"   Mode: Naive...")
    naive_bert = compute_bert_score_batch(naive_responses, ground_truths)
    print(f"   Mode: Mix...")
    mix_bert = compute_bert_score_batch(mix_responses, ground_truths)
    
    df_valid['naive_bertscore'] = naive_bert['f1']
    df_valid['mix_bertscore'] = mix_bert['f1']
    print(f"   ✅ Done")
    
    # ============================================================
    # Display Results
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"📊 EVALUATION RESULTS SUMMARY")
    print(f"{'=' * 70}")
    
    metrics = ['f1', 'rouge_l', 'bleu', 'bertscore']
    metric_names = ['Token F1', 'ROUGE-L', 'BLEU', 'BERTScore']
    
    print(f"\n{'Metric':<15} | {'Naive (Avg)':<15} | {'Mix (Avg)':<15} | {'Winner':<10} | {'Δ Improvement'}")
    print("-" * 75)
    
    results_summary = []
    for metric, name in zip(metrics, metric_names):
        naive_avg = df_valid[f'naive_{metric}'].mean()
        mix_avg = df_valid[f'mix_{metric}'].mean()
        winner = "Mix ✓" if mix_avg > naive_avg else ("Naive ✓" if naive_avg > mix_avg else "Tie")
        delta = mix_avg - naive_avg
        delta_pct = (delta / naive_avg * 100) if naive_avg > 0 else 0
        
        print(f"{name:<15} | {naive_avg:<15.4f} | {mix_avg:<15.4f} | {winner:<10} | {delta:+.4f} ({delta_pct:+.1f}%)")
        
        results_summary.append({
            'Metric': name,
            'Naive_Avg': round(naive_avg, 4),
            'Mix_Avg': round(mix_avg, 4),
            'Winner': winner.replace(' ✓', ''),
            'Delta': round(delta, 4),
            'Delta_Pct': round(delta_pct, 1)
        })
    
    print(f"\n{'=' * 70}")
    print(f"📈 DETAILED STATISTICS")
    print(f"{'=' * 70}")
    
    for metric, name in zip(metrics, metric_names):
        naive_col = f'naive_{metric}'
        mix_col = f'mix_{metric}'
        print(f"\n  {name}:")
        print(f"    Naive  → Mean: {df_valid[naive_col].mean():.4f}, Std: {df_valid[naive_col].std():.4f}, "
              f"Min: {df_valid[naive_col].min():.4f}, Max: {df_valid[naive_col].max():.4f}")
        print(f"    Mix    → Mean: {df_valid[mix_col].mean():.4f}, Std: {df_valid[mix_col].std():.4f}, "
              f"Min: {df_valid[mix_col].min():.4f}, Max: {df_valid[mix_col].max():.4f}")
    
    # Count wins per metric
    print(f"\n{'=' * 70}")
    print(f"🏆 HEAD-TO-HEAD COMPARISON (Per Question)")
    print(f"{'=' * 70}")
    print(f"\n{'Metric':<15} | {'Naive Wins':<12} | {'Mix Wins':<12} | {'Ties':<8}")
    print("-" * 55)
    
    for metric, name in zip(metrics, metric_names):
        naive_wins = (df_valid[f'naive_{metric}'] > df_valid[f'mix_{metric}']).sum()
        mix_wins = (df_valid[f'mix_{metric}'] > df_valid[f'naive_{metric}']).sum()
        ties = (df_valid[f'naive_{metric}'] == df_valid[f'mix_{metric}']).sum()
        print(f"{name:<15} | {naive_wins:<12} | {mix_wins:<12} | {ties:<8}")
    
    # ============================================================
    # Save Results
    # ============================================================
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"eval_text_similarity_{timestamp}.xlsx"
    
    # Save detailed results
    output_path = f"C:\\Users\\VUDUYLINH\\PycharmProjects\\KLTN\\LightRAG\\testcase\\{output_file}"
    
    # Select columns for output
    output_cols = ['question', 'answer', 'naive_response', 'mix_response',
                   'naive_f1', 'mix_f1', 'naive_rouge_l', 'mix_rouge_l',
                   'naive_bleu', 'mix_bleu', 'naive_bertscore', 'mix_bertscore']
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Detailed per-question results
        df_valid[output_cols].to_excel(writer, sheet_name='Detail', index=False)
        
        # Sheet 2: Summary
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"   • Sheet 'Detail': Per-question scores")
    print(f"   • Sheet 'Summary': Average metrics comparison")
    print(f"\n✅ Evaluation complete!")


if __name__ == '__main__':
    INPUT_FILE = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\testcase\test_results_20260204_014454.xlsx"
    evaluate_responses(INPUT_FILE)
