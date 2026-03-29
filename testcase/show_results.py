import pandas as pd

df = pd.read_excel(r'C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\testcase\eval_text_similarity_20260302_010240.xlsx', sheet_name='Detail')

metrics = ['f1', 'rouge_l', 'bleu', 'bertscore']
names = ['Token F1', 'ROUGE-L', 'BLEU', 'BERTScore']

print("Head-to-Head Comparison:")
for m, n in zip(metrics, names):
    naive_wins = int((df[f'naive_{m}'] > df[f'mix_{m}']).sum())
    mix_wins = int((df[f'mix_{m}'] > df[f'naive_{m}']).sum())
    ties = int((df[f'naive_{m}'] == df[f'mix_{m}']).sum())
    print(f"  {n}: Naive wins={naive_wins}, Mix wins={mix_wins}, Ties={ties}")

print("\nDetailed Stats:")
for m, n in zip(metrics, names):
    print(f"\n  {n}:")
    print(f"    Naive -> Mean={df[f'naive_{m}'].mean():.4f}, Std={df[f'naive_{m}'].std():.4f}")
    print(f"    Mix   -> Mean={df[f'mix_{m}'].mean():.4f}, Std={df[f'mix_{m}'].std():.4f}")
