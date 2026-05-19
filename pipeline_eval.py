import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config         import CACHE_CORPUS, CACHE_QA, RESULTS_CSV, TOP_K, RESULTS_PLOT
from src.train      import load_best_model
from src.retrieval  import retrieve
from src.generation import generate_answer, judge_answers

METRICS = ["faithfulness", "answer_relevance", "completeness", "conciseness", "semantic_similarity", "overall"]

# ── Load caches ───────────────────────────────────────────────────────────────
print("Loading caches...")
corpus_df = pd.read_pickle(CACHE_CORPUS)
qa_df     = pd.read_pickle(CACHE_QA)

corpus_embeddings = torch.tensor(
    np.stack(corpus_df['text_embedding'].values), dtype=torch.float32
)
corpus_store = corpus_df[['chunk_id', 'text']].reset_index(drop=True)
test_qa      = qa_df[['chunk_id', 'question', 'question_embedding']]
print(f"✅ Loaded {len(corpus_store)} corpus chunks | {len(test_qa)} test questions")

# ── Load neural net ───────────────────────────────────────────────────────────
model = load_best_model()

# ── Step 1: Retrieval Hit Rate ────────────────────────────────────────────────
print("\n── Step 1: Retrieval Evaluation ──────────────────────────────────────")
baseline_df = retrieve(test_qa, corpus_embeddings, corpus_store, model=None)
neural_df   = retrieve(test_qa, corpus_embeddings, corpus_store, model=model)

baseline_hits = baseline_df['hit'].sum()
neural_hits   = neural_df['hit'].sum()
baseline_rate = baseline_df['hit'].mean()
neural_rate   = neural_df['hit'].mean()
total         = len(baseline_df)

# ── Print retrieval report ────────────────────────────────────────────────────
improvement = neural_rate - baseline_rate
print(f"\n{'':30s} {'Hits':>6}  {'Rate':>7}")
print(f"{'-'*46}")
print(f"{'Baseline (raw embeddings)':<30s} {baseline_hits:>4}/{total:<4}  {baseline_rate:.2%}")
print(f"{'Neural RAG (transformed)':<30s} {neural_hits:>4}/{total:<4}  {neural_rate:.2%}")
print(f"{'-'*46}")
print(f"{'Δ improvement':<30s} {neural_hits - baseline_hits:>+5}      {improvement:>+.2%}")

# ── Plot retrieval comparison ─────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

fig, axes  = plt.subplots(1, 2, figsize=(12, 5))
colors     = {"Baseline": "#4C72B0", "Neural": "#DD8452"}
labels     = ["Baseline", "Neural"]
fig.suptitle("Retrieval: Baseline vs Neural RAG", fontsize=14, fontweight='bold')

ax1  = axes[0]
rates = [baseline_rate, neural_rate]
bars  = ax1.bar(labels, [r * 100 for r in rates], color=colors.values(), width=0.4, edgecolor='white')
for bar, rate in zip(bars, rates):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{rate:.1%}", ha='center', va='bottom', fontweight='bold')
ax1.set_title(f"Hit Rate @ Top-{TOP_K}")
ax1.set_ylabel("Hit Rate (%)")
ax1.set_ylim(0, max(rates) * 100 * 1.2)
ax1.spines[['top', 'right']].set_visible(False)

ax2    = axes[1]
hits   = [baseline_hits, neural_hits]
misses = [total - baseline_hits, total - neural_hits]
ax2.bar(labels, hits,   color=colors.values(), label='Hit',  edgecolor='white')
ax2.bar(labels, misses, bottom=hits, color=["#aec6e8", "#f4c79a"], label='Miss', edgecolor='white')
for i, (h, m) in enumerate(zip(hits, misses)):
    ax2.text(i, h / 2,     f"{h}", ha='center', va='center', fontweight='bold', color='white')
    ax2.text(i, h + m / 2, f"{m}", ha='center', va='center', fontweight='bold', color='grey')
ax2.set_title("Hits vs Misses")
ax2.set_ylabel("Count")
ax2.legend(loc='upper right')
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(RESULTS_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Retrieval plot saved to {RESULTS_PLOT}")

# ── Step 2: Generate Answers ──────────────────────────────────────────────────
print("\n── Step 2: Generating Answers ────────────────────────────────────────")

def get_context(chunk_ids, corpus_store):
    return "\n\n".join(
        corpus_store[corpus_store['chunk_id'].isin(chunk_ids)]['text'].tolist()
    )

answers = []
for i, (_, row) in enumerate(qa_df.iterrows()):
    print(f"  Generating {i+1}/{len(qa_df)}: {row['question'][:60]}...")
    answers.append({
        "question":        row['question'],
        "ground_truth":    row['answer'],
        "chunk_id":        row['chunk_id'],
        "answer_baseline": generate_answer(row['question'], get_context(baseline_df.iloc[i]['top_k_ids'], corpus_store)),
        "answer_neural":   generate_answer(row['question'], get_context(neural_df.iloc[i]['top_k_ids'],   corpus_store)),
    })

print(f"Generated {len(answers)} answer pairs")

# ── Step 3: LLM Judge ─────────────────────────────────────────────────────────
print("\n── Step 3: LLM Judge Evaluation ──────────────────────────────────────")
results = []

for i, ans in enumerate(answers):
    print(f"\n  Judging {i+1}/{len(answers)}: {ans['question'][:60]}...")
    score_baseline, score_neural = judge_answers(
        question        = ans['question'],
        ground_truth    = ans['ground_truth'],
        result_baseline = ans['answer_baseline'],
        result_neural   = ans['answer_neural'],
    )

    if not score_baseline or not score_neural:
        print(f"  ⚠️ Judge returned None — skipping")
        continue

    if not all(m in score_baseline and m in score_neural for m in METRICS):
        print(f"  ⚠️ Malformed response — skipping")
        continue

    results.append({**ans, "baseline": score_baseline, "neural": score_neural})

    # ── Running averages ──────────────────────────────────────────────────────
    print(f"\n  {'Metric':<25} {'Baseline':>10} {'Neural':>10} {'Winner':>10}")
    print(f"  {'─'*58}")
    for metric in METRICS:
        avg_b  = np.mean([r['baseline'][metric] for r in results])
        avg_n  = np.mean([r['neural'][metric]   for r in results])
        winner = "Neural ✅" if avg_n > avg_b else ("Baseline ✅" if avg_b > avg_n else "Tie 🟰")
        print(f"  {metric:<25} {avg_b:>10.2f} {avg_n:>10.2f} {winner:>10}")

    time.sleep(1)

# ── Save results ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_CSV, index=False)
print(f"\n Results saved to {RESULTS_CSV}")

# ── Plot LLM judge metrics ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

x      = np.arange(len(METRICS))
width  = 0.35
avgs_b = [results_df['baseline'].apply(lambda r: r[m]).mean() for m in METRICS]
avgs_n = [results_df['neural'].apply(lambda r: r[m]).mean()   for m in METRICS]

bars_b = ax.bar(x - width/2, avgs_b, width, label='Baseline', color='#4C72B0', edgecolor='white')
bars_n = ax.bar(x + width/2, avgs_n, width, label='Neural',   color='#DD8452', edgecolor='white')

for bar in bars_b:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=9)
for bar in bars_n:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', ' ').title() for m in METRICS], rotation=15, ha='right')
ax.set_ylabel("Average Score (1-5)")
ax.set_ylim(0, 6)
ax.set_title("LLM Judge: Baseline vs Neural RAG")
ax.legend()
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("results/judge_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(" Judge plot saved to results/judge_comparison.png")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"FINAL SUMMARY  ({len(results)}/{len(answers)} questions evaluated)")
print(f"{'='*70}")
print(f"{'Metric':<25} {'Baseline':>10} {'Neural':>10} {'Winner':>10}")
print(f"{'─'*58}")
for metric in METRICS:
    avg_b  = results_df['baseline'].apply(lambda x: x[metric]).mean()
    avg_n  = results_df['neural'].apply(lambda x: x[metric]).mean()
    winner = "Neural ✅" if avg_n > avg_b else ("Baseline ✅" if avg_b > avg_n else "Tie 🟰")
    print(f"{metric:<25} {avg_b:>10.2f} {avg_n:>10.2f} {winner:>10}")

print(f"\n{'─'*58}")
print(f"Retrieval — Baseline: {baseline_rate:.2%} | Neural: {neural_rate:.2%} | Δ {neural_rate - baseline_rate:+.2%}")
print(f"{'='*70}")