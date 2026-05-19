import os
import datasets
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config          import EMBED_MODEL, CACHE_CORPUS, CACHE_QUESTIONS, CACHE_QA, SYNTH_QUESTIONS
from src.generation  import generate_questions

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading datasets...")
corpus = datasets.load_dataset("isaacus/legal-rag-bench", name="corpus")
qa     = datasets.load_dataset("isaacus/legal-rag-bench", name="qa")

# Create corpus_df to later make corpus store, corpus embeddings
corpus_df = pd.DataFrame({
    "chunk_id": corpus['test']['id'],
    "text":     corpus['test']['text'],
}).reset_index(drop=True)

# Remove test chunks from corpus for training
test_chunk_ids = set(qa['test']['relevant_passage_id'])

# On this corpus the questions will be generated
train_corpus = corpus_df[~corpus_df['chunk_id'].isin(test_chunk_ids)].reset_index(drop=True)

# ── Generate synthetic questions ──────────────────────────────────────────────
print("Generating synthetic questions...")
question_rows = []
for _, row in train_corpus.iterrows():
    response  = generate_questions(row['text']) # Let LLM generate questions per chunk
    questions = response.get("questions", [])
    for q in questions:
        question_text = q.get("question", "").strip()
        if question_text:
            question_rows.append({"chunk_id": row['chunk_id'], "question": question_text})
    print(f"  Chunk {row['chunk_id']}: {len(questions)} question(s)")

questions_df = pd.DataFrame(question_rows)
questions_df = questions_df[~questions_df['chunk_id'].isin(test_chunk_ids)]
questions_df.to_excel(SYNTH_QUESTIONS, index=False)
print(f"✅ Saved synthetic questions to {SYNTH_QUESTIONS}")

# # ── Embed ─────────────────────────────────────────────────────────────────────
print("Embedding...")
embed_model = SentenceTransformer(EMBED_MODEL)

corpus_df['text_embedding'] = list(
    embed_model.encode(corpus_df['text'].tolist(), show_progress_bar=True)
)

question_embeddings = embed_model.encode(questions_df['question'].tolist(), show_progress_bar=True)
questions_df['question_embedding'] = list(question_embeddings)
questions_df = questions_df.merge(corpus_df[['chunk_id', 'text_embedding']], on='chunk_id', how='left')

qa_df = pd.DataFrame({
    "chunk_id": qa['test']['relevant_passage_id'],
    "question": qa['test']['question'],
    "answer":   qa['test']['answer']
})
qa_embeddings = embed_model.encode(qa_df['question'].tolist(), show_progress_bar=True)
qa_df['question_embedding'] = list(qa_embeddings)
qa_df = qa_df.merge(corpus_df[['chunk_id', 'text_embedding']], on='chunk_id', how='left')

# ── Save caches ───────────────────────────────────────────────────────────────
print("Saving caches...")
os.makedirs("data", exist_ok=True)
corpus_df.to_pickle(CACHE_CORPUS)
questions_df.to_pickle(CACHE_QUESTIONS)
qa_df.to_pickle(CACHE_QA)
print(f"Saved caches to data/")