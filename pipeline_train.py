import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from config    import CACHE_QUESTIONS, CACHE_QA, CACHE_CORPUS
from src.train import run_optuna

print("Loading caches...")
questions_df = pd.read_pickle(CACHE_QUESTIONS)
qa_df        = pd.read_pickle(CACHE_QA)
corpus_df    = pd.read_pickle(CACHE_CORPUS)

# ── Split by chunk_id so no chunk appears in both splits ──────────────────────
unique_chunks              = questions_df['chunk_id'].unique()
train_chunks, val_chunks   = train_test_split(unique_chunks, test_size=0.15, random_state=42)
train_df                   = questions_df[questions_df['chunk_id'].isin(train_chunks)].reset_index(drop=True)
val_df                     = questions_df[questions_df['chunk_id'].isin(val_chunks)].reset_index(drop=True)

embeddings_train = train_df[['chunk_id', 'question_embedding', 'text_embedding']]

corpus_embeddings = torch.tensor(
    np.stack(corpus_df['text_embedding'].values), dtype=torch.float32
)
corpus_store = corpus_df[['chunk_id', 'text']].reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Corpus: {len(corpus_store)}")

study = run_optuna(embeddings_train, val_df, corpus_embeddings, corpus_store)
print("Training complete.")