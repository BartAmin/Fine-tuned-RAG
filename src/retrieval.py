import torch
import torch.nn.functional as F
import pandas as pd
from config import TOP_K

def retrieve(test_qa, corpus_embeddings, corpus_store, model=None, top_k=TOP_K):
    """Retrieve top-k chunks for each question. Returns df with top_k_ids."""
    results = []
    for _, row in test_qa.iterrows():
        q_emb = torch.tensor(row['question_embedding'], dtype=torch.float32).unsqueeze(0)

        if model is not None:
            with torch.no_grad():
                q_emb = model(q_emb)

        similarities  = F.cosine_similarity(q_emb, corpus_embeddings)
        top_k_indices = similarities.topk(top_k).indices.tolist()
        top_k_ids     = corpus_store.iloc[top_k_indices]['chunk_id'].tolist()

        results.append({
            "question":      row['question'],
            "true_chunk_id": row['chunk_id'],
            "top_k_ids":     top_k_ids,
            "hit":           row['chunk_id'] in top_k_ids,
        })

    return pd.DataFrame(results)