import numpy as np

def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)

def mrr_at_k(retrieved_ids, relevant_ids, k):
    for i, idx in enumerate(retrieved_ids[:k]):
        if idx in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    # simplified: assumes binary relevance
    rel = [1 if idx in relevant_ids else 0 for idx in retrieved_ids[:k]]
    dcg = sum(rel[i] / np.log2(i+2) for i in range(len(rel)))
    ideal_rel = sorted(rel, reverse=True)
    idcg = sum(ideal_rel[i] / np.log2(i+2) for i in range(len(ideal_rel)))
    return dcg / idcg if idcg > 0 else 0.0