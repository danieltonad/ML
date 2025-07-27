def get_top_k_chunks(query, k=3):
    q_embedding = model.encode([query])
    D, I = index.search(q_embedding, k)
    return [chunks[i] for i in I[0]]
