from pydoc import text
import faiss, pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Load the index
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("my_index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def get_top_k_chunks(query, k=3):
    """Search FAISS index for top-k most relevant chunks."""
    # Encode the query into vector space
    query_vec = model.encode([query])
    
    # Search in FAISS
    distances, indices = index.search(query_vec, k)
    
    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        results.append({
            "rank": rank,
            "score": float(dist),  # Lower score = more similar for L2
            "text": chunks[idx]
        })
    
    return results



# print(get_top_k_chunks("What's this file about?", k=3))
