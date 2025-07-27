from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss
import numpy as np

# Load and split document
text = Path("mydoc.txt").read_text()
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# Embed chunks
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
