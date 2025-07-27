from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

# Load and split document
text = Path("mydoc.txt").read_text(encoding="utf-8", errors="replace")
# chunks = [text[i:i+500] for i in range(0, len(text), 500)]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_text(text)
# Embed chunks
model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('bge-base-en')
embeddings = model.encode(docs)

# Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "my_index.faiss")

