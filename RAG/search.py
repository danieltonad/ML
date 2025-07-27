from pydoc import text
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

def get_top_k_chunks(query, k=3):
    text = Path("mydoc.txt").read_text(encoding="utf-8", errors="replace")
    # chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text)

    index = faiss.read_index("my_index.faiss")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(docs)
    q_embedding = model.encode([query])
    D, I = index.search(q_embedding, k)
    return [docs[i] for i in I[0]]


# print(get_top_k_chunks("What service does the company render?", k=3))
