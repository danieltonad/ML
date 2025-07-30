from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss
from bs4 import BeautifulSoup
import docx
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------- FILE LOADING HELPERS ---------------

def load_txt(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")

def load_pdf(path):
    reader = PdfReader(path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def load_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def load_html(path):
    html = Path(path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")

# --------------- MAIN ---------------

filepath = "sample.pdf"  # can be .txt, .pdf, .docx, .html

if filepath.endswith(".txt"):
    text = load_txt(filepath)
elif filepath.endswith(".pdf"):
    text = load_pdf(filepath)
elif filepath.endswith(".docx"):
    text = load_docx(filepath)
elif filepath.endswith(".html"):
    text = load_html(filepath)
else:
    raise ValueError("Unsupported file type")

print(f"Loaded {len(text)} characters from {filepath}")

# --------------- SMART CHUNKING ---------------

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
chunks = splitter.split_text(text)

print(f"✅ Split into {len(chunks)} chunks")

# --------------- EMBEDDINGS + FAISS ---------------

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "my_index.faiss")
print("✅ Index saved: my_index.faiss")

import pickle

# After creating `chunks` and `index`
faiss.write_index(index, "my_index.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

