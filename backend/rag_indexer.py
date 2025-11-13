# rag_indexer.py
#import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import faiss
import numpy as np
import tiktoken
from openai import OpenAI

client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"
ENC = tiktoken.get_encoding("cl100k_base")


def extract_text_from_pdf(path):
    """Extract text page by page"""
    reader = PdfReader(path)
    text = []
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            text.append(content)
    return text


def chunk_text(texts, max_tokens=800, overlap=80):
    """Split text into overlapping chunks for embeddings"""
    chunks = []
    for i, page in enumerate(texts):
        tokens = ENC.encode(page)
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = ENC.decode(chunk_tokens)
            chunks.append({"text": chunk_text, "meta": {"page": i + 1}})
            start += max_tokens - overlap
    return chunks

def embed_texts(chunks):
    """Call OpenAI embeddings API"""
    texts = [c["text"] for c in chunks]
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    embs = np.array([r.embedding for r in res.data]).astype("float32")
    return embs


def build_faiss_index(embs):
    """Create FAISS index and add vectors"""
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embs)
    index.add(embs)
    return index


def retrieve(index, embs, chunks, query, k=5):
    """Retrieve top-k relevant chunks for a query"""
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    q_emb = np.array([q_emb]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, k)
    results = []
    for i, s in zip(ids[0], scores[0]):
        results.append({"text": chunks[i]["text"], "meta": chunks[i]["meta"], "score": float(s)})
    return results
