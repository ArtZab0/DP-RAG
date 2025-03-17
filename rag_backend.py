import os
import numpy as np
import torch
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# Global data structures to store document and chunk info
documents = []   # List of dicts: {doc_id, filename, priority}
all_chunks = []  # List of dicts: {doc_id, doc_name, chunk_text, embedding, priority, page_number?}
document_counter = 0

# Set device (GPU if available) and load the embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)

def split_list(lst, chunk_size):
    """Splits a list into sublists of size 'chunk_size'."""
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def process_pdf(file_path, doc_id, doc_name, priority):
    """
    Processes a PDF file:
      - Extracts text page by page (using PyMuPDF)
      - Splits each page's text into sentences and groups them into chunks
      - Computes embeddings for each chunk
      - Appends each chunk's info (including its page number) to the global all_chunks list
    """
    doc = fitz.open(file_path)
    for page in doc:
        text = page.get_text().replace("\n", " ").strip()
        sentences = [s.strip() for s in text.split(". ") if s.strip()]
        chunk_size = 5  # adjust chunk size as needed
        sentence_groups = split_list(sentences, chunk_size)
        for group in sentence_groups:
            chunk_text = " ".join(group)
            if len(chunk_text) < 30:
                continue
            embedding = embedding_model.encode(chunk_text, convert_to_numpy=True)
            all_chunks.append({
                "doc_id": doc_id,
                "doc_name": doc_name,
                "chunk_text": chunk_text,
                "embedding": embedding,
                "priority": priority,
                "page_number": page.number  # store the page number (zero-indexed)
            })
    doc.close()

def process_text_file(file_path, doc_id, doc_name, priority):
    """
    Processes a plain text file:
      - Reads the file
      - Splits text into sentences and groups them into chunks
      - Computes embeddings for each chunk
      - Appends each chunk's info to the global all_chunks list
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    sentence_groups = split_list(sentences, 5)
    for group in sentence_groups:
        chunk_text = " ".join(group)
        if len(chunk_text) < 30:
            continue
        embedding = embedding_model.encode(chunk_text, convert_to_numpy=True)
        all_chunks.append({
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chunk_text": chunk_text,
            "embedding": embedding,
            "priority": priority
        })
