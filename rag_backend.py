import os
import numpy as np
import torch
import fitz  # PyMuPDF
import pickle
from sentence_transformers import SentenceTransformer


# Global data structures to store document and chunk info
documents = []   # List of dicts: {doc_id, filename, priority}
all_chunks = []  # List of dicts: {doc_id, doc_name, chunk_text, embedding, priority, page_number?}
document_counter = 0

# Global constants and parameters
chunk_length = 5    # 5 sentences per chunk

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
    if check_for_pkl(file_path):
        with open(file_path.split('.')[0] + '.pkl', 'rb') as file:
            document_chunks = pickle.load(file)  

        for chunk in document_chunks:
            all_chunks.append({
                "doc_id": doc_id,
                "doc_name": chunk["doc_name"],
                "chunk_text": chunk["chunk_text"],
                "embedding": chunk["embedding"],
                "priority": chunk["priority"]
            })
    else:

        doc = fitz.open(file_path)
        document_chunks = []

        for page in doc:
            # Extract text, sentences, and chunks from each page
            text = page.get_text().replace("\n", " ").strip()
            sentences = [s.strip() for s in text.split(". ") if s.strip()]
            sentence_groups = split_list(sentences, chunk_length)
            print(f'{len(sentence_groups)} chunks')

            # Encode and save the chunks
            for group in sentence_groups:
                chunk_text = " ".join(group)
                # Do not add the chunk if it has less than 30 characters (likely ill-formatted)
                if len(chunk_text) < 30:
                    continue

                print('Saving chunk')
                embedding = embedding_model.encode(chunk_text, convert_to_numpy=True)
                all_chunks.append({
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "chunk_text": chunk_text,
                    "embedding": embedding,
                    "priority": priority,
                    "page_number": page.number  # store the page number (zero-indexed)
                })
                document_chunks.append({
                    "doc_name": doc_name,
                    "chunk_text": chunk_text,
                    "embedding": embedding,
                    "priority": priority
                })
        doc.close()

        print(document_chunks)

        # Save chunks to pkl file for quick reload
        doc_chunk_save_file = file_path.split('.')[0] + '.pkl'
        with open(doc_chunk_save_file, 'wb') as file:
            pickle.dump(document_chunks, file)


def process_text_file(file_path, doc_id, doc_name, priority):
    """
    Processes a plain text file:
      - Reads the file
      - Splits text into sentences and groups them into chunks
      - Computes embeddings for each chunk
      - Appends each chunk's info to the global all_chunks list
    """
    if check_for_pkl(file_path):
        with open(file_path.split('.')[0] + '.pkl', 'rb') as file:
            document_chunks = pickle.load(file)  

        for chunk in document_chunks:
            all_chunks.append({
                "doc_id": doc_id,
                "doc_name": chunk["doc_name"],
                "chunk_text": chunk["chunk_text"],
                "embedding": chunk["embedding"],
                "priority": chunk["priority"]
            })
    
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split document into sentences
        sentences = [s.strip() for s in text.split(". ") if s.strip()]
        sentence_groups = split_list(sentences, chunk_length)
        document_chunks = []

        # Encode and save the chunks
        for group in sentence_groups:
            chunk_text = " ".join(group)
            # Do not add the chunk if it has less than 30 characters (likely ill-formatted)
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
            document_chunks.append({
                "doc_name": doc_name,
                "chunk_text": chunk_text,
                "embedding": embedding,
                "priority": priority
            })

        # Save chunks to pkl file for quick reload
        doc_chunk_save_file = file_path.split('.')[0] + '.pkl'
        with open(doc_chunk_save_file, 'wb') as file:
            pickle.dump(document_chunks, file)


def check_for_pkl(file_path):
    pkl_file = file_path.split('.')[0] + '.pkl'
    return os.path.exists(pkl_file)