
import os
import json
import argparse

import faiss
import numpy as np
import pymupdf

from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# Extract text from PDF
# --------------------------------------------------

def extract_pdf_pages(pdf_path):
    """
    Extract text from every page of a PDF.

    Returns:
        List of dictionaries containing page number and text.
    """

    pages = []

    document = pymupdf.open(pdf_path)

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text").strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    document.close()

    return pages


# --------------------------------------------------
# Split text into chunks
# --------------------------------------------------

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping character-based chunks.
    """

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# --------------------------------------------------
# Detect department
# --------------------------------------------------

def get_department(pdf_path, data_dir):
    """
    Department is determined from the first folder
    inside the main knowledge-base directory.
    """

    relative_path = os.path.relpath(pdf_path, data_dir)

    parts = relative_path.split(os.sep)

    if len(parts) >= 2:
        return parts[0]

    return "unknown"


# --------------------------------------------------
# Build chunks and metadata
# --------------------------------------------------

def build_chunks(data_dir):

    all_chunks = []

    pdf_files = []

    for root, dirs, files in os.walk(data_dir):

        for filename in files:

            if filename.lower().endswith(".pdf"):

                pdf_files.append(
                    os.path.join(root, filename)
                )

    print(f"Found {len(pdf_files)} PDF files.")

    chunk_id = 0

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):

        source_file = os.path.basename(pdf_path)

        department = get_department(
            pdf_path,
            data_dir
        )

        pages = extract_pdf_pages(pdf_path)

        for page_data in pages:

            page_number = page_data["page"]

            text = page_data["text"]

            chunks = split_text(text)

            for chunk in chunks:

                metadata = {
                    "id": chunk_id,
                    "department": department,
                    "source_file": source_file,
                    "page": page_number,
                    "text": chunk
                }

                all_chunks.append(metadata)

                chunk_id += 1

    return all_chunks


# --------------------------------------------------
# Create FAISS index
# --------------------------------------------------

def create_faiss_index(chunks, output_dir):

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(f"Creating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")

    # Inner Product works well with normalized embeddings
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    index_path = os.path.join(
        output_dir,
        "daraz.index"
    )

    faiss.write_index(
        index,
        index_path
    )

    print(f"\nFAISS index saved to:")
    print(index_path)

    return embeddings


# --------------------------------------------------
# Save metadata
# --------------------------------------------------

def save_metadata(chunks, output_dir):

    metadata_path = os.path.join(
        output_dir,
        "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Metadata saved to:")
    print(metadata_path)


# --------------------------------------------------
# Save manifest
# --------------------------------------------------

def save_manifest(chunks, output_dir):

    departments = {}

    source_files = set()

    for chunk in chunks:

        department = chunk["department"]

        departments[department] = (
            departments.get(department, 0) + 1
        )

        source_files.add(
            chunk["source_file"]
        )

    manifest = {
        "total_chunks": len(chunks),
        "total_source_files": len(source_files),
        "departments": departments,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP
    }

    manifest_path = os.path.join(
        output_dir,
        "manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Manifest saved to:")
    print(manifest_path)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    print("=" * 60)
    print("DARAZ KNOWLEDGE BASE INGESTION")
    print("=" * 60)

    print(f"\nData directory:")
    print(args.data_dir)

    print(f"\nOutput directory:")
    print(args.output_dir)

    # Build chunks
    chunks = build_chunks(
        args.data_dir
    )

    if not chunks:

        raise ValueError(
            "No text chunks were created. "
            "Check your PDF files."
        )

    print(
        f"\nTotal chunks created: {len(chunks)}"
    )

    # Create embeddings + FAISS
    create_faiss_index(
        chunks,
        args.output_dir
    )

    # Save metadata
    save_metadata(
        chunks,
        args.output_dir
    )

    # Save manifest
    save_manifest(
        chunks,
        args.output_dir
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
