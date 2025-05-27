#!/usr/bin/env python
"""
Pinecone Document Embedder

This script processes documents from a dataset directory, extracts text content,
creates embeddings using OpenAI's text-embedding-3-large model, and indexes
them in Pinecone for efficient vector search.

Usage:
    python pinecone_embedder.py [--dataset_dir PATH] [--index_name NAME] [--namespace NAME]
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add these libraries to requirements.txt if not already present
try:
    import fitz  # PyMuPDF
    from bs4 import BeautifulSoup
    import openai
    from pinecone import Pinecone, PodSpec, CloudProvider, AwsRegion
    from tqdm import tqdm
except ImportError:
    print("Installing required dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "pymupdf", "beautifulsoup4", "openai", "pinecone-python>=3.0.0", "tqdm"])
    import fitz  # PyMuPDF
    from bs4 import BeautifulSoup
    import openai
    from pinecone import Pinecone, PodSpec, CloudProvider, AwsRegion
    from tqdm import tqdm

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
SUPPORTED_EXTENSIONS = ['.pdf', '.html', '.txt']
OPENAI_EMBED_MODEL = "text-embedding-3-large"
OPENAI_EMBED_DIMENSIONS = 3072  # Dimensions for text-embedding-3-large


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pinecone Document Embedder')
    parser.add_argument('--dataset_dir', type=str, default='../dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--index_name', type=str, default='document-embeddings',
                        help='Name of the Pinecone index')
    parser.add_argument('--namespace', type=str, default='default',
                        help='Namespace within the Pinecone index')
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f'Chunk size for text splitting (default: {DEFAULT_CHUNK_SIZE})')
    parser.add_argument('--chunk_overlap', type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f'Chunk overlap for text splitting (default: {DEFAULT_CHUNK_OVERLAP})')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of vectors to upsert in each batch')
    return parser.parse_args()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""


def extract_text_from_html(html_path: str) -> str:
    """Extract text content from an HTML file."""
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            return text
    except Exception as e:
        print(f"Error extracting text from HTML {html_path}: {e}")
        return ""


def extract_text_from_txt(txt_path: str) -> str:
    """Extract text content from a text file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error extracting text from TXT {txt_path}: {e}")
        return ""


def extract_text(file_path: str) -> str:
    """Extract text from a file based on its extension."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext == '.html':
        return extract_text_from_html(file_path)
    elif file_ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file extension: {file_ext}")
        return ""


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, 
               chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks with overlap."""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # If this is not the last chunk, try to break at a space
        if end < text_len:
            # Find the last space within the chunk
            while end > start and text[end] != ' ':
                end -= 1
            if end == start:  # No space found, just use the chunk_size
                end = min(start + chunk_size, text_len)
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move the start pointer, considering overlap
        start = end - chunk_overlap if end < text_len else text_len
    
    return chunks


def create_document_chunks(file_path: str, text_chunks: List[str]) -> List[Dict]:
    """Create document chunks with metadata."""
    rel_path = os.path.relpath(file_path)
    file_name = os.path.basename(file_path)
    
    documents = []
    for i, chunk in enumerate(text_chunks):
        doc = {
            "id": f"{uuid.uuid4()}",
            "text": chunk,
            "metadata": {
                "source": rel_path,
                "filename": file_name,
                "chunk": i,
                "chunk_size": len(chunk)
            }
        }
        documents.append(doc)
    
    return documents


def process_file(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Process a single file and return document chunks."""
    print(f"Processing {file_path}...")
    text = extract_text(file_path)
    if not text:
        print(f"No text extracted from {file_path}")
        return []
    
    text_chunks = chunk_text(text, chunk_size, chunk_overlap)
    return create_document_chunks(file_path, text_chunks)


def process_directory(directory: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Process all supported files in a directory and return document chunks."""
    all_documents = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in SUPPORTED_EXTENSIONS:
                document_chunks = process_file(file_path, chunk_size, chunk_overlap)
                all_documents.extend(document_chunks)
    
    return all_documents


def create_embeddings(documents: List[Dict]) -> List[Dict]:
    """Create embeddings for document chunks using OpenAI."""
    # Initialize the OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Prepare data for embedding
    embeddings_data = []
    
    # Process in batches to avoid rate limits
    batch_size = 100  # OpenAI API can handle up to 2048 in a batch, but we'll be conservative
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc["text"] for doc in batch]
        
        try:
            # Request embeddings
            response = client.embeddings.create(
                input=texts,
                model=OPENAI_EMBED_MODEL,
                dimensions=OPENAI_EMBED_DIMENSIONS
            )
            
            # Extract embeddings from response
            for j, embedding_data in enumerate(response.data):
                embeddings_data.append({
                    "id": batch[j]["id"],
                    "values": embedding_data.embedding,
                    "metadata": {
                        "text": batch[j]["text"],
                        **batch[j]["metadata"]
                    }
                })
            
            print(f"Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            print(f"Error creating embeddings for batch starting at index {i}: {e}")
            # Continue with the next batch
    
    return embeddings_data


def initialize_pinecone_index(index_name: str, dimension: int = OPENAI_EMBED_DIMENSIONS):
    """Initialize Pinecone client and create index if it doesn't exist."""
    # Initialize Pinecone client
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists, create if it doesn't
    existing_indexes = pc.list_indexes().names()
    
    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: {index_name}")
        # Create a new index with the OpenAI embeddings dimension
        # Using the latest API approach
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=PodSpec(
                environment="gcp-starter"  # Use the appropriate environment
            )
        )
        # Wait for index to be ready
        while True:
            try:
                index_info = pc.describe_index(index_name)
                if index_info.status.ready:
                    break
                print("Waiting for index to be ready...")
                time.sleep(5)
            except Exception as e:
                print(f"Error checking index status: {e}")
                time.sleep(5)
    
    # Connect to the index
    index = pc.Index(name=index_name)
    return index


def upsert_to_pinecone(index, embeddings_data: List[Dict], namespace: str, batch_size: int = 100):
    """Upsert embeddings to Pinecone index."""
    total_batches = (len(embeddings_data) + batch_size - 1) // batch_size
    
    for i in range(0, len(embeddings_data), batch_size):
        batch = embeddings_data[i:i + batch_size]
        
        try:
            # Format vectors for Pinecone using the latest API format
            vectors = [
                {
                    "id": item["id"],
                    "values": item["values"],
                    "metadata": item["metadata"]
                }
                for item in batch
            ]
            
            # Upsert to Pinecone with the latest API
            index.upsert(vectors=vectors, namespace=namespace)
            
            print(f"Upserted batch {i//batch_size + 1}/{total_batches} to Pinecone")
            
        except Exception as e:
            print(f"Error upserting batch to Pinecone: {e}")
            print(f"Error details: {str(e)}")
            # Continue with next batch rather than stopping everything
    
    print(f"Upserted {len(embeddings_data)} vectors to Pinecone namespace: {namespace}")
    
    # Display stats about the index after upserting
    try:
        stats = index.describe_index_stats()
        print(f"Index stats after upsert: {stats.namespaces.get(namespace, 'No stats available')}")
    except Exception as e:
        print(f"Could not retrieve index stats: {e}")
        pass


def main():
    """Main function to process documents and create embeddings."""
    args = parse_arguments()
    
    # Convert relative paths to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(script_dir, args.dataset_dir))
    
    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file or environment.")
        sys.exit(1)
    
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not found in environment variables.")
        print("Please set your Pinecone API key in the .env file or environment.")
        sys.exit(1)
    
    print(f"Processing documents from: {dataset_dir}")
    print(f"Documents will be indexed in Pinecone: {args.index_name}/{args.namespace}")
    print(f"Chunk size: {args.chunk_size}, Chunk overlap: {args.chunk_overlap}")
    
    # Process all documents
    documents = process_directory(dataset_dir, args.chunk_size, args.chunk_overlap)
    print(f"Total document chunks: {len(documents)}")
    
    # Create embeddings
    print("Creating embeddings with OpenAI's text-embedding-3-large model...")
    embeddings_data = create_embeddings(documents)
    
    # Initialize Pinecone
    print("Initializing Pinecone...")
    index = initialize_pinecone_index(args.index_name)
    
    # Upsert to Pinecone
    print(f"Upserting embeddings to Pinecone index: {args.index_name}")
    upsert_to_pinecone(index, embeddings_data, args.namespace, args.batch_size)
    
    print("Processing complete!")
    print(f"Embedded {len(embeddings_data)} document chunks in Pinecone index: {args.index_name}/{args.namespace}")


if __name__ == "__main__":
    main()
