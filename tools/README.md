# Document Embedder Tools

This directory contains tools for embedding documents from the dataset directory into Pinecone using OpenAI's text-embedding-3-large model.

## Setup

1. Create or update your `.env` file in the project root with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

2. Install the required dependencies:

```bash
pip install openai pinecone-python>=3.0.0 pymupdf beautifulsoup4 tqdm python-dotenv
```

## Tools

### 1. Pinecone Embedder

This script processes documents from your dataset directory, creates embeddings using OpenAI's text-embedding-3-large model, and stores them in Pinecone.

**Usage:**

```bash
python pinecone_embedder.py [--dataset_dir PATH] [--index_name NAME] [--namespace NAME]
```

**Arguments:**

- `--dataset_dir`: Path to the dataset directory (default: ../dataset)
- `--index_name`: Name of the Pinecone index (default: document-embeddings)
- `--namespace`: Namespace within the Pinecone index (default: default)
- `--chunk_size`: Size of text chunks for embedding (default: 1000)
- `--chunk_overlap`: Overlap between chunks (default: 200)
- `--batch_size`: Number of vectors to upsert in each batch (default: 100)

**Example:**

```bash
python pinecone_embedder.py --dataset_dir ../dataset --index_name company-docs --namespace internal-policy
```

### 2. Document Embedder (Local Storage Version)

This script is similar to the Pinecone Embedder but stores embeddings locally using FAISS.

**Usage:**

```bash
python document_embedder.py [--dataset_dir PATH] [--output_dir PATH]
```

**Arguments:**

- `--dataset_dir`: Path to the dataset directory (default: ../dataset)
- `--output_dir`: Path to save embeddings (default: ../embeddings)
- `--chunk_size`: Size of text chunks for embedding (default: 1000)
- `--chunk_overlap`: Overlap between chunks (default: 200)

**Example:**

```bash
python document_embedder.py --dataset_dir ../dataset --output_dir ../embeddings
```

## Supported File Types

Both tools currently support the following document types:
- PDF (.pdf)
- HTML (.html)
- Text (.txt)

## How It Works

1. The tools scan the dataset directory for supported document types
2. They extract text from each document
3. The text is split into chunks with specified size and overlap
4. OpenAI's text-embedding-3-large model generates embeddings for each chunk
5. The embeddings are stored in Pinecone (pinecone_embedder.py) or locally using FAISS (document_embedder.py)

## Using with Context7 MCP

These tools incorporate the latest Pinecone Python client (v3.0.0) and OpenAI embedding models based on the Context7 MCP documentation. The embeddings can be used with any Context7 MCP that supports Pinecone vector retrieval.
