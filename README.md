# Vector Database Document Search

A Python application that demonstrates the use of vector databases for efficient document search and retrieval using LangChain and Pinecone. This implementation showcases a Retrieval-Augmented Generation (RAG) system that enhances Large Language Model responses with relevant context from your documents.

## Overview

This project implements a document ingestion and search system that:

1. Loads text documents using LangChain's document loaders
2. Splits them into manageable chunks using character-based text splitting
3. Creates vector embeddings using OpenAI's embedding model
4. Stores these embeddings in Pinecone for efficient similarity search
5. Provides a RAG implementation for context-aware question answering

## Features

- Document ingestion pipeline with customizable chunk sizes
- Vector embeddings generation using OpenAI
- Efficient similarity search using Pinecone vector database
- Context-aware question answering using LangChain's retrieval chain
- Customizable prompt templates for different use cases

## Prerequisites

- Python 3.x
- Pipenv for dependency management
- OpenAI API key
- Pinecone API key and environment

## Environment Variables

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
INDEX_NAME=your_pinecone_index_name
```

## Installation

1. Clone the repository
2. Install dependencies using Pipenv:
   ```bash
   pipenv install
   ```
3. Create and configure your `.env` file with the required API keys

## Usage

### Document Ingestion

Run the ingestion script to process and store documents:

```bash
python ingestion.py
```

This will:
- Load the text document
- Split it into chunks of 1000 characters
- Create embeddings
- Store them in Pinecone

### Question Answering

Run the main script to perform context-aware question answering:

```bash
python main.py
```

The system will:
- Retrieve relevant document chunks from Pinecone
- Use LangChain's retrieval chain to generate context-aware responses
- Provide answers based on the retrieved context

## Implementation Details

- Uses `CharacterTextSplitter` with 1000-character chunks for document processing
- Implements OpenAI embeddings for vector representation
- Utilizes LangChain's hub prompts for retrieval QA
- Employs the "stuff" documents chain for combining retrieved documents
- Implements a retrieval chain for context-aware responses

## Files

- `ingestion.py`: Handles document processing and storage
- `main.py`: Implements the RAG system for question answering
- `main-custom-prompt.py`: Alternative implementation with custom prompts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
