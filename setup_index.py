#!/usr/bin/env python3
"""Script to build the vector index."""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

print(f"API Key loaded: {os.getenv('OPENAI_API_KEY', 'NOT SET')[:25]}...")

from talent_rag.retrieval import IndexBuilder  # noqa: E402
from talent_rag.config import settings  # noqa: E402

print("Building index with Ollama embeddings...")
print("Make sure Ollama is running and nomic-embed-text is pulled:")
print("  ollama serve")
print("  ollama pull nomic-embed-text")
print()

builder = IndexBuilder(use_ollama_embeddings=True)
vector_store = builder.build_index(
    candidates_path=settings.candidates_path,
    roles_path=settings.roles_path
)
builder.save_index(settings.index_dir)
print(f"Index built and saved to {settings.index_dir}")
print(f"Total documents indexed: {len(vector_store)}")
