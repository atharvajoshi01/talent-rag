# Talent Intelligence Assistant

A production-grade, end-to-end Retrieval-Augmented Generation (RAG) system for talent intelligence and recruitment automation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TALENT INTELLIGENCE ASSISTANT                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Streamlit  │───▶│   FastAPI   │───▶│ RAG Pipeline│───▶│    LLM      │  │
│  │     UI      │    │   Backend   │    │             │    │  (GPT-4o)   │  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘    └─────────────┘  │
│                                               │                             │
│                           ┌───────────────────┼───────────────────┐         │
│                           │                   │                   │         │
│                           ▼                   ▼                   ▼         │
│                    ┌───────────┐       ┌───────────┐       ┌───────────┐   │
│                    │ Retriever │       │ Reranker  │       │ Generator │   │
│                    │ (Hybrid)  │       │           │       │           │   │
│                    └─────┬─────┘       └───────────┘       └───────────┘   │
│                          │                                                  │
│                          ▼                                                  │
│                    ┌───────────┐       ┌───────────┐       ┌───────────┐   │
│                    │   FAISS   │◀──────│ Embeddings│◀──────│  Chunker  │   │
│                    │  Index    │       │ (MPNet)   │       │           │   │
│                    └───────────┘       └───────────┘       └───────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Data Layer: candidates.json, roles.json, FAISS indices                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Semantic Search**: Find candidates using natural language queries
- **Hybrid Retrieval**: Combines semantic similarity with metadata filtering
- **Intelligent Chunking**: Document-type-aware chunking for resumes, roles, and interviews
- **Reranking**: Cross-encoder and hybrid score reranking for improved relevance
- **Ensemble Generation**: Multiple response variations with judge-based selection
- **Grounded Responses**: LLM responses with evidence citations
- **Evaluation Suite**: Comprehensive metrics (Recall, Precision, NDCG, MRR)

## Project Structure

```
talent_rag/
├── data/                    # Generated data storage
├── data_generation/         # Synthetic data generators
│   ├── candidate_generator.py
│   ├── role_generator.py
│   └── generate_all.py
├── preprocessing/           # Text cleaning and chunking
│   ├── text_cleaner.py
│   ├── chunker.py
│   └── document_processor.py
├── embeddings/              # Embedding model wrappers
│   └── embedding_model.py
├── vectorstore/             # FAISS implementation
│   └── faiss_store.py
├── retrieval/               # Search functionality
│   ├── semantic.py
│   ├── hybrid.py
│   └── index_builder.py
├── rag/                     # RAG pipeline components
│   ├── retriever.py
│   ├── reranker.py
│   ├── prompt_builder.py
│   ├── generator.py
│   ├── evaluator.py
│   └── pipeline.py
├── ensemble/                # Ensemble generation
│   ├── ensemble_generator.py
│   └── judge.py
├── api/                     # FastAPI backend
│   ├── main.py
│   └── schemas.py
├── ui/                      # Streamlit frontend
│   └── app.py
├── evaluation/              # Metrics and benchmarking
│   ├── metrics.py
│   └── benchmark.py
├── tests/                   # Test suite
├── docker/                  # Docker configurations
├── config.py                # Configuration management
├── requirements.txt
└── docker-compose.yml
```

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for LLM generation)
- Docker & Docker Compose (optional)

### Installation

1. **Clone and setup environment:**

```bash
cd talent_rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment:**

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. **Generate synthetic data:**

```python
from talent_rag.data_generation import generate_all_data

generate_all_data(
    num_candidates=100,
    num_roles=20,
    seed=42
)
```

4. **Build the index:**

```python
from talent_rag.retrieval import IndexBuilder

builder = IndexBuilder()
vector_store = builder.build_index(
    candidates_path="data/candidates.json",
    roles_path="data/roles.json"
)
builder.save_index("data/indices")
```

5. **Start the API:**

```bash
uvicorn talent_rag.api.main:app --reload
```

6. **Start the UI:**

```bash
streamlit run talent_rag/ui/app.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# API available at: http://localhost:8000
# UI available at: http://localhost:8501
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and index status |
| `/search_candidates` | POST | Search for candidates with filters |
| `/compare_candidates` | POST | Compare multiple candidates for a role |
| `/ask` | POST | General RAG question answering |
| `/roles` | GET | List all available roles |
| `/candidates` | GET | List all candidates (paginated) |

### Example API Usage

```python
import requests

# Search for candidates
response = requests.post("http://localhost:8000/search_candidates", json={
    "query": "Senior ML engineer with NLP experience",
    "filters": {
        "skills": ["Python", "PyTorch"],
        "seniority": ["Senior", "Staff"],
        "min_experience": 5
    },
    "k": 10
})

print(response.json())
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | Sentence transformer model |
| `LLM_MODEL` | `gpt-4o` | OpenAI model for generation |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |
| `TOP_K_RETRIEVAL` | `10` | Documents to retrieve |
| `RERANK_TOP_K` | `5` | Documents after reranking |
| `FAISS_INDEX_TYPE` | `flat` | Index type (flat/hnsw) |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=talent_rag --cov-report=html
```

## Evaluation

Run the benchmark suite:

```python
from talent_rag.evaluation import RAGBenchmark

benchmark = RAGBenchmark(pipeline)
results = benchmark.run(verbose=True)
benchmark.save_results(results, "benchmark_results.json")
```

## Key Design Decisions

### Why FAISS with IndexFlatIP?
- **Exact search**: IndexFlatIP provides exact nearest neighbor search
- **Cosine similarity**: With normalized embeddings, inner product equals cosine similarity
- **Simplicity**: No training required, easy to update

### Why Hybrid Retrieval?
- **Semantic matching**: Captures meaning beyond keywords
- **Metadata filtering**: Enables precise skill/location/seniority filtering
- **Keyword boosting**: Ensures important terms are weighted

### Why Ensemble Generation?
- **Diversity**: Multiple prompt variations increase coverage
- **Quality**: Judge selection improves response quality
- **Reliability**: Reduces single-point failures

## Technology Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o |
| Embeddings | Sentence Transformers (all-mpnet-base-v2) |
| Vector Store | FAISS |
| Backend | FastAPI (async) |
| Frontend | Streamlit |
| Containerization | Docker & Docker Compose |

## Performance Considerations

- **Embedding caching**: Embeddings are cached to avoid recomputation
- **Batch processing**: Documents are embedded in batches for efficiency
- **Async API**: FastAPI provides async request handling
- **Index persistence**: FAISS indices are persisted to disk

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
