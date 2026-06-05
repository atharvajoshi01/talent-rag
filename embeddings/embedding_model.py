"""
Embedding Model Wrapper Module.

This module provides a unified interface for generating embeddings
using either local Sentence Transformers or OpenAI's embedding API.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from functools import lru_cache

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.

    This class defines the interface that all embedding model
    implementations must follow.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Numpy array of shape (dimension,)
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of shape (len(texts), dimension)
        """
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.

        Some models use different embeddings for queries vs documents.
        Default implementation calls embed_text.

        Args:
            query: Query text to embed

        Returns:
            Numpy array of shape (dimension,)
        """
        return self.embed_text(query)

    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings for cosine similarity via inner product.

        Args:
            embeddings: Embeddings to normalize

        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms


class SentenceTransformerEmbedding(EmbeddingModel):
    """
    Embedding model using Sentence Transformers.

    This class wraps the sentence-transformers library to provide
    local embedding generation using models like all-mpnet-base-v2.

    Attributes:
        model: Loaded SentenceTransformer model
        _dimension: Embedding dimensionality
        _model_name: Name of the model
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the Sentence Transformer embedding model.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on ('cpu', 'cuda', or None for auto)
            normalize_embeddings: Whether to L2 normalize embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self._model_name = model_name
        self._normalize = normalize_embeddings

        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"Loaded {model_name} with dimension={self._dimension}, "
            f"device={self.model.device}"
        )

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize
        )
        return embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=show_progress
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.

        For symmetric models like mpnet, this is the same as embed_text.

        Args:
            query: Query text

        Returns:
            Query embedding
        """
        return self.embed_text(query)


class OllamaEmbedding(EmbeddingModel):
    """
    Embedding model using Ollama's local embedding API.

    This class uses Ollama for lightweight local embeddings
    without the memory overhead of loading large transformer models.

    Attributes:
        _model_name: Name of the Ollama embedding model
        _dimension: Embedding dimensionality
        _host: Ollama server host URL
    """

    # Known dimensions for Ollama embedding models
    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "all-minilm": 384,
        "mxbai-embed-large": 1024,
    }

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        dimensions: Optional[int] = None
    ):
        """
        Initialize the Ollama embedding model.

        Args:
            model_name: Name of the Ollama embedding model
            host: Ollama server host URL
            dimensions: Override embedding dimensions
        """
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError(
                "requests is required. Install with: pip install requests"
            )

        self._model_name = model_name
        self._host = host.rstrip('/')
        self._api_url = f"{self._host}/api/embeddings"

        # Get dimension from known models or use default
        if dimensions:
            self._dimension = dimensions
        else:
            self._dimension = self.MODEL_DIMENSIONS.get(model_name, 768)

        # Verify connection and get actual dimension
        self._verify_and_detect_dimension()

        logger.info(
            f"Initialized Ollama embedding model: {model_name}, "
            f"dimension={self._dimension}, host={host}"
        )

    def _verify_and_detect_dimension(self):
        """Verify Ollama connection and detect embedding dimension."""
        try:
            # Try to embed a test string to get actual dimension
            response = self._requests.post(
                self._api_url,
                json={"model": self._model_name, "prompt": "test"},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                if "embedding" in data:
                    self._dimension = len(data["embedding"])
                    logger.debug(f"Detected embedding dimension: {self._dimension}")
            else:
                logger.warning(
                    f"Ollama returned status {response.status_code}. "
                    f"Make sure model '{self._model_name}' is pulled."
                )
        except self._requests.exceptions.ConnectionError:
            logger.warning(
                f"Cannot connect to Ollama at {self._host}. "
                "Make sure Ollama is running: ollama serve"
            )

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        response = self._requests.post(
            self._api_url,
            json={"model": self._model_name, "prompt": text},
            timeout=60
        )
        response.raise_for_status()

        data = response.json()
        embedding = np.array(data["embedding"], dtype=np.float32)
        return self.normalize(embedding.reshape(1, -1))[0]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Ollama doesn't have native batch support, so we call sequentially.

        Args:
            texts: List of texts to embed
            batch_size: Ignored (for interface compatibility)

        Returns:
            Array of embeddings
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.

        Args:
            query: Query text

        Returns:
            Query embedding
        """
        return self.embed_text(query)


class OpenAIEmbedding(EmbeddingModel):
    """
    Embedding model using OpenAI's embedding API.

    This class uses OpenAI's text-embedding models for high-quality
    embeddings with the tradeoff of API costs and latency.

    Attributes:
        client: OpenAI client instance
        _model_name: Name of the OpenAI embedding model
        _dimension: Embedding dimensionality
    """

    # Known dimensions for OpenAI models
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None
    ):
        """
        Initialize the OpenAI embedding model.

        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key (uses env var if not provided)
            dimensions: Optional dimension reduction (for embedding-3 models)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required. Install with: pip install openai"
            )

        self._model_name = model_name

        # Handle dimensions
        if dimensions:
            self._dimension = dimensions
        else:
            self._dimension = self.MODEL_DIMENSIONS.get(model_name, 3072)

        self._request_dimensions = dimensions  # Only used for embedding-3 models

        # Initialize client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()  # Uses OPENAI_API_KEY env var

        logger.info(
            f"Initialized OpenAI embedding model: {model_name}, "
            f"dimension={self._dimension}"
        )

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """
        Call OpenAI embedding API with retry logic.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        kwargs = {
            "input": texts,
            "model": self._model_name
        }

        # Add dimensions parameter for embedding-3 models if specified
        if self._request_dimensions and "embedding-3" in self._model_name:
            kwargs["dimensions"] = self._request_dimensions

        response = self.client.embeddings.create(**kwargs)

        # Sort by index to ensure correct order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [e.embedding for e in embeddings]

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        embeddings = self._call_api([text])
        embedding = np.array(embeddings[0], dtype=np.float32)
        return self.normalize(embedding)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        OpenAI API has a limit on input size, so we batch requests.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            Array of embeddings
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        result = np.array(all_embeddings, dtype=np.float32)
        return self.normalize(result)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.

        Args:
            query: Query text

        Returns:
            Query embedding
        """
        return self.embed_text(query)


@lru_cache(maxsize=1)
def get_embedding_model(
    use_openai: bool = False,
    use_ollama: bool = False,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    ollama_host: Optional[str] = None
) -> EmbeddingModel:
    """
    Get a cached embedding model instance.

    This function provides a singleton-like access to embedding models,
    avoiding repeated model loading.

    Args:
        use_openai: Whether to use OpenAI embeddings
        use_ollama: Whether to use Ollama embeddings (recommended for local)
        model_name: Model name to use
        api_key: API key for OpenAI (if applicable)
        ollama_host: Ollama server host URL

    Returns:
        EmbeddingModel instance
    """
    from ..config import settings

    if use_openai:
        model_name = model_name or settings.openai_embedding_model
        logger.info(f"Creating OpenAI embedding model: {model_name}")
        return OpenAIEmbedding(model_name=model_name, api_key=api_key or settings.openai_api_key)
    elif use_ollama:
        model_name = model_name or "nomic-embed-text"
        host = ollama_host or settings.ollama_host
        logger.info(f"Creating Ollama embedding model: {model_name}")
        return OllamaEmbedding(model_name=model_name, host=host)
    else:
        model_name = model_name or settings.embedding_model
        logger.info(f"Creating SentenceTransformer embedding model: {model_name}")
        return SentenceTransformerEmbedding(model_name=model_name)


def create_embedding_model(
    use_openai: bool = False,
    use_ollama: bool = False,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    ollama_host: Optional[str] = None,
    **kwargs
) -> EmbeddingModel:
    """
    Create a new embedding model instance (not cached).

    Use this when you need a fresh instance with custom configuration.

    Args:
        use_openai: Whether to use OpenAI embeddings
        use_ollama: Whether to use Ollama embeddings
        model_name: Model name to use
        api_key: API key for OpenAI (if applicable)
        ollama_host: Ollama server host URL
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        EmbeddingModel instance
    """
    if use_openai:
        model_name = model_name or "text-embedding-3-large"
        return OpenAIEmbedding(model_name=model_name, api_key=api_key, **kwargs)
    elif use_ollama:
        model_name = model_name or "nomic-embed-text"
        return OllamaEmbedding(model_name=model_name, host=ollama_host or "http://localhost:11434", **kwargs)
    else:
        model_name = model_name or "all-mpnet-base-v2"
        return SentenceTransformerEmbedding(model_name=model_name, **kwargs)
