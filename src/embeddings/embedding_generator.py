"""Embedding generation using sentence transformers."""

import logging
import hashlib
import re
import string
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
from pathlib import Path

from ..models.resume import ResumeData, Experience, Education
from ..models.job import JobDescription
from .model_manager import ModelManager
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 200  # tokens (approximate)
DEFAULT_CHUNK_OVERLAP = 50  # tokens overlap between chunks


class EmbeddingGenerator:
    """Generates semantic embeddings for resumes and job descriptions."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory for caching embeddings
            use_cache: Whether to use embedding cache
            chunk_size: Approximate token count per chunk for long texts
            chunk_overlap: Approximate token overlap between consecutive chunks
        """
        self.model_name = model_name
        self.use_cache = use_cache
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize components
        self.model_manager = ModelManager()
        self.cache_manager = CacheManager(cache_dir) if use_cache else None

        # Load model
        self.model = None
        self.embedding_dimension = None
        self._load_model()

        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = self.model_manager.load_model(self.model_name)

            # Get embedding dimension
            test_embedding = self.model.encode("test", convert_to_tensor=False)
            self.embedding_dimension = len(test_embedding)

            logger.info(
                f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}"
            )

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise RuntimeError(f"Could not load embedding model: {str(e)}")

    # ------------------------------------------------------------------
    # Public encoding API
    # ------------------------------------------------------------------

    def encode_resume(self, resume_data: ResumeData) -> np.ndarray:
        """
        Generate embedding vector for a resume.

        Combines all resume sections into a coherent text representation,
        applies preprocessing, and generates a single embedding (with
        chunked mean-pooling for long texts).

        Args:
            resume_data: Resume data object

        Returns:
            Numpy array containing the embedding vector
        """
        text = self._build_resume_text(resume_data)

        # Check cache first
        if self.use_cache and self.cache_manager:
            cache_key = self._generate_cache_key(text, "resume")
            cached = self.cache_manager.get_embedding(cache_key)
            if cached is not None:
                logger.debug(f"Retrieved resume embedding from cache: {cache_key}")
                resume_data.embedding = cached
                return cached

        embedding = self._generate_embedding(text)

        if self.use_cache and self.cache_manager:
            self.cache_manager.store_embedding(cache_key, embedding)

        resume_data.embedding = embedding
        return embedding

    def encode_job_description(self, job_desc: JobDescription) -> np.ndarray:
        """
        Generate embedding vector for a job description.

        Args:
            job_desc: Job description object

        Returns:
            Numpy array containing the embedding vector
        """
        text = job_desc.get_combined_text()

        if self.use_cache and self.cache_manager:
            cache_key = self._generate_cache_key(text, "job")
            cached = self.cache_manager.get_embedding(cache_key)
            if cached is not None:
                logger.debug(f"Retrieved job embedding from cache: {cache_key}")
                job_desc.embedding = cached
                return cached

        embedding = self._generate_embedding(text)

        if self.use_cache and self.cache_manager:
            self.cache_manager.store_embedding(cache_key, embedding)

        job_desc.embedding = embedding
        return embedding

    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for arbitrary text.

        Args:
            text: Input text

        Returns:
            Numpy array containing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dimension)

        return self._generate_embedding(text)

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts

        Returns:
            2D numpy array where each row is an embedding
        """
        if not texts:
            return np.empty((0, self.embedding_dimension))

        valid_texts = [t for t in texts if t and t.strip()]

        if not valid_texts:
            logger.warning("No valid texts provided for batch encoding")
            return np.zeros((len(texts), self.embedding_dimension))

        try:
            logger.info(f"Batch encoding {len(valid_texts)} texts")

            embeddings = self.model.encode(
                valid_texts,
                convert_to_tensor=False,
                show_progress_bar=len(valid_texts) > 10,
                batch_size=32,
                normalize_embeddings=True,
            )

            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)

            logger.info(f"Batch encoding completed. Shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Batch encoding failed: {str(e)}")
            return np.zeros((len(texts), self.embedding_dimension))

    def batch_encode_resumes(self, resumes: List[ResumeData]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple resumes efficiently.

        Args:
            resumes: List of resume data objects

        Returns:
            List of embedding arrays
        """
        if not resumes:
            return []

        texts = [self._build_resume_text(r) for r in resumes]
        embeddings = self.batch_encode(texts)

        for resume, embedding in zip(resumes, embeddings):
            resume.embedding = embedding

        return [embeddings[i] for i in range(len(resumes))]

    # ------------------------------------------------------------------
    # Resume text construction
    # ------------------------------------------------------------------

    def _build_resume_text(self, resume_data: ResumeData) -> str:
        """
        Combine resume sections into a coherent text representation.

        Sections are ordered by semantic importance:
        summary/raw_text → skills → experience → education.
        Each section is prefixed with a label so the model can
        distinguish between different types of information.

        Args:
            resume_data: Parsed resume data

        Returns:
            Single cleaned string ready for embedding
        """
        parts: List[str] = []

        # 1. Professional summary / raw text snippet
        if resume_data.raw_text:
            summary = self._preprocess_text(resume_data.raw_text)
            if summary:
                parts.append(f"Summary: {summary}")

        # 2. Skills
        if resume_data.skills:
            skills_text = ", ".join(resume_data.skills)
            parts.append(f"Skills: {skills_text}")

        # 3. Work experience
        for exp in resume_data.experience:
            exp_text = self._format_experience(exp)
            if exp_text:
                parts.append(exp_text)

        # 4. Education
        for edu in resume_data.education:
            edu_text = self._format_education(edu)
            if edu_text:
                parts.append(edu_text)

        combined = " | ".join(parts)
        return self._preprocess_text(combined)

    def _format_experience(self, exp: Experience) -> str:
        """Format a single experience entry into a text snippet."""
        tokens: List[str] = [f"Experience: {exp.title} at {exp.company}"]

        if exp.start_date or exp.end_date:
            period = f"{exp.start_date or ''} - {'Present' if exp.is_current else (exp.end_date or '')}"
            tokens.append(period.strip(" -"))

        if exp.description:
            tokens.append(self._preprocess_text(exp.description))

        if exp.skills_used:
            tokens.append(f"Technologies: {', '.join(exp.skills_used)}")

        return " ".join(t for t in tokens if t)

    def _format_education(self, edu: Education) -> str:
        """Format a single education entry into a text snippet."""
        tokens: List[str] = [f"Education: {edu.degree}"]

        if edu.institution:
            tokens.append(f"at {edu.institution}")

        if edu.major:
            tokens.append(f"in {edu.major}")

        if edu.graduation_date:
            tokens.append(f"({edu.graduation_date})")

        if edu.relevant_coursework:
            tokens.append(f"Coursework: {', '.join(edu.relevant_coursework)}")

        return " ".join(t for t in tokens if t)

    # ------------------------------------------------------------------
    # Text preprocessing
    # ------------------------------------------------------------------

    def _preprocess_text(self, text: str) -> str:
        """
        Clean and normalise text before embedding.

        Steps:
        - Strip leading/trailing whitespace
        - Collapse multiple whitespace characters into a single space
        - Remove non-printable / control characters
        - Normalise common unicode punctuation to ASCII equivalents
        - Remove URLs (they add noise without semantic value)

        Args:
            text: Raw text

        Returns:
            Cleaned text string
        """
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Normalise unicode dashes and quotes to ASCII
        text = (
            text.replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
        )

        # Remove non-printable / control characters (keep newlines temporarily)
        text = re.sub(r"[^\x20-\x7E\n]", " ", text)

        # Replace newlines / tabs with spaces
        text = re.sub(r"[\n\r\t]+", " ", text)

        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        return text.strip()

    # ------------------------------------------------------------------
    # Chunking for long texts
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split long text into overlapping chunks using a sliding window.

        The window is word-based (words ≈ tokens for English text).
        Each chunk contains at most `chunk_size` words; consecutive
        chunks share `chunk_overlap` words.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks; returns [text] if no chunking needed
        """
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += self.chunk_size - self.chunk_overlap

        logger.debug(f"Text split into {len(chunks)} chunks (word count: {len(words)})")
        return chunks

    def _embed_with_chunking(self, text: str) -> np.ndarray:
        """
        Generate a single embedding for potentially long text.

        If the text fits within the chunk size, a single embedding is
        returned directly.  For longer texts the text is split into
        overlapping chunks, each chunk is embedded, and the final
        embedding is the mean of all chunk embeddings (re-normalised).

        Args:
            text: Preprocessed input text

        Returns:
            Single embedding numpy array
        """
        chunks = self._chunk_text(text)

        if len(chunks) == 1:
            return self._encode_single(chunks[0])

        # Encode all chunks in one batch call for efficiency
        try:
            chunk_embeddings = self.model.encode(
                chunks,
                convert_to_tensor=False,
                normalize_embeddings=True,
                batch_size=32,
            )
            if not isinstance(chunk_embeddings, np.ndarray):
                chunk_embeddings = np.array(chunk_embeddings)

            # Mean pool across chunks then re-normalise
            mean_embedding = chunk_embeddings.mean(axis=0)
            norm = np.linalg.norm(mean_embedding)
            if norm > 0:
                mean_embedding = mean_embedding / norm

            return mean_embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Chunked embedding failed: {str(e)}")
            return np.zeros(self.embedding_dimension, dtype=np.float32)

    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string (no chunking)."""
        try:
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True,
            )
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Single encode failed: {str(e)}")
            return np.zeros(self.embedding_dimension, dtype=np.float32)

    # ------------------------------------------------------------------
    # Core embedding generation
    # ------------------------------------------------------------------

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text, handling long inputs via chunking.

        Args:
            text: Preprocessed input text

        Returns:
            Numpy array embedding
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return np.zeros(self.embedding_dimension, dtype=np.float32)

        embedding = self._embed_with_chunking(text)

        # Validate
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            logger.error("Generated embedding contains NaN or Inf values")
            return np.zeros(self.embedding_dimension, dtype=np.float32)

        return embedding

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _generate_cache_key(self, text: str, prefix: str = "") -> str:
        """Generate a stable cache key for a text + model combination."""
        content = f"{self.model_name}:{text}"
        hash_key = hashlib.md5(content.encode("utf-8")).hexdigest()
        return f"{prefix}_{hash_key}" if prefix else hash_key

    def _get_max_sequence_length(self) -> int:
        """Return approximate character limit for the current model."""
        model_token_limits = {
            "all-MiniLM-L6-v2": 256,
            "all-mpnet-base-v2": 384,
            "multi-qa-MiniLM-L6-cos-v1": 512,
            "paraphrase-multilingual-MiniLM-L12-v2": 128,
        }
        token_limit = model_token_limits.get(self.model_name, 256)
        return token_limit * 4  # rough chars-per-token estimate

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the current model configuration."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": self._get_max_sequence_length(),
            "cache_enabled": self.use_cache,
            "model_loaded": self.model is not None,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache_manager:
            self.cache_manager.clear_cache()
            logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        if self.cache_manager:
            return self.cache_manager.get_stats()
        return {"cache_enabled": False}

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score clamped to [0, 1]
        """
        try:
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)

            if embedding1.shape != embedding2.shape:
                logger.error(
                    f"Embedding dimension mismatch: {embedding1.shape} vs {embedding2.shape}"
                )
                return 0.0

            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
