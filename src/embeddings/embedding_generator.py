"""Embedding generation using sentence transformers."""

import logging
import hashlib
from typing import List, Optional, Union, Dict, Any
import numpy as np
from pathlib import Path

from ..models.resume import ResumeData
from ..models.job import JobDescription
from .model_manager import ModelManager
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates semantic embeddings for resumes and job descriptions."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory for caching embeddings
            use_cache: Whether to use embedding cache
        """
        self.model_name = model_name
        self.use_cache = use_cache
        
        # Initialize components
        self.model_manager = ModelManager()
        self.cache_manager = CacheManager(cache_dir) if use_cache else None
        
        # Load model
        self.model = None
        self.embedding_dimension = None
        self._load_model()
        
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = self.model_manager.load_model(self.model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode("test", convert_to_tensor=False)
            self.embedding_dimension = len(test_embedding)
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise RuntimeError(f"Could not load embedding model: {str(e)}")
    
    def encode_resume(self, resume_data: ResumeData) -> np.ndarray:
        """
        Generate embedding vector for resume.
        
        Args:
            resume_data: Resume data object
            
        Returns:
            Numpy array containing the embedding vector
        """
        # Get combined text for embedding
        text = resume_data.get_combined_text()
        
        # Check cache first
        if self.use_cache and self.cache_manager:
            cache_key = self._generate_cache_key(text, "resume")
            cached_embedding = self.cache_manager.get_embedding(cache_key)
            
            if cached_embedding is not None:
                logger.debug(f"Retrieved resume embedding from cache: {cache_key}")
                return cached_embedding
        
        # Generate embedding
        embedding = self._generate_embedding(text)
        
        # Cache the result
        if self.use_cache and self.cache_manager:
            self.cache_manager.store_embedding(cache_key, embedding)
            logger.debug(f"Stored resume embedding in cache: {cache_key}")
        
        # Store embedding in resume data
        resume_data.embedding = embedding
        
        return embedding
    
    def encode_job_description(self, job_desc: JobDescription) -> np.ndarray:
        """
        Generate embedding vector for job description.
        
        Args:
            job_desc: Job description object
            
        Returns:
            Numpy array containing the embedding vector
        """
        # Get combined text for embedding
        text = job_desc.get_combined_text()
        
        # Check cache first
        if self.use_cache and self.cache_manager:
            cache_key = self._generate_cache_key(text, "job")
            cached_embedding = self.cache_manager.get_embedding(cache_key)
            
            if cached_embedding is not None:
                logger.debug(f"Retrieved job embedding from cache: {cache_key}")
                return cached_embedding
        
        # Generate embedding
        embedding = self._generate_embedding(text)
        
        # Cache the result
        if self.use_cache and self.cache_manager:
            self.cache_manager.store_embedding(cache_key, embedding)
            logger.debug(f"Stored job embedding in cache: {cache_key}")
        
        # Store embedding in job description
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
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        
        if not valid_texts:
            logger.warning("No valid texts provided for batch encoding")
            return np.zeros((len(texts), self.embedding_dimension))
        
        try:
            logger.info(f"Batch encoding {len(valid_texts)} texts")
            
            # Use model's batch encoding for efficiency
            embeddings = self.model.encode(
                valid_texts,
                convert_to_tensor=False,
                show_progress_bar=len(valid_texts) > 10,
                batch_size=32
            )
            
            # Ensure we have the right shape
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            logger.info(f"Batch encoding completed. Shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch encoding failed: {str(e)}")
            # Return zero embeddings as fallback
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
        
        # Extract texts
        texts = [resume.get_combined_text() for resume in resumes]
        
        # Generate embeddings in batch
        embeddings = self.batch_encode(texts)
        
        # Store embeddings in resume objects
        for i, (resume, embedding) in enumerate(zip(resumes, embeddings)):
            resume.embedding = embedding
        
        return [embeddings[i] for i in range(len(resumes))]
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return np.zeros(self.embedding_dimension)
        
        try:
            # Truncate text if too long (model-specific limits)
            max_length = self._get_max_sequence_length()
            if len(text) > max_length:
                text = text[:max_length]
                logger.debug(f"Text truncated to {max_length} characters")
            
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True  # L2 normalize for cosine similarity
            )
            
            # Ensure it's a numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Validate embedding
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                logger.error("Generated embedding contains NaN or Inf values")
                return np.zeros(self.embedding_dimension)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(self.embedding_dimension)
    
    def _generate_cache_key(self, text: str, prefix: str = "") -> str:
        """Generate cache key for text."""
        # Create hash of text + model name for uniqueness
        content = f"{self.model_name}:{text}"
        hash_obj = hashlib.md5(content.encode('utf-8'))
        hash_key = hash_obj.hexdigest()
        
        return f"{prefix}_{hash_key}" if prefix else hash_key
    
    def _get_max_sequence_length(self) -> int:
        """Get maximum sequence length for the model."""
        # Common limits for sentence transformer models
        model_limits = {
            'all-MiniLM-L6-v2': 256,
            'all-mpnet-base-v2': 384,
            'multi-qa-MiniLM-L6-cos-v1': 512,
            'paraphrase-multilingual-MiniLM-L12-v2': 128
        }
        
        # Convert to approximate character count (rough estimate: 4 chars per token)
        token_limit = model_limits.get(self.model_name, 256)
        return token_limit * 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'max_sequence_length': self._get_max_sequence_length(),
            'cache_enabled': self.use_cache,
            'model_loaded': self.model is not None
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache_manager:
            self.cache_manager.clear_cache()
            logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache_manager:
            return self.cache_manager.get_stats()
        return {'cache_enabled': False}
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1 for normalized embeddings)
        """
        try:
            # Ensure embeddings are numpy arrays
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)
            
            # Check dimensions match
            if embedding1.shape != embedding2.shape:
                logger.error(f"Embedding dimension mismatch: {embedding1.shape} vs {embedding2.shape}")
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Clamp to [0, 1] range (for normalized embeddings, this should already be the case)
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0