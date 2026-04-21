"""Model management for sentence transformers."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and caching of sentence transformer models."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model manager.

        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "sentence-transformers"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_models = {}
        self._model_info = {}

        logger.info(f"ModelManager initialized with cache dir: {self.cache_dir}")

    def load_model(self, model_name: str, force_reload: bool = False):
        """
        Load a sentence transformer model.

        Args:
            model_name: Name of the model to load
            force_reload: Whether to force reload even if already loaded

        Returns:
            Loaded sentence transformer model
        """
        # Return cached model if available
        if not force_reload and model_name in self._loaded_models:
            logger.debug(f"Returning cached model: {model_name}")
            return self._loaded_models[model_name]

        try:
            logger.info(f"Loading sentence transformer model: {model_name}")

            # Try to import sentence_transformers
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers library not found. " "Install it with: pip install sentence-transformers"
                )

            # Load model with caching
            model = SentenceTransformer(model_name, cache_folder=str(self.cache_dir))

            # Cache the loaded model
            self._loaded_models[model_name] = model

            # Store model info
            self._model_info[model_name] = self._get_model_info(model)

            logger.info(f"Successfully loaded model: {model_name}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")

            # Try fallback models
            fallback_models = self._get_fallback_models(model_name)

            for fallback in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback}")
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(fallback, cache_folder=str(self.cache_dir))

                    self._loaded_models[model_name] = model  # Cache under original name
                    self._model_info[model_name] = self._get_model_info(model)

                    logger.warning(f"Using fallback model {fallback} for {model_name}")
                    return model

                except Exception as fallback_error:
                    logger.debug(f"Fallback model {fallback} also failed: {str(fallback_error)}")
                    continue

            # If all models fail, raise the original error
            raise RuntimeError(f"Could not load model {model_name} or any fallbacks: {str(e)}")

    def _get_model_info(self, model) -> Dict[str, Any]:
        """Extract information about a loaded model."""
        try:
            # Get basic model info
            info = {
                "model_name": getattr(model, "_model_name", "unknown"),
                "max_seq_length": getattr(model, "max_seq_length", 512),
                "embedding_dimension": None,
            }

            # Try to get embedding dimension
            try:
                test_embedding = model.encode("test", convert_to_tensor=False)
                info["embedding_dimension"] = len(test_embedding)
            except Exception:
                info["embedding_dimension"] = "unknown"

            # Try to get tokenizer info
            if hasattr(model, "tokenizer"):
                info["vocab_size"] = getattr(model.tokenizer, "vocab_size", "unknown")
                info["model_max_length"] = getattr(model.tokenizer, "model_max_length", "unknown")

            return info

        except Exception as e:
            logger.debug(f"Could not extract model info: {str(e)}")
            return {"error": str(e)}

    def _get_fallback_models(self, original_model: str) -> list:
        """Get list of fallback models to try if the original fails."""
        # Define fallback chains for common models
        fallback_chains = {
            "all-MiniLM-L6-v2": ["paraphrase-MiniLM-L6-v2", "all-MiniLM-L12-v2"],
            "all-mpnet-base-v2": ["all-MiniLM-L6-v2", "paraphrase-mpnet-base-v2"],
            "multi-qa-MiniLM-L6-cos-v1": ["all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-dot-v1"],
            "paraphrase-multilingual-MiniLM-L12-v2": ["paraphrase-MiniLM-L6-v2", "all-MiniLM-L6-v2"],
        }

        # Get specific fallbacks or use general ones
        fallbacks = fallback_chains.get(original_model, [])

        # Add general fallbacks
        general_fallbacks = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"]

        # Combine and deduplicate
        all_fallbacks = fallbacks + [fb for fb in general_fallbacks if fb not in fallbacks]

        # Remove the original model from fallbacks
        return [fb for fb in all_fallbacks if fb != original_model]

    def list_available_models(self) -> list:
        """List commonly available sentence transformer models."""
        return [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1",
            "paraphrase-MiniLM-L6-v2",
            "paraphrase-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "distilbert-base-nli-stsb-mean-tokens",
            "roberta-large-nli-stsb-mean-tokens",
        ]

    def get_model_recommendations(self, use_case: str = "general") -> Dict[str, str]:
        """Get model recommendations for different use cases."""
        recommendations = {
            "general": {"fast": "all-MiniLM-L6-v2", "balanced": "all-mpnet-base-v2", "quality": "paraphrase-mpnet-base-v2"},
            "multilingual": {
                "fast": "paraphrase-multilingual-MiniLM-L12-v2",
                "quality": "paraphrase-multilingual-mpnet-base-v2",
            },
            "qa": {"fast": "multi-qa-MiniLM-L6-cos-v1", "quality": "multi-qa-mpnet-base-cos-v1"},
            "semantic_search": {"fast": "all-MiniLM-L6-v2", "quality": "all-mpnet-base-v2"},
        }

        return recommendations.get(use_case, recommendations["general"])

    def get_loaded_models(self) -> Dict[str, Any]:
        """Get information about currently loaded models."""
        return {model_name: self._model_info.get(model_name, {}) for model_name in self._loaded_models.keys()}

    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            if model_name in self._model_info:
                del self._model_info[model_name]
            logger.info(f"Unloaded model: {model_name}")
        else:
            logger.warning(f"Model {model_name} was not loaded")

    def clear_cache(self):
        """Clear all loaded models from memory."""
        self._loaded_models.clear()
        self._model_info.clear()
        logger.info("Cleared all loaded models from memory")

    def get_cache_size(self) -> Dict[str, Any]:
        """Get information about cache usage."""
        try:
            cache_size = 0
            file_count = 0

            if self.cache_dir.exists():
                for file_path in self.cache_dir.rglob("*"):
                    if file_path.is_file():
                        cache_size += file_path.stat().st_size
                        file_count += 1

            return {
                "cache_dir": str(self.cache_dir),
                "total_size_bytes": cache_size,
                "total_size_mb": round(cache_size / (1024 * 1024), 2),
                "file_count": file_count,
                "loaded_models": len(self._loaded_models),
            }

        except Exception as e:
            logger.error(f"Error getting cache size: {str(e)}")
            return {"error": str(e)}

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if a model name is likely to be valid."""
        # Basic validation - check if it looks like a valid model name
        if not model_name or not isinstance(model_name, str):
            return False

        # Check against known models
        known_models = self.list_available_models()
        if model_name in known_models:
            return True

        # Check if it follows common naming patterns
        valid_patterns = [
            r"^[a-zA-Z0-9\-_]+$",  # Basic alphanumeric with hyphens/underscores
            r"^[a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+$",  # HuggingFace format (user/model)
        ]

        import re

        for pattern in valid_patterns:
            if re.match(pattern, model_name):
                return True

        return False
