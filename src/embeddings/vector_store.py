"""FAISS-based vector store for resume embedding storage and similarity search."""

import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages FAISS index for efficient vector similarity search.

    Uses IndexFlatIP (inner product) with L2-normalised vectors, which is
    equivalent to cosine similarity search.  Metadata (mapping from FAISS
    integer IDs to external document/resume IDs) is stored separately and
    persisted alongside the index.
    """

    def __init__(self, dimension: int = 384):
        """Initialise the vector store.

        Args:
            dimension: Dimensionality of the embedding vectors.
                       384 for all-MiniLM-L6-v2, 768 for MPNet-base-v2.

        Raises:
            ValueError: If dimension is not a positive integer.
            ImportError: If faiss-cpu is not installed.
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {dimension}")

        try:
            import faiss  # noqa: F401 – validate availability at init time
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for VectorStoreManager. "
                "Install it with: pip install faiss-cpu"
            ) from exc

        self.dimension = dimension
        self._index = self._create_index()

        # Maps FAISS sequential integer ID → external document/resume ID string
        self._id_map: Dict[int, str] = {}
        # Maps external ID → FAISS integer ID (reverse lookup)
        self._reverse_id_map: Dict[str, int] = {}

        logger.info(
            "VectorStoreManager initialised (dimension=%d, index_type=IndexFlatIP)",
            dimension,
        )

    # ------------------------------------------------------------------
    # Index creation
    # ------------------------------------------------------------------

    def _create_index(self):
        """Create a new FAISS IndexFlatIP index."""
        import faiss

        index = faiss.IndexFlatIP(self.dimension)
        return index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Return the number of vectors currently stored in the index."""
        return self._index.ntotal

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: np.ndarray,
    ) -> None:
        """Add embedding vectors to the FAISS index.

        Vectors are L2-normalised before insertion so that inner-product
        search is equivalent to cosine similarity.

        Args:
            ids: External document/resume IDs corresponding to each vector.
            embeddings: 2-D float32 array of shape (n, dimension).

        Raises:
            ValueError: If ids and embeddings lengths differ, if embeddings
                        have wrong dimensionality, or if any id is a duplicate.
        """
        if len(ids) == 0:
            logger.debug("add_embeddings called with empty list – nothing to do")
            return

        embeddings = self._validate_and_prepare(embeddings, expected_n=len(ids))

        # Check for duplicate IDs
        duplicates = [doc_id for doc_id in ids if doc_id in self._reverse_id_map]
        if duplicates:
            raise ValueError(
                f"Duplicate IDs detected (already in index): {duplicates}"
            )

        # Assign sequential FAISS integer IDs
        start_idx = self._index.ntotal
        faiss_ids = list(range(start_idx, start_idx + len(ids)))

        # Update metadata maps
        for faiss_id, doc_id in zip(faiss_ids, ids):
            self._id_map[faiss_id] = doc_id
            self._reverse_id_map[doc_id] = faiss_id

        # Add to index (FAISS expects float32 C-contiguous array)
        self._index.add(embeddings)

        logger.info(
            "Added %d vectors to index (total: %d)", len(ids), self._index.ntotal
        )

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find the top-k most similar vectors to the query.

        Args:
            query_embedding: 1-D or 2-D float array of shape (dimension,) or
                             (1, dimension).
            k: Number of nearest neighbours to return.

        Returns:
            List of (document_id, similarity_score) tuples sorted by
            descending similarity.  May be shorter than k if the index
            contains fewer vectors.

        Raises:
            ValueError: If the index is empty or query has wrong dimension.
        """
        if self._index.ntotal == 0:
            logger.warning("search called on empty index – returning empty results")
            return []

        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        # Normalise and reshape query
        query = self._validate_and_prepare(query_embedding, expected_n=None)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        effective_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query, effective_k)

        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                # FAISS returns -1 for padding when fewer results exist
                continue
            doc_id = self._id_map.get(int(idx))
            if doc_id is not None:
                results.append((doc_id, float(score)))

        logger.debug("search returned %d results (k=%d)", len(results), k)
        return results

    def remove_embedding(self, doc_id: str) -> bool:
        """Remove a single embedding by its external document ID.

        Note: FAISS IndexFlatIP does not support in-place removal.  This
        method rebuilds the index without the specified vector, which is an
        O(n) operation.

        Args:
            doc_id: External document ID to remove.

        Returns:
            True if the ID was found and removed, False otherwise.
        """
        if doc_id not in self._reverse_id_map:
            logger.warning("remove_embedding: ID '%s' not found in index", doc_id)
            return False

        # Collect all vectors except the one to remove
        faiss_id_to_remove = self._reverse_id_map[doc_id]
        remaining_ids = [
            (fid, did)
            for fid, did in self._id_map.items()
            if fid != faiss_id_to_remove
        ]

        if not remaining_ids:
            # Nothing left – reset to empty index
            self._index = self._create_index()
            self._id_map.clear()
            self._reverse_id_map.clear()
            logger.info("Removed last vector; index is now empty")
            return True

        # Reconstruct index from remaining vectors
        import faiss

        all_vectors = faiss.rev_swig_ptr(
            self._index.get_xb(), self._index.ntotal * self.dimension
        ).reshape(self._index.ntotal, self.dimension).copy()

        remaining_faiss_ids = [fid for fid, _ in remaining_ids]
        remaining_doc_ids = [did for _, did in remaining_ids]
        remaining_vectors = all_vectors[remaining_faiss_ids]

        new_index = self._create_index()
        new_index.add(remaining_vectors)

        self._index = new_index
        self._id_map = {
            new_fid: did
            for new_fid, did in enumerate(remaining_doc_ids)
        }
        self._reverse_id_map = {did: new_fid for new_fid, did in self._id_map.items()}

        logger.info("Removed vector for ID '%s'; index size now %d", doc_id, self._index.ntotal)
        return True

    def get_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Retrieve the stored embedding for a given document ID.

        Args:
            doc_id: External document ID.

        Returns:
            The embedding vector as a float32 numpy array, or None if not found.
        """
        if doc_id not in self._reverse_id_map:
            return None

        import faiss

        faiss_id = self._reverse_id_map[doc_id]
        all_vectors = faiss.rev_swig_ptr(
            self._index.get_xb(), self._index.ntotal * self.dimension
        ).reshape(self._index.ntotal, self.dimension)

        return all_vectors[faiss_id].copy()

    def clear(self) -> None:
        """Remove all vectors and metadata from the store."""
        self._index = self._create_index()
        self._id_map.clear()
        self._reverse_id_map.clear()
        logger.info("VectorStoreManager cleared")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str) -> None:
        """Persist the FAISS index and metadata to disk.

        Two files are written:
        - ``<path>.index`` – the FAISS binary index file
        - ``<path>.meta.json`` – JSON file with id_map and dimension

        Args:
            path: Base file path (without extension).

        Raises:
            OSError: If the directory cannot be created or files cannot be written.
        """
        import faiss

        path = str(path)
        index_path = path + ".index"
        meta_path = path + ".meta.json"

        # Ensure parent directory exists
        parent = Path(index_path).parent
        parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, index_path)

        metadata = {
            "dimension": self.dimension,
            "id_map": {str(k): v for k, v in self._id_map.items()},
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        logger.info(
            "Index saved to '%s' (%d vectors)", index_path, self._index.ntotal
        )

    def load_index(self, path: str) -> None:
        """Load a previously saved FAISS index and metadata from disk.

        Args:
            path: Base file path used when saving (without extension).

        Raises:
            FileNotFoundError: If the index or metadata file does not exist.
            ValueError: If the stored dimension does not match self.dimension.
        """
        import faiss

        path = str(path)
        index_path = path + ".index"
        meta_path = path + ".meta.json"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        stored_dim = metadata.get("dimension")
        if stored_dim != self.dimension:
            raise ValueError(
                f"Dimension mismatch: index was saved with dimension={stored_dim}, "
                f"but this VectorStoreManager has dimension={self.dimension}"
            )

        self._index = faiss.read_index(index_path)
        self._id_map = {int(k): v for k, v in metadata["id_map"].items()}
        self._reverse_id_map = {v: int(k) for k, v in metadata["id_map"].items()}

        logger.info(
            "Index loaded from '%s' (%d vectors)", index_path, self._index.ntotal
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_and_prepare(
        self,
        embeddings: np.ndarray,
        expected_n: Optional[int],
    ) -> np.ndarray:
        """Validate shape/dtype and L2-normalise embeddings.

        Args:
            embeddings: Input array (1-D or 2-D).
            expected_n: Expected number of vectors (None to skip check).

        Returns:
            Float32 C-contiguous 2-D array with unit-norm rows.

        Raises:
            ValueError: On shape or dimension mismatch.
        """
        embeddings = np.array(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 1-D or 2-D, got shape {embeddings.shape}"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        if expected_n is not None and embeddings.shape[0] != expected_n:
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) does not match "
                f"number of IDs ({expected_n})"
            )

        # L2-normalise each row so inner product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def get_stats(self) -> Dict:
        """Return basic statistics about the current index state."""
        return {
            "dimension": self.dimension,
            "total_vectors": self._index.ntotal,
            "index_type": type(self._index).__name__,
        }
