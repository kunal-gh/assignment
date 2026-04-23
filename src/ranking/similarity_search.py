"""Higher-level similarity search pipeline integrating EmbeddingGenerator and VectorStoreManager."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..embeddings.embedding_generator import EmbeddingGenerator
from ..embeddings.vector_store import VectorStoreManager
from ..models.job import JobDescription
from ..models.resume import ResumeData

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single similarity search result."""

    candidate_id: str
    resume: ResumeData
    similarity_score: float  # cosine similarity in [0, 1]
    rank: int


class SimilaritySearchEngine:
    """End-to-end similarity search pipeline.

    Combines :class:`EmbeddingGenerator` and :class:`VectorStoreManager` to
    provide a single interface that:

    1. Generates embeddings for a job description and a list of resumes.
    2. Adds resume embeddings to a fresh FAISS index.
    3. Searches for the top-k resumes most similar to the job description.
    4. Returns :class:`SearchResult` objects with cosine similarity scores.

    The engine creates a *new* FAISS index for every :meth:`search` call so
    that results are always consistent with the resumes passed in.  If you
    need to reuse an index across calls, use :class:`VectorStoreManager`
    directly.
    """

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialise the engine.

        Args:
            embedding_generator: Pre-initialised generator.  A new one is
                created with *model_name* when ``None``.
            model_name: Sentence-transformer model to use when creating a
                default generator.
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator(model_name=model_name)
        logger.info(
            "SimilaritySearchEngine initialised (model=%s)",
            self.embedding_generator.model_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        job_description: JobDescription,
        resumes: List[ResumeData],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Find the top-k resumes most similar to *job_description*.

        Steps performed internally:
        - Generate (or reuse) the job-description embedding.
        - Generate (or reuse) embeddings for every resume.
        - Build a temporary FAISS index from the resume embeddings.
        - Query the index with the job-description embedding.
        - Return ranked :class:`SearchResult` objects.

        Args:
            job_description: The job to match against.
            resumes: Candidate resumes to rank.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`SearchResult` sorted by descending similarity
            score.  May be shorter than *top_k* when fewer resumes are
            provided.  Returns an empty list when *resumes* is empty.

        Raises:
            ValueError: If *top_k* is not a positive integer.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if not resumes:
            logger.warning("search called with empty resume list – returning []")
            return []

        # 1. Ensure job-description embedding exists
        job_embedding = self._get_job_embedding(job_description)

        # 2. Ensure resume embeddings exist (batch for efficiency)
        self._ensure_resume_embeddings(resumes)

        # 3. Build a fresh FAISS index for this search
        dimension = self.embedding_generator.embedding_dimension
        store = VectorStoreManager(dimension=dimension)

        ids = [r.candidate_id for r in resumes]
        embeddings = np.stack([r.embedding for r in resumes]).astype(np.float32)
        store.add_embeddings(ids, embeddings)

        # 4. Query the index
        effective_k = min(top_k, len(resumes))
        raw_results: List[Tuple[str, float]] = store.search(job_embedding, k=effective_k)

        if not raw_results:
            logger.warning("FAISS search returned no results")
            return []

        # 5. Build a lookup map and assemble SearchResult objects
        resume_map = {r.candidate_id: r for r in resumes}
        results: List[SearchResult] = []

        for rank, (candidate_id, score) in enumerate(raw_results, start=1):
            resume = resume_map.get(candidate_id)
            if resume is None:
                logger.warning("Unknown candidate_id '%s' in search results", candidate_id)
                continue

            # Clamp score to [0, 1] – FAISS inner-product on L2-normalised
            # vectors is cosine similarity and should already be in this range,
            # but we guard against floating-point edge cases.
            clamped_score = float(max(0.0, min(1.0, score)))

            results.append(
                SearchResult(
                    candidate_id=candidate_id,
                    resume=resume,
                    similarity_score=clamped_score,
                    rank=rank,
                )
            )

        logger.info(
            "search returned %d results (top_k=%d, total_resumes=%d)",
            len(results),
            top_k,
            len(resumes),
        )
        return results

    def search_by_text(
        self,
        query_text: str,
        resumes: List[ResumeData],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Convenience method: search using raw text instead of a JobDescription.

        Args:
            query_text: Free-form text describing the ideal candidate.
            resumes: Candidate resumes to rank.
            top_k: Maximum number of results to return.

        Returns:
            Same as :meth:`search`.
        """
        job_desc = JobDescription(title="Query", description=query_text)
        return self.search(job_desc, resumes, top_k=top_k)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_job_embedding(self, job_description: JobDescription) -> np.ndarray:
        """Return (and cache on the object) the job-description embedding."""
        if job_description.embedding is None:
            job_description.embedding = self.embedding_generator.encode_job_description(job_description)
        return job_description.embedding

    def _ensure_resume_embeddings(self, resumes: List[ResumeData]) -> None:
        """Generate embeddings for any resume that does not yet have one."""
        missing = [r for r in resumes if r.embedding is None]
        if missing:
            logger.info("Generating embeddings for %d resumes", len(missing))
            self.embedding_generator.batch_encode_resumes(missing)
