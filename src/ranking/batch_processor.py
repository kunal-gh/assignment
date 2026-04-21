"""Batch processing capabilities for large-scale resume screening."""

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from ..embeddings.embedding_generator import EmbeddingGenerator
from ..models.job import JobDescription
from ..models.resume import ResumeData
from .similarity_search import SearchResult, SimilaritySearchEngine

logger = logging.getLogger(__name__)

# Default chunk size for batch processing
DEFAULT_BATCH_CHUNK_SIZE = 50


@dataclass
class BatchResult:
    """Aggregated results from a batch processing run."""

    results: List[SearchResult] = field(default_factory=list)
    total_processed: int = 0
    total_failed: int = 0
    failed_ids: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of resumes processed successfully (0.0–1.0)."""
        total = self.total_processed + self.total_failed
        return self.total_processed / total if total > 0 else 0.0


class BatchProcessor:
    """Memory-efficient batch processor for large resume collections.

    Processes resumes in configurable chunks so that memory usage stays
    bounded regardless of the total number of resumes.  Each chunk is
    embedded and searched independently; results are aggregated and
    re-ranked at the end.

    Args:
        embedding_generator: Pre-initialised :class:`EmbeddingGenerator`.
            A new one is created with *model_name* when ``None``.
        model_name: Sentence-transformer model to use when creating a
            default generator.
        chunk_size: Number of resumes to process per chunk.  Smaller
            values reduce peak memory usage at the cost of slightly more
            overhead per chunk.
    """

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = DEFAULT_BATCH_CHUNK_SIZE,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")

        self.chunk_size = chunk_size
        self.embedding_generator = embedding_generator or EmbeddingGenerator(model_name=model_name)
        self._search_engine = SimilaritySearchEngine(embedding_generator=self.embedding_generator)

        logger.info(
            "BatchProcessor initialised (model=%s, chunk_size=%d)",
            self.embedding_generator.model_name,
            chunk_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_batch(
        self,
        job_description: JobDescription,
        resumes: List[ResumeData],
        top_k: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """Process a large list of resumes against a job description.

        Resumes are processed in chunks of :attr:`chunk_size`.  Failed
        resumes (those that raise an exception during embedding or search)
        are skipped and recorded in :attr:`BatchResult.failed_ids`.

        Args:
            job_description: The job to match against.
            resumes: Full list of candidate resumes.
            top_k: Number of top results to return in the final ranking.
            progress_callback: Optional callable invoked after each chunk
                with ``(processed_so_far, total)`` arguments.

        Returns:
            :class:`BatchResult` with aggregated, re-ranked results.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if not resumes:
            logger.warning("process_batch called with empty resume list")
            return BatchResult()

        total = len(resumes)
        logger.info(
            "Starting batch processing: %d resumes, chunk_size=%d, top_k=%d",
            total,
            self.chunk_size,
            top_k,
        )

        # Pre-compute the job embedding once so every chunk reuses it.
        job_embedding = self._get_job_embedding(job_description)

        all_results: List[SearchResult] = []
        failed_ids: List[str] = []
        processed = 0

        chunks = self._split_into_chunks(resumes, self.chunk_size)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_results, chunk_failed = self._process_chunk(
                job_description=job_description,
                job_embedding=job_embedding,
                chunk=chunk,
                chunk_idx=chunk_idx,
            )
            all_results.extend(chunk_results)
            failed_ids.extend(chunk_failed)
            processed += len(chunk) - len(chunk_failed)

            if progress_callback is not None:
                try:
                    progress_callback(processed + len(failed_ids), total)
                except Exception as cb_exc:  # pragma: no cover
                    logger.warning("progress_callback raised an exception: %s", cb_exc)

        # Re-rank all collected results and keep top_k
        all_results.sort(key=lambda r: r.similarity_score, reverse=True)
        top_results = all_results[:top_k]

        # Re-assign ranks after global sort
        for rank, result in enumerate(top_results, start=1):
            result.rank = rank

        batch_result = BatchResult(
            results=top_results,
            total_processed=processed,
            total_failed=len(failed_ids),
            failed_ids=failed_ids,
        )

        logger.info(
            "Batch processing complete: %d processed, %d failed, %d results returned",
            processed,
            len(failed_ids),
            len(top_results),
        )
        return batch_result

    def generate_batch_embeddings(
        self,
        resumes: List[ResumeData],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Pre-generate embeddings for all resumes in memory-efficient chunks.

        Useful when you want to pre-compute embeddings before calling
        :meth:`process_batch`, or when you need embeddings for other
        downstream tasks.

        Args:
            resumes: Resumes to embed.  Resumes that already have an
                embedding are skipped.
            progress_callback: Optional callable invoked after each chunk
                with ``(processed_so_far, total)`` arguments.

        Returns:
            Number of embeddings successfully generated.
        """
        missing = [r for r in resumes if r.embedding is None]
        if not missing:
            logger.debug("generate_batch_embeddings: all embeddings already present")
            return 0

        total = len(missing)
        generated = 0
        chunks = self._split_into_chunks(missing, self.chunk_size)

        for chunk in chunks:
            try:
                self.embedding_generator.batch_encode_resumes(chunk)
                generated += len(chunk)
            except Exception as exc:
                logger.error("Embedding chunk failed: %s", exc)
                # Mark individual failures as zero vectors so downstream
                # code can still run.
                for resume in chunk:
                    if resume.embedding is None:
                        resume.embedding = np.zeros(
                            self.embedding_generator.embedding_dimension,
                            dtype=np.float32,
                        )

            if progress_callback is not None:
                try:
                    progress_callback(generated, total)
                except Exception as cb_exc:  # pragma: no cover
                    logger.warning("progress_callback raised an exception: %s", cb_exc)

        return generated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_chunks(items: List, chunk_size: int) -> List[List]:
        """Split *items* into consecutive sub-lists of at most *chunk_size*."""
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def _get_job_embedding(self, job_description: JobDescription) -> np.ndarray:
        """Return (and cache on the object) the job-description embedding."""
        if job_description.embedding is None:
            job_description.embedding = self.embedding_generator.encode_job_description(job_description)
        return job_description.embedding

    def _process_chunk(
        self,
        job_description: JobDescription,
        job_embedding: np.ndarray,
        chunk: List[ResumeData],
        chunk_idx: int,
    ):
        """Process a single chunk of resumes.

        Returns:
            Tuple of (list of SearchResult, list of failed candidate_ids).
        """
        results: List[SearchResult] = []
        failed_ids: List[str] = []

        # Generate embeddings for resumes that don't have one yet.
        missing = [r for r in chunk if r.embedding is None]
        if missing:
            try:
                self.embedding_generator.batch_encode_resumes(missing)
            except Exception as exc:
                logger.error(
                    "Chunk %d: batch embedding failed for %d resumes: %s",
                    chunk_idx,
                    len(missing),
                    exc,
                )
                # Fall back to individual encoding; mark failures.
                for resume in missing:
                    if resume.embedding is None:
                        try:
                            resume.embedding = self.embedding_generator.encode_resume(resume)
                        except Exception as inner_exc:
                            logger.error(
                                "Chunk %d: individual embedding failed for '%s': %s",
                                chunk_idx,
                                resume.candidate_id,
                                inner_exc,
                            )
                            failed_ids.append(resume.candidate_id)

        # Search within this chunk.
        valid_resumes = [r for r in chunk if r.embedding is not None and r.candidate_id not in failed_ids]

        if not valid_resumes:
            logger.warning("Chunk %d: no valid resumes to search", chunk_idx)
            return results, failed_ids

        try:
            chunk_results = self._search_engine.search(
                job_description=job_description,
                resumes=valid_resumes,
                top_k=len(valid_resumes),
            )
            results.extend(chunk_results)
        except Exception as exc:
            logger.error("Chunk %d: search failed: %s", chunk_idx, exc)
            # Mark all resumes in this chunk as failed.
            for resume in valid_resumes:
                failed_ids.append(resume.candidate_id)

        logger.debug(
            "Chunk %d: %d results, %d failed",
            chunk_idx,
            len(results),
            len(failed_ids),
        )
        return results, failed_ids
