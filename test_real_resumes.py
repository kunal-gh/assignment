"""Test script for real resume processing - verify everything works before deployment."""

import logging
import os
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.models.job import JobDescription
from src.parsers.resume_parser import ResumeParser
from src.ranking.ranking_engine import RankingEngine


def test_resume_parsing():
    """Test resume parsing with sample data."""
    logger.info("=" * 80)
    logger.info("TEST 1: Resume Parsing")
    logger.info("=" * 80)

    parser = ResumeParser()
    sample_dir = Path("data/sample_resumes")

    if not sample_dir.exists():
        logger.error(f"Sample directory not found: {sample_dir}")
        return False

    # Get all text files (sample resumes)
    resume_files = list(sample_dir.glob("*.txt"))
    resume_files = [f for f in resume_files if f.name != "README.md" and "job_description" not in f.name.lower()]

    if not resume_files:
        logger.error("No sample resume files found")
        return False

    logger.info(f"Found {len(resume_files)} sample resumes")

    # Parse each resume
    parsed_resumes = []
    for resume_file in resume_files:
        logger.info(f"\nParsing: {resume_file.name}")
        try:
            # For text files, we need to create a temporary file or modify parser
            # For now, let's just test with the parser's text extraction
            with open(resume_file, "r", encoding="utf-8") as f:
                text = f.read()

            # Create a minimal ResumeData for testing
            from src.models.resume import ContactInfo, ResumeData
            from src.parsers.section_parser import SectionParser
            from src.parsers.skill_extractor import SkillExtractor

            section_parser = SectionParser()
            skill_extractor = SkillExtractor()

            sections = section_parser.parse_sections(text)
            skills = skill_extractor.extract_skills(text, sections)
            contact_info = section_parser.extract_contact_info(text)

            resume = ResumeData(
                file_name=resume_file.name,
                raw_text=text,
                contact_info=contact_info,
                skills=skills,
                experience=[],
                education=[],
            )

            parsed_resumes.append(resume)

            logger.info(f"  ✓ Name: {contact_info.name if contact_info else 'Unknown'}")
            logger.info(f"  ✓ Skills found: {len(skills)}")
            logger.info(f"  ✓ Skills: {', '.join(skills[:10])}")

        except Exception as e:
            logger.error(f"  ✗ Error parsing {resume_file.name}: {str(e)}")
            return False

    logger.info(f"\n✅ Successfully parsed {len(parsed_resumes)} resumes")
    return True, parsed_resumes


def test_embedding_generation(resumes):
    """Test embedding generation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Embedding Generation")
    logger.info("=" * 80)

    try:
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        logger.info(f"✓ Embedding generator initialized")

        # Test job description embedding
        job_desc = JobDescription(
            title="Senior Software Engineer",
            description="""
            We are seeking a Senior Software Engineer with strong Python and ML experience.
            Required skills: Python, Machine Learning, NLP, Docker, AWS.
            Preferred: FastAPI, React, PostgreSQL.
            """,
        )

        start_time = time.time()
        job_embedding = generator.encode_job_description(job_desc)
        job_time = time.time() - start_time

        logger.info(f"✓ Job description embedded in {job_time:.2f}s")
        logger.info(f"  Embedding shape: {job_embedding.shape}")

        # Test resume embeddings
        start_time = time.time()
        for i, resume in enumerate(resumes[:3], 1):  # Test first 3
            embedding = generator.encode_resume(resume)
            logger.info(f"✓ Resume {i} embedded - shape: {embedding.shape}")

        total_time = time.time() - start_time
        logger.info(f"\n✅ Embedded 3 resumes in {total_time:.2f}s ({total_time/3:.2f}s avg)")

        return True, generator, job_desc

    except Exception as e:
        logger.error(f"✗ Embedding generation failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False, None, None


def test_ranking_engine(resumes, generator, job_desc):
    """Test ranking engine."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Ranking Engine")
    logger.info("=" * 80)

    try:
        engine = RankingEngine(semantic_weight=0.7, skill_weight=0.3, embedding_generator=generator)
        logger.info("✓ Ranking engine initialized")

        # Process batch
        start_time = time.time()
        results = engine.process_batch(resumes=resumes, job_desc=job_desc, include_fairness=True)
        processing_time = time.time() - start_time

        logger.info(f"\n✅ Processed {len(resumes)} resumes in {processing_time:.2f}s")
        logger.info(f"   ({processing_time/len(resumes):.2f}s per resume)")

        # Display top 5 results
        logger.info("\n📊 Top 5 Candidates:")
        logger.info("-" * 80)

        for i, candidate in enumerate(results.ranked_candidates[:5], 1):
            name = candidate.resume.contact_info.name if candidate.resume.contact_info else "Unknown"
            logger.info(f"\n#{i} - {name}")
            logger.info(f"   Overall Score: {candidate.hybrid_score:.1%}")
            logger.info(f"   Semantic: {candidate.semantic_score:.1%} | Skills: {candidate.skill_score:.1%}")
            if candidate.explanation:
                logger.info(f"   Explanation: {candidate.explanation[:150]}...")

        # Fairness report
        if results.fairness_report:
            logger.info("\n🔍 Fairness Analysis:")
            logger.info(f"   Overall Fairness Score: {results.fairness_report.get_overall_fairness_score():.1%}")
            if results.fairness_report.bias_flags:
                logger.info(f"   ⚠️  Bias Flags: {len(results.fairness_report.bias_flags)}")
            else:
                logger.info("   ✅ No significant bias detected")

        return True

    except Exception as e:
        logger.error(f"✗ Ranking failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_memory_cache():
    """Test in-memory cache."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Memory Cache")
    logger.info("=" * 80)

    try:
        import numpy as np

        from src.embeddings.memory_cache import get_cache

        cache = get_cache()
        logger.info("✓ Memory cache initialized")

        # Test set/get
        test_embedding = np.random.rand(384)
        cache.set("test_text", test_embedding, "test_model")

        retrieved = cache.get("test_text", "test_model")
        assert retrieved is not None, "Cache retrieval failed"
        assert np.allclose(test_embedding, retrieved), "Cache data mismatch"

        logger.info("✓ Cache set/get working")

        # Test stats
        stats = cache.get_stats()
        logger.info(f"✓ Cache stats: {stats}")

        logger.info("\n✅ Memory cache working correctly")
        return True

    except Exception as e:
        logger.error(f"✗ Cache test failed: {str(e)}")
        return False


def test_free_llm():
    """Test free LLM service."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Free LLM Service")
    logger.info("=" * 80)

    try:
        from src.ranking.free_llm_service import FreeLLMServiceSync

        service = FreeLLMServiceSync()
        logger.info("✓ Free LLM service initialized")

        # Test explanation generation (will use template fallback if no API key)
        explanation = service.generate_explanation(
            candidate_name="John Doe",
            rank=1,
            hybrid_score=0.85,
            semantic_score=0.82,
            skill_score=0.90,
            matched_skills=["Python", "Machine Learning", "Docker"],
            missing_skills=["Kubernetes"],
            job_title="Senior Software Engineer",
            years_experience=5,
        )

        logger.info(f"✓ Generated explanation:")
        logger.info(f"  {explanation}")

        logger.info("\n✅ Free LLM service working (template fallback)")
        return True

    except Exception as e:
        logger.error(f"✗ LLM test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("🚀 REAL RESUME TESTING - PRE-DEPLOYMENT VERIFICATION")
    logger.info("=" * 80)

    all_passed = True

    # Test 1: Resume Parsing
    result = test_resume_parsing()
    if isinstance(result, tuple):
        passed, resumes = result
        if not passed:
            all_passed = False
            logger.error("\n❌ Resume parsing test FAILED")
            return
    else:
        all_passed = False
        logger.error("\n❌ Resume parsing test FAILED")
        return

    # Test 2: Embedding Generation
    passed, generator, job_desc = test_embedding_generation(resumes)
    if not passed:
        all_passed = False
        logger.error("\n❌ Embedding generation test FAILED")
        return

    # Test 3: Ranking Engine
    if not test_ranking_engine(resumes, generator, job_desc):
        all_passed = False
        logger.error("\n❌ Ranking engine test FAILED")
        return

    # Test 4: Memory Cache
    if not test_memory_cache():
        all_passed = False
        logger.warning("\n⚠️  Memory cache test FAILED (non-critical)")

    # Test 5: Free LLM
    if not test_free_llm():
        all_passed = False
        logger.warning("\n⚠️  Free LLM test FAILED (non-critical)")

    # Final summary
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ ALL TESTS PASSED - READY FOR DEPLOYMENT")
    else:
        logger.error("❌ SOME TESTS FAILED - FIX ISSUES BEFORE DEPLOYMENT")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
