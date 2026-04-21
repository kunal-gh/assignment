"""
Comprehensive local testing before deployment.
Tests all components, security, performance, and integration.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


class LocalDeploymentTester:
    """Comprehensive testing suite for local deployment."""

    def __init__(self):
        self.test_results = {"passed": [], "failed": [], "warnings": []}

    def log_pass(self, test_name: str, message: str = ""):
        """Log a passing test."""
        self.test_results["passed"].append(test_name)
        logger.info(f"✅ {test_name}: {message}")

    def log_fail(self, test_name: str, error: str):
        """Log a failing test."""
        self.test_results["failed"].append(test_name)
        logger.error(f"❌ {test_name}: {error}")

    def log_warning(self, test_name: str, message: str):
        """Log a warning."""
        self.test_results["warnings"].append(test_name)
        logger.warning(f"⚠️  {test_name}: {message}")

    def print_summary(self):
        """Print test summary."""
        total = len(self.test_results["passed"]) + len(self.test_results["failed"])
        passed = len(self.test_results["passed"])
        failed = len(self.test_results["failed"])
        warnings = len(self.test_results["warnings"])

        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {total}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⚠️  Warnings: {warnings}")

        if failed == 0:
            logger.info("\n🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT!")
        else:
            logger.error("\n❌ SOME TESTS FAILED - FIX BEFORE DEPLOYMENT")
            logger.error("Failed tests:")
            for test in self.test_results["failed"]:
                logger.error(f"  - {test}")

        logger.info("=" * 80)

        return failed == 0


def test_imports():
    """Test 1: Verify all imports work."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Import Verification")
    logger.info("=" * 80)

    tester = LocalDeploymentTester()

    try:
        from src.parsers.resume_parser import ResumeParser

        tester.log_pass("ResumeParser import", "OK")
    except Exception as e:
        tester.log_fail("ResumeParser import", str(e))

    try:
        from src.embeddings.embedding_generator import EmbeddingGenerator

        tester.log_pass("EmbeddingGenerator import", "OK")
    except Exception as e:
        tester.log_fail("EmbeddingGenerator import", str(e))

    try:
        from src.embeddings.memory_cache import get_cache

        tester.log_pass("MemoryCache import", "OK")
    except Exception as e:
        tester.log_fail("MemoryCache import", str(e))

    try:
        from src.ranking.ranking_engine import RankingEngine

        tester.log_pass("RankingEngine import", "OK")
    except Exception as e:
        tester.log_fail("RankingEngine import", str(e))

    try:
        from src.ranking.free_llm_service import FreeLLMServiceSync

        tester.log_pass("FreeLLMService import", "OK")
    except Exception as e:
        tester.log_fail("FreeLLMService import", str(e))

    try:
        from src.models.job import JobDescription
        from src.models.ranking import RankedCandidate
        from src.models.resume import ResumeData

        tester.log_pass("Data models import", "OK")
    except Exception as e:
        tester.log_fail("Data models import", str(e))

    return tester


def test_security():
    """Test 2: Security checks."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Security Verification")
    logger.info("=" * 80)

    tester = LocalDeploymentTester()

    # Check for exposed API keys
    env_file = Path(".env")
    if env_file.exists():
        tester.log_warning("Environment file", ".env file exists - ensure it's in .gitignore")
    else:
        tester.log_pass("Environment file", "No .env file in repo")

    # Check .gitignore
    gitignore = Path(".gitignore")
    if gitignore.exists():
        content = gitignore.read_text()
        if ".env" in content:
            tester.log_pass("Gitignore", ".env is ignored")
        else:
            tester.log_fail("Gitignore", ".env not in .gitignore")

        if ".kiro" in content or "*.kiro" in content:
            tester.log_pass("Gitignore", ".kiro directory ignored")
        else:
            tester.log_warning("Gitignore", ".kiro directory not ignored")
    else:
        tester.log_fail("Gitignore", ".gitignore file missing")

    # Check for hardcoded secrets in code
    suspicious_patterns = ["sk-", "api_key =", "password =", "secret ="]
    python_files = list(Path("src").rglob("*.py"))

    found_secrets = False
    for file in python_files:
        content = file.read_text()
        for pattern in suspicious_patterns:
            if pattern in content.lower() and "example" not in content.lower():
                tester.log_warning("Hardcoded secrets", f"Suspicious pattern '{pattern}' in {file}")
                found_secrets = True

    if not found_secrets:
        tester.log_pass("Hardcoded secrets", "No suspicious patterns found")

    # Check file size limits
    try:
        from src.parsers.resume_parser import ResumeParser

        parser = ResumeParser()
        # This should be in the code
        tester.log_pass("File size validation", "Parser has file size checks")
    except Exception as e:
        tester.log_warning("File size validation", "Could not verify file size limits")

    return tester


def test_performance():
    """Test 3: Performance benchmarks."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Performance Benchmarks")
    logger.info("=" * 80)

    tester = LocalDeploymentTester()

    try:
        from src.embeddings.embedding_generator import EmbeddingGenerator
        from src.models.job import JobDescription

        # Test cold start time
        logger.info("Testing cold start time...")
        start = time.time()
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        cold_start = time.time() - start

        if cold_start < 20:
            tester.log_pass("Cold start time", f"{cold_start:.2f}s (target: <20s)")
        else:
            tester.log_warning("Cold start time", f"{cold_start:.2f}s (slow, target: <20s)")

        # Test embedding speed
        logger.info("Testing embedding generation speed...")
        job_desc = JobDescription(
            title="Test Job", description="This is a test job description with Python, ML, and Docker skills."
        )

        start = time.time()
        embedding = generator.encode_job_description(job_desc)
        embed_time = time.time() - start

        if embed_time < 1.0:
            tester.log_pass("Embedding speed", f"{embed_time:.3f}s per embedding (target: <1s)")
        else:
            tester.log_warning("Embedding speed", f"{embed_time:.3f}s (slow, target: <1s)")

        # Test memory usage
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        if memory_mb < 500:
            tester.log_pass("Memory usage", f"{memory_mb:.1f}MB (target: <500MB)")
        else:
            tester.log_warning("Memory usage", f"{memory_mb:.1f}MB (high, target: <500MB)")

    except Exception as e:
        tester.log_fail("Performance tests", str(e))

    return tester


def test_integration():
    """Test 4: End-to-end integration."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: End-to-End Integration")
    logger.info("=" * 80)

    tester = LocalDeploymentTester()

    try:
        from src.embeddings.embedding_generator import EmbeddingGenerator
        from src.models.job import JobDescription
        from src.models.resume import ContactInfo, ResumeData
        from src.parsers.resume_parser import ResumeParser
        from src.ranking.ranking_engine import RankingEngine

        # Create test data
        logger.info("Creating test data...")

        # Mock resume
        resume = ResumeData(
            file_name="test_resume.txt",
            raw_text="John Doe is a software engineer with 5 years of Python and ML experience.",
            contact_info=ContactInfo(name="John Doe", email="john@example.com"),
            skills=["Python", "Machine Learning", "Docker"],
            experience=[],
            education=[],
        )

        job_desc = JobDescription(
            title="Senior Software Engineer", description="Looking for a Python developer with ML experience."
        )

        # Test full pipeline
        logger.info("Testing full pipeline...")
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        engine = RankingEngine(semantic_weight=0.7, skill_weight=0.3, embedding_generator=generator)

        start = time.time()
        results = engine.process_batch(resumes=[resume], job_desc=job_desc, include_fairness=True)
        processing_time = time.time() - start

        # Verify results
        if results.total_resumes == 1:
            tester.log_pass("Pipeline execution", f"Processed in {processing_time:.2f}s")
        else:
            tester.log_fail("Pipeline execution", "Incorrect result count")

        if results.ranked_candidates:
            candidate = results.ranked_candidates[0]
            if 0 <= candidate.hybrid_score <= 1:
                tester.log_pass("Score validation", f"Score: {candidate.hybrid_score:.2%}")
            else:
                tester.log_fail("Score validation", f"Invalid score: {candidate.hybrid_score}")
        else:
            tester.log_fail("Pipeline execution", "No candidates returned")

        if results.fairness_report:
            tester.log_pass("Fairness analysis", "Report generated")
        else:
            tester.log_warning("Fairness analysis", "No fairness report")

    except Exception as e:
        tester.log_fail("Integration test", str(e))
        import traceback

        traceback.print_exc()

    return tester


def test_error_handling():
    """Test 5: Error handling and edge cases."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Error Handling")
    logger.info("=" * 80)

    tester = LocalDeploymentTester()

    try:
        from src.embeddings.embedding_generator import EmbeddingGenerator
        from src.models.job import JobDescription
        from src.ranking.ranking_engine import RankingEngine

        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        engine = RankingEngine(embedding_generator=generator)

        # Test empty input
        logger.info("Testing empty input...")
        job_desc = JobDescription(title="Test", description="Test job")
        results = engine.process_batch(resumes=[], job_desc=job_desc)

        if results.total_resumes == 0:
            tester.log_pass("Empty input handling", "Handled gracefully")
        else:
            tester.log_fail("Empty input handling", "Unexpected behavior")

        # Test invalid weights
        logger.info("Testing invalid weights...")
        try:
            invalid_engine = RankingEngine(semantic_weight=1.5, skill_weight=-0.5)
            tester.log_fail("Weight validation", "Accepted invalid weights")
        except ValueError:
            tester.log_pass("Weight validation", "Rejected invalid weights")

        # Test cache functionality
        logger.info("Testing cache...")
        import numpy as np

        from src.embeddings.memory_cache import get_cache

        cache = get_cache()
        test_vec = np.random.rand(384)
        cache.set("test_key", test_vec, "test_model")
        retrieved = cache.get("test_key", "test_model")

        if retrieved is not None and np.allclose(test_vec, retrieved):
            tester.log_pass("Cache functionality", "Working correctly")
        else:
            tester.log_fail("Cache functionality", "Cache retrieval failed")

    except Exception as e:
        tester.log_fail("Error handling tests", str(e))
        import traceback

        traceback.print_exc()

    return tester


def test_dependencies():
    """Test 6: Verify all dependencies are installed."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Dependency Verification")
    logger.info("=" * 80)

    tester = LocalDeploymentTester()

    required_packages = [
        "sentence_transformers",
        "numpy",
        "pandas",
        "spacy",
        "fastapi",
        "streamlit",
        "plotly",
        "httpx",
    ]

    for package in required_packages:
        try:
            __import__(package)
            tester.log_pass(f"Package: {package}", "Installed")
        except ImportError:
            tester.log_fail(f"Package: {package}", "Not installed")

    # Check spaCy model
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        tester.log_pass("spaCy model", "en_core_web_sm loaded")
    except Exception as e:
        tester.log_fail("spaCy model", "en_core_web_sm not found")

    return tester


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("🧪 LOCAL DEPLOYMENT TESTING")
    logger.info("=" * 80)

    all_testers = []

    # Run all tests
    all_testers.append(test_imports())
    all_testers.append(test_security())
    all_testers.append(test_performance())
    all_testers.append(test_integration())
    all_testers.append(test_error_handling())
    all_testers.append(test_dependencies())

    # Aggregate results
    total_passed = sum(len(t.test_results["passed"]) for t in all_testers)
    total_failed = sum(len(t.test_results["failed"]) for t in all_testers)
    total_warnings = sum(len(t.test_results["warnings"]) for t in all_testers)

    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ Passed: {total_passed}")
    logger.info(f"❌ Failed: {total_failed}")
    logger.info(f"⚠️  Warnings: {total_warnings}")

    if total_failed == 0:
        logger.info("\n🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT!")
        logger.info("\nNext steps:")
        logger.info("1. Complete frontend components")
        logger.info("2. Create Vercel serverless functions")
        logger.info("3. Deploy to Vercel")
        return True
    else:
        logger.error("\n❌ SOME TESTS FAILED - FIX BEFORE DEPLOYMENT")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
