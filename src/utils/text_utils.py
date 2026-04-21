"""Text processing utility functions."""

import re
import string
from typing import List, Optional


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)]", " ", text)

    # Remove multiple spaces
    text = re.sub(r" +", " ", text)

    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison and matching.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_email_addresses(text: str) -> List[str]:
    """
    Extract email addresses from text.

    Args:
        text: Text to search

    Returns:
        List of email addresses found
    """
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return re.findall(email_pattern, text)


def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract phone numbers from text.

    Args:
        text: Text to search

    Returns:
        List of phone numbers found
    """
    # Pattern for various phone number formats
    phone_patterns = [
        r"\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",  # US format
        r"\+?[0-9]{1,3}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}",  # International
    ]

    phone_numbers = []
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        phone_numbers.extend(matches)

    return list(set(phone_numbers))  # Remove duplicates


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    if not text:
        return []

    # Simple sentence splitting on periods, exclamation marks, and question marks
    sentences = re.split(r"[.!?]+", text)

    # Clean and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences
