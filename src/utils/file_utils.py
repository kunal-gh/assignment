"""File utility functions."""

import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional


def validate_file(file_path: str, max_size_mb: int = 10) -> Dict[str, Any]:
    """
    Validate a file for resume processing.

    Args:
        file_path: Path to the file
        max_size_mb: Maximum file size in MB

    Returns:
        Dictionary with validation results
    """
    result = {"is_valid": False, "errors": [], "warnings": [], "file_info": {}}

    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        result["errors"].append(f"File does not exist: {file_path}")
        return result

    # Check file extension
    allowed_extensions = {".pdf", ".docx"}
    if path.suffix.lower() not in allowed_extensions:
        result["errors"].append(f"Unsupported file type: {path.suffix}")
        return result

    # Check file size
    file_size = path.stat().st_size
    max_size_bytes = max_size_mb * 1024 * 1024

    if file_size > max_size_bytes:
        result["errors"].append(f"File too large: {file_size} bytes (max: {max_size_bytes})")
        return result

    # Get file info
    result["file_info"] = get_file_info(file_path)

    # File is valid if no errors
    result["is_valid"] = len(result["errors"]) == 0

    return result


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information
    """
    path = Path(file_path)

    if not path.exists():
        return {"error": "File does not exist"}

    stat = path.stat()

    return {
        "name": path.name,
        "size": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "extension": path.suffix.lower(),
        "mime_type": mimetypes.guess_type(file_path)[0],
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "is_readable": os.access(file_path, os.R_OK),
    }
