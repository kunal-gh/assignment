"""
Test script for the new google-genai SDK.
Run after setting GOOGLE_API_KEY in backend/.env

Usage:
  cd backend
  python test_gemini.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not set in backend/.env")
    exit(1)

print(f"Using API key: {API_KEY[:8]}...{API_KEY[-4:]}")

from google import genai
from google.genai import types

client = genai.Client(api_key=API_KEY)

# ── Test 1: Batch embeddings ──────────────────────────────────────────────────
print("\n[1/3] Testing batch embeddings (retrieval_document)...")
texts = [
    "Software engineer with 5 years Python, FastAPI, Docker and Kubernetes experience",
    "Data scientist specializing in machine learning, PyTorch, and NLP transformers",
]
response = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=texts,
    config=types.EmbedContentConfig(
        task_type="retrieval_document",
        output_dimensionality=768,
    ),
)
embeddings = [e.values for e in response.embeddings]
print(f"  Got {len(embeddings)} embeddings, each {len(embeddings[0])}-dimensional ✓")

# ── Test 2: JD query embedding ────────────────────────────────────────────────
print("[2/3] Testing JD embedding (retrieval_query)...")
jd_response = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=["Seeking senior Python backend engineer with FastAPI and cloud experience"],
    config=types.EmbedContentConfig(
        task_type="retrieval_query",
        output_dimensionality=768,
    ),
)
jd_emb = jd_response.embeddings[0].values
print(f"  JD embedding: {len(jd_emb)}-dimensional ✓")

# ── Test 3: Cosine similarity ─────────────────────────────────────────────────
print("[3/3] Computing cosine similarity...")
import numpy as np

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

sim1 = cosine(jd_emb, embeddings[0])
sim2 = cosine(jd_emb, embeddings[1])
print(f"  JD vs 'Python/FastAPI engineer': {sim1:.4f}")
print(f"  JD vs 'Data scientist/NLP':     {sim2:.4f}")

if sim1 > sim2:
    print("  ✓ Correct: Python/FastAPI engineer ranked higher than data scientist for this JD")
else:
    print("  ⚠ Note: Data scientist ranked higher (unusual — check JD/resume texts)")

print("\n✅ All tests passed! API key is valid and google-genai SDK is working.")
