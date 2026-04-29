# AI Resume Screener — Technical Documentation

## 1. Overview and Architecture

The AI Resume Screener is a sophisticated multi-modal application designed to automate the initial screening of resumes against a specific job description. It moves beyond simple keyword matching by employing semantic vector search and visual document parsing.

### 1.1 High-Level Architecture
The system consists of a decoupled frontend and backend:
*   **Frontend (Next.js 15):** A server-side rendered React application hosted on Vercel. It manages the user interface, state (via Zustand), and file uploads. It utilizes a Next.js API Route (`/api/screen`) to proxy requests to the backend, circumventing browser CORS restrictions for `multipart/form-data`.
*   **Backend (FastAPI):** A high-performance, asynchronous Python web server hosted on Render (using Docker). It orchestrates the Machine Learning (ML) pipeline.
*   **AI Services (Google Gemini):** The backend relies heavily on the Google Gemini API (via the `google-genai` SDK) for both Vision OCR (Optical Character Recognition) and generating text embeddings.

## 2. The 4-Layer Machine Learning Pipeline

The core logic resides in `backend/main.py`. The `screen_resumes` endpoint triggers a 4-layer pipeline for each uploaded resume.

### Layer 1: Multi-Modal Text Extraction
The goal is to robustly extract text from various file formats.
*   **Digital PDFs (`fitz` / PyMuPDF):** The primary method. It reads the binary bytes of a PDF and extracts the embedded text layer. It's extremely fast and accurate for digitally generated PDFs.
*   **Scanned PDFs (Gemini Vision OCR):** If `fitz` extracts fewer than 100 characters (indicating a likely scanned image), the system falls back to `gemini-1.5-flash`. The PDF bytes are sent as an image to Gemini with a prompt instructing it to extract all text exactly as written. This is handled synchronously (`_gemini_vision_ocr_sync`) but executed in a thread pool (`asyncio.to_thread`) to prevent blocking the async event loop.
*   **DOCX Files (`python-docx`):** Parses Word documents directly from memory (`io.BytesIO`). It extracts text from paragraphs and tables.

*Concurrency:* Text extraction for all uploaded resumes happens concurrently using `asyncio.gather(*extraction_tasks)`, significantly reducing overall latency.

### Layer 2: Regex-Based Skill Extraction
*   **Taxonomy:** A predefined list of over 60 canonical technical skills and their aliases (e.g., "javascript" -> ["js", "es6"]).
*   **Matching:** A regular expression search (`re.search`) scans the normalized lowercase text. For short acronyms (<= 4 chars, like "js" or "aws"), word boundary matching `(?:^|[\s,;(])` is used to prevent false positives (e.g., matching "aws" inside "flaws").
*   **IDF Scoring:** Skills are weighted using an Inverse Document Frequency (IDF) dictionary (`SKILL_IDF`). Rare, high-value skills (like "rag", "mcp") have higher weights (e.g., 3.5) than common skills like "python" (1.5).

### Layer 3: Semantic Vector Embeddings
This layer captures the semantic meaning of the text.
*   **Model:** `models/gemini-embedding-001` (truncated to 768 dimensions for performance).
*   **Task Types:** The Job Description (JD) is embedded as a `retrieval_query`. The resumes are embedded as `retrieval_document`. This helps the model optimize the vector space for search.
*   **Batching & Rate Limiting (`get_embeddings_batch`):**
    *   Texts are truncated to 6500 characters to safely stay under the 2048 token limit.
    *   Requests are batched in chunks of 50 to avoid payload size limits.
    *   A robust retry mechanism with exponential backoff handles transient `429 RESOURCE_EXHAUSTED` (rate limit) or `503 UNAVAILABLE` errors.

### Layer 4: Hybrid Scoring & Ranking
The final score combines semantic understanding with exact keyword matches.
*   **Semantic Score (Cosine Similarity):** Measures the angle between the JD vector and the resume vector using NumPy (`np.dot(va, vb) / (na * nb)`).
*   **Skill Score:** An IDF-weighted ratio of matched skills versus all skills required in the JD.
*   **Hybrid Score:** `(semantic_weight * semantic_score) + ((1 - semantic_weight) * skill_score)`. The `semantic_weight` defaults to 0.7 but is configurable from the frontend.
*   **Hidden Gem Detection:** If the semantic score is high (>= 0.55) but the skill score is relatively low (difference > 0.3), the candidate is flagged as a "Hidden Gem". This highlights candidates who have the right conceptual background but didn't use the exact keywords from the JD.

## 3. Frontend Implementation Details

Located in the `frontend/` directory.
*   **Framework:** Next.js 15 (App Router).
*   **State Management (`store/screeningStore.ts`):** Zustand manages the complex state machine of the screening process (job title, description, files, progress, status messages, results, errors). The `processResumes` action orchestrates the API call and updates progress.
*   **API Proxy (`app/api/screen/route.ts`):** This is critical. The browser cannot send a `multipart/form-data` POST request directly to the Render backend due to CORS restrictions. Instead, the browser sends the data to this Next.js API route (which is on the same origin), and this route forwards the request server-to-server to Render. It sets a 4-minute timeout to accommodate Render's cold start and ML processing time.
*   **UI Components:**
    *   `FileUpload.tsx`: Uses `react-dropzone` for handling file selection and drag-and-drop.
    *   `ResultsView.tsx`: Displays the final ranked list, utilizing `framer-motion` for animations and `recharts` for visualizing score breakdowns.

## 4. Fallback Mechanisms

*   **No API Key / Gemini Failure:** If `GOOGLE_API_KEY` is missing or the API completely fails, the system falls back to a custom TF-IDF (Term Frequency-Inverse Document Frequency) algorithm (`tfidf_cosine`) for calculating semantic similarity based purely on word frequencies. OCR for scanned PDFs will not function in this mode.

## 5. Deployment & Infrastructure

*   **Backend (Render):** Defined by `render.yaml` and `backend/Dockerfile`. It runs as a Docker container.
*   **Frontend (Vercel):** Configured via `frontend/vercel.json`. The `maxDuration` for the `/api/screen` function is explicitly set to 300 seconds (5 minutes) to prevent Vercel's serverless functions from timing out prematurely during heavy ML workloads.
*   **CI/CD (GitHub Actions):** `.github/workflows/ci.yml` defines the pipeline. It includes Black formatting checks, Flake8 linting, Bandit security scanning, backend import checks, and a Next.js build verification.
