# Deployment Guide

This document outlines the deployment process for the AI Resume Screener using Docker, Kubernetes (optional), and GitHub Actions.

## Recommended Stack
- **Web App**: Streamlit (port 8501)
- **API Server**: FastAPI serving over Uvicorn (port 8000)
- **Caching**: Redis
- **Reverse Proxy**: NGINX / Caddy

## Docker Compose Setup

### Local / Staging
To run the entire stack locally or in a staging VM, simply run the included Docker Compose configuration:

```bash
cp .env.example .env
# Edit .env with your specific API keys
docker-compose up -d --build
```
This binds Streamlit to port 8501 and FastAPI to port 8000. It also starts a Redis container to cache job states and embeddings.

## Security Improvements (Phase 9)
By default, the Fast API leverages simple token-based verification using `OAuth2PasswordBearer` and handles rate limiting (via `slowapi`) across all critical endpoints:
- Limits model requests (`/screen`) to 10/min per IP
- Restricts Authentication attempts (`/token`) to 5/min.
- Overrides basic HTTP headers for added XSS protection (`X-Content-Type-Options: nosniff`).

## Monitoring & Alerting
We recommend integrating Datadog or Prometheus telemetry for production deployments.

- **Status Healthcheck**: Accessible at `/health`. Add this to your orchestrator's Liveness/Readiness probes.
- **Metrics Endpoint**: `/metrics` exports real-time component caching data.

## Environment Breakdown
Refer to `.env.example` in the source directory for configuration details. Specifically, `OPENAI_API_KEY` is completely optional: lacking it simply redirects the screening engine to use reliable string-based template explanations instead of ChatGPT generated explanations.
