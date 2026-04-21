# 🚀 Vercel Deployment Plan - AI Resume Screener

## Phase 1: Make Everything 100% Free ✅

### 1.1 Replace OpenAI with Free Alternatives
- ✅ Use Hugging Face Inference API (Free tier: 30k tokens/month)
- ✅ Qwen2.5-72B-Instruct (Free via HF Inference)
- ✅ Fallback to template-based explanations (already implemented)

### 1.2 Replace Redis with In-Memory Cache
- ✅ Use Python's functools.lru_cache
- ✅ Implement simple in-memory dict cache for embeddings

### 1.3 Optimize Model Loading
- ✅ Use smaller embedding model for Vercel (all-MiniLM-L6-v2)
- ✅ Lazy loading to reduce cold start time

## Phase 2: Test with Real Resumes ✅

### 2.1 Local Testing
- Test with sample resumes
- Verify parsing accuracy
- Check scoring consistency
- Test fairness detection

### 2.2 Security Audit
- Input validation
- File size limits
- Rate limiting
- XSS protection

## Phase 3: Modern Frontend (Next.js 15 + Preact) 🎨

### 3.1 Tech Stack
- **Framework**: Next.js 15 (App Router)
- **UI Library**: Preact (lightweight React alternative)
- **Styling**: Tailwind CSS + Framer Motion
- **Components**: shadcn/ui (minimalist, modern)
- **Charts**: Recharts (lightweight)
- **File Upload**: react-dropzone
- **State**: Zustand (minimal state management)

### 3.2 Design System
- **Color**: Single shade - Deep Purple (#6366f1)
- **Style**: Minimalist, clean, spacious
- **Animations**: Smooth transitions, micro-interactions
- **Typography**: Inter font family
- **Layout**: Card-based, glassmorphism effects

### 3.3 Pages
1. **Home** - Upload & Configure
2. **Results** - Ranked candidates with animations
3. **Analytics** - Charts and insights
4. **API Docs** - Interactive API documentation

## Phase 4: Vercel Deployment 🌐

### 4.1 Backend (Serverless Functions)
- Convert FastAPI to Vercel Serverless Functions
- Python runtime on Vercel
- Edge caching for embeddings

### 4.2 Frontend (Static + SSR)
- Next.js static generation
- API routes for backend calls
- Edge functions for performance

### 4.3 Environment Variables (Hidden)
- Use Vercel Environment Variables
- No .env files in repo
- Secrets encrypted in Vercel dashboard

### 4.4 Domain & SSL
- Free Vercel subdomain: resume-ai-screener.vercel.app
- Automatic SSL certificate
- CDN distribution

## Security Checklist ✅

- [ ] No API keys in code
- [ ] Rate limiting implemented
- [ ] File size validation (10MB max)
- [ ] Input sanitization
- [ ] CORS configured
- [ ] Security headers
- [ ] No sensitive data in logs

## Performance Targets 🎯

- Cold start: < 3s
- Resume processing: < 5s per resume
- Frontend load: < 1s
- Lighthouse score: > 90

## Free Tier Limits 📊

- Vercel: 100GB bandwidth/month
- Hugging Face: 30k tokens/month
- Vercel Functions: 100 hours/month
- Storage: In-memory only (no persistence)
