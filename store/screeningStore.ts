import { create } from 'zustand';

export interface Candidate {
  rank: number;
  name: string;
  email?: string;
  hybrid_score: number;
  semantic_score: number;
  skill_score: number;
  matched_skills: string[];
  missing_skills: string[];
  years_experience: number;
  explanation?: string;
}

export interface ScreeningResults {
  job_id: string;
  job_title: string;
  total_resumes: number;
  successfully_parsed: number;
  processing_time_seconds: number;
  candidates: Candidate[];
  fairness_summary?: {
    overall_score: number;
    bias_flags: string[];
    recommendations: string[];
  };
  created_at: string;
  model_used?: string;
}

interface ScreeningState {
  jobTitle: string;
  jobDescription: string;
  setJobTitle: (title: string) => void;
  setJobDescription: (description: string) => void;

  files: File[];
  addFiles: (newFiles: File[]) => void;
  removeFile: (index: number) => void;
  clearFiles: () => void;

  semanticWeight: number;
  setSemanticWeight: (weight: number) => void;
  includeFairness: boolean;
  setIncludeFairness: (include: boolean) => void;

  isProcessing: boolean;
  progress: number;
  statusMessage: string;
  error: string | null;

  results: ScreeningResults | null;
  setResults: (results: ScreeningResults) => void;

  processResumes: () => Promise<void>;
}

// Call the real ML backend directly to bypass Vercel's 60-second serverless timeout limit.
function getApiUrl(): string {
  const base = process.env.NEXT_PUBLIC_API_URL || 'https://ai-resume-screener-api-5iq6.onrender.com';
  return base.endsWith('/screen') ? base : `${base.replace(/\/$/, '')}/screen`;
}

const STATUS_STAGES = [
  { at: 0,  msg: 'Waking up AI backend...' },
  { at: 10, msg: 'Extracting text (PyMuPDF / python-docx)...' },
  { at: 22, msg: 'Scanned PDF? Invoking Gemini Vision OCR...' },
  { at: 35, msg: 'Running regex skill extraction (60+ skills)...' },
  { at: 48, msg: 'Generating Gemini embeddings (batch, 768-dim)...' },
  { at: 65, msg: 'Computing NumPy cosine similarity...' },
  { at: 75, msg: 'Computing hybrid scores + Hidden Gem detection...' },
  { at: 85, msg: 'Building AI explanations...' },
  { at: 90, msg: 'Analyzing fairness metrics...' },
];

export const useScreeningStore = create<ScreeningState>((set, get) => ({
  jobTitle: '',
  jobDescription: '',
  files: [],
  semanticWeight: 0.7,
  includeFairness: true,
  isProcessing: false,
  progress: 0,
  statusMessage: '',
  results: null,
  error: null,

  setJobTitle: (title) => set({ jobTitle: title }),
  setJobDescription: (desc) => set({ jobDescription: desc }),
  addFiles: (f) => set((s) => ({ files: [...s.files, ...f] })),
  removeFile: (i) => set((s) => ({ files: s.files.filter((_, idx) => idx !== i) })),
  clearFiles: () => set({ files: [] }),
  setSemanticWeight: (w) => set({ semanticWeight: w }),
  setIncludeFairness: (v) => set({ includeFairness: v }),
  setResults: (r) => set({ results: r, isProcessing: false, error: null }),

  processResumes: async () => {
    const state = get();
    if (!state.jobTitle || !state.jobDescription || !state.files.length) {
      set({ error: 'Please provide job title, description, and at least one resume.' });
      return;
    }

    set({ isProcessing: true, progress: 0, error: null, results: null, statusMessage: 'Starting...' });

    // Animate progress with stage messages
    let currentProgress = 0;
    const progressInterval = setInterval(() => {
      currentProgress = Math.min(currentProgress + 2, 92);
      const stage = [...STATUS_STAGES].reverse().find(s => currentProgress >= s.at);
      set({ progress: currentProgress, statusMessage: stage?.msg || 'Processing...' });
    }, 800);

    try {
      // The Next.js route proxies to Render server-side (no CORS)
      set({ statusMessage: 'Connecting to AI backend...' });

      const formData = new FormData();
      formData.append('job_title', state.jobTitle);
      formData.append('job_description', state.jobDescription);
      formData.append('semantic_weight', state.semanticWeight.toString());
      formData.append('include_fairness', state.includeFairness.toString());
      state.files.forEach((f) => formData.append('files', f));

      // 3-minute timeout for cold start + ML inference
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 180000);

      const response = await fetch(getApiUrl(), {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeout);
      clearInterval(progressInterval);

      if (!response.ok) {
        let msg = `Server error: ${response.status}`;
        try { const d = await response.json(); msg = d.error || d.detail || msg; } catch { /* ignore */ }
        throw new Error(msg);
      }

      const data = await response.json();
      const results: ScreeningResults = typeof data.body === 'string' ? JSON.parse(data.body) : data;

      set({ progress: 100, statusMessage: 'Done!' });
      setTimeout(() => set({ results, isProcessing: false, progress: 0, statusMessage: '' }), 400);

    } catch (err) {
      clearInterval(progressInterval);
      const msg = err instanceof Error
        ? (err.name === 'AbortError'
          ? 'Request timed out. The AI backend may be waking up — please try again in 30 seconds.'
          : err.message)
        : String(err);
      set({ isProcessing: false, progress: 0, error: msg, statusMessage: '' });
    }
  },
}));
