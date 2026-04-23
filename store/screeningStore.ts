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
  embeddingModel: string;
  setEmbeddingModel: (model: string) => void;
  includeFairness: boolean;
  setIncludeFairness: (include: boolean) => void;

  isProcessing: boolean;
  progress: number;
  setProgress: (progress: number) => void;
  error: string | null;

  results: ScreeningResults | null;
  setResults: (results: ScreeningResults) => void;
  clearResults: () => void;

  processResumes: () => Promise<void>;
}

export const useScreeningStore = create<ScreeningState>((set, get) => ({
  jobTitle: '',
  jobDescription: '',
  files: [],
  semanticWeight: 0.7,
  embeddingModel: 'all-MiniLM-L6-v2',
  includeFairness: true,
  isProcessing: false,
  progress: 0,
  results: null,
  error: null,

  setJobTitle: (title) => set({ jobTitle: title }),
  setJobDescription: (description) => set({ jobDescription: description }),

  addFiles: (newFiles) =>
    set((state) => ({ files: [...state.files, ...newFiles] })),
  removeFile: (index) =>
    set((state) => ({ files: state.files.filter((_, i) => i !== index) })),
  clearFiles: () => set({ files: [] }),

  setSemanticWeight: (weight) => set({ semanticWeight: weight }),
  setEmbeddingModel: (model) => set({ embeddingModel: model }),
  setIncludeFairness: (include) => set({ includeFairness: include }),
  setProgress: (progress) => set({ progress }),
  setResults: (results) => set({ results, isProcessing: false, error: null }),
  clearResults: () => set({ results: null, error: null }),

  processResumes: async () => {
    const state = get();

    if (!state.jobTitle || !state.jobDescription || state.files.length === 0) {
      set({ error: 'Please provide job title, description, and at least one resume.' });
      return;
    }

    set({ isProcessing: true, progress: 0, error: null, results: null });

    // Animate progress while waiting
    const progressInterval = setInterval(() => {
      set((s) => ({ progress: Math.min(s.progress + 8, 88) }));
    }, 400);

    try {
      const formData = new FormData();
      formData.append('job_title', state.jobTitle);
      formData.append('job_description', state.jobDescription);
      formData.append('semantic_weight', state.semanticWeight.toString());
      formData.append('include_fairness', state.includeFairness.toString());
      formData.append('embedding_model', state.embeddingModel);

      state.files.forEach((file) => {
        formData.append('files', file);
      });

      const response = await fetch('/api/screen', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        let errMsg = `Server error: ${response.status} ${response.statusText}`;
        try {
          const errData = await response.json();
          errMsg = errData.error || errMsg;
        } catch {}
        throw new Error(errMsg);
      }

      const data = await response.json();

      // Handle Vercel serverless response format (body may be a string)
      const results: ScreeningResults =
        typeof data.body === 'string' ? JSON.parse(data.body) : data;

      set({ progress: 100 });

      setTimeout(() => {
        set({ results, isProcessing: false, progress: 0 });
      }, 400);
    } catch (error) {
      clearInterval(progressInterval);
      const msg = error instanceof Error ? error.message : String(error);
      console.error('Processing error:', msg);
      set({ isProcessing: false, progress: 0, error: msg });
    }
  },
}));
