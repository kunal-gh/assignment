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
  // Job description
  jobTitle: string;
  jobDescription: string;
  setJobTitle: (title: string) => void;
  setJobDescription: (description: string) => void;

  // Files
  files: File[];
  addFiles: (newFiles: File[]) => void;
  removeFile: (index: number) => void;
  clearFiles: () => void;

  // Configuration
  semanticWeight: number;
  setSemanticWeight: (weight: number) => void;
  embeddingModel: string;
  setEmbeddingModel: (model: string) => void;
  includeFairness: boolean;
  setIncludeFairness: (include: boolean) => void;

  // Processing
  isProcessing: boolean;
  progress: number;
  setProgress: (progress: number) => void;

  // Results
  results: ScreeningResults | null;
  setResults: (results: ScreeningResults) => void;
  clearResults: () => void;

  // Actions
  processResumes: () => Promise<void>;
}

export const useScreeningStore = create<ScreeningState>((set, get) => ({
  // Initial state
  jobTitle: '',
  jobDescription: '',
  files: [],
  semanticWeight: 0.7,
  embeddingModel: 'all-MiniLM-L6-v2',
  includeFairness: true,
  isProcessing: false,
  progress: 0,
  results: null,

  // Setters
  setJobTitle: (title) => set({ jobTitle: title }),
  setJobDescription: (description) => set({ jobDescription: description }),
  
  addFiles: (newFiles) => set((state) => ({
    files: [...state.files, ...newFiles]
  })),
  
  removeFile: (index) => set((state) => ({
    files: state.files.filter((_, i) => i !== index)
  })),
  
  clearFiles: () => set({ files: [] }),
  
  setSemanticWeight: (weight) => set({ semanticWeight: weight }),
  setEmbeddingModel: (model) => set({ embeddingModel: model }),
  setIncludeFairness: (include) => set({ includeFairness: include }),
  
  setProgress: (progress) => set({ progress }),
  setResults: (results) => set({ results, isProcessing: false }),
  clearResults: () => set({ results: null }),

  // Process resumes
  processResumes: async () => {
    const state = get();
    
    if (!state.jobTitle || !state.jobDescription || state.files.length === 0) {
      alert('Please provide job title, description, and at least one resume');
      return;
    }

    set({ isProcessing: true, progress: 0 });

    try {
      // Create form data
      const formData = new FormData();
      formData.append('job_title', state.jobTitle);
      formData.append('job_description', state.jobDescription);
      formData.append('semantic_weight', state.semanticWeight.toString());
      formData.append('include_fairness', state.includeFairness.toString());
      formData.append('embedding_model', state.embeddingModel);

      // Add files
      state.files.forEach((file) => {
        formData.append('files', file);
      });

      // Simulate progress
      const progressInterval = setInterval(() => {
        set((state) => ({
          progress: Math.min(state.progress + 10, 90)
        }));
      }, 500);

      // Call API
      const response = await fetch('/api/screen', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const results = await response.json();
      
      set({ progress: 100 });
      setTimeout(() => {
        set({ results, isProcessing: false, progress: 0 });
      }, 500);

    } catch (error) {
      console.error('Processing error:', error);
      alert(`Error processing resumes: ${error instanceof Error ? error.message : String(error)}`);
      set({ isProcessing: false, progress: 0 });
    }
  },
}));
