'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Zap, FlaskConical } from 'lucide-react';
import FileUpload from '@/components/FileUpload';
import JobDescriptionForm from '@/components/JobDescriptionForm';
import ResultsView from '@/components/ResultsView';
import LoadingScreen from '@/components/LoadingScreen';
import { useScreeningStore } from '@/store/screeningStore';

// Sample demo data — pre-built results so reviewers can see the UI instantly
const DEMO_RESULTS = {
  job_id: 'demo_001',
  job_title: 'Senior ML Engineer',
  total_resumes: 6,
  successfully_parsed: 6,
  processing_time_seconds: 2.34,
  created_at: new Date().toISOString(),
  candidates: [
    {
      rank: 1,
      name: 'Priya Sharma',
      email: 'priya.sharma@example.com',
      hybrid_score: 0.912,
      semantic_score: 0.934,
      skill_score: 0.857,
      matched_skills: ['python', 'machine learning', 'nlp', 'faiss', 'pytorch', 'spacy'],
      missing_skills: ['kubernetes'],
      years_experience: 6,
      explanation:
        'Priya shows excellent fit for the Senior ML Engineer position with an overall score of 91.2% (Rank #1). Strong semantic alignment (93.4%) indicates deep NLP and ML expertise. Matched 6 required skills including python, machine learning, nlp, faiss. Development opportunity: kubernetes.',
    },
    {
      rank: 2,
      name: 'Aisha Rodriguez',
      email: 'aisha.rodriguez@example.com',
      hybrid_score: 0.871,
      semantic_score: 0.889,
      skill_score: 0.821,
      matched_skills: ['python', 'faiss', 'docker', 'kubernetes', 'mlops', 'embeddings'],
      missing_skills: ['spacy', 'nlp'],
      years_experience: 5,
      explanation:
        'Aisha shows excellent fit with 87.1% overall (Rank #2). Expert in embedding pipelines and MLOps. Strong DevOps skills complement the ML stack. Missing some NLP-specific experience.',
    },
    {
      rank: 3,
      name: 'Alex Chen',
      email: 'alex.chen@example.com',
      hybrid_score: 0.798,
      semantic_score: 0.812,
      skill_score: 0.762,
      matched_skills: ['python', 'tensorflow', 'scikit-learn', 'pandas', 'git', 'docker'],
      missing_skills: ['faiss', 'spacy', 'mlops'],
      years_experience: 4,
      explanation:
        'Alex shows good fit with 79.8% overall (Rank #3). Strong Python and ML fundamentals. Good all-rounder with solid engineering practices. Could benefit from more NLP-specific tooling experience.',
    },
    {
      rank: 4,
      name: 'Dr. Sarah Okonkwo',
      email: 'sarah.okonkwo@example.com',
      hybrid_score: 0.743,
      semantic_score: 0.821,
      skill_score: 0.571,
      matched_skills: ['python', 'nlp', 'transformers', 'research'],
      missing_skills: ['faiss', 'docker', 'kubernetes', 'mlops'],
      years_experience: 7,
      explanation:
        '⭐ Hidden Gem: Sarah scores 74.3% overall (Rank #4) but her semantic score (82.1%) is much higher than her skill-match (57.1%). Her NLP research background uses different vocabulary — "document similarity systems" = FAISS, "information extraction" = spaCy. Worth a closer look.',
    },
    {
      rank: 5,
      name: 'Marcus Johnson',
      email: 'marcus.johnson@example.com',
      hybrid_score: 0.381,
      semantic_score: 0.412,
      skill_score: 0.286,
      matched_skills: ['javascript', 'react', 'python'],
      missing_skills: ['machine learning', 'nlp', 'faiss', 'pytorch', 'docker'],
      years_experience: 3,
      explanation:
        'Marcus shows limited fit with 38.1% overall (Rank #5). Strong frontend skills but limited ML/NLP experience. Growing interest in data science but not yet ready for a senior ML role.',
    },
    {
      rank: 6,
      name: 'James Whitfield',
      email: 'james.whitfield@example.com',
      hybrid_score: 0.089,
      semantic_score: 0.094,
      skill_score: 0.071,
      matched_skills: [],
      missing_skills: ['python', 'machine learning', 'nlp', 'faiss', 'docker'],
      years_experience: 12,
      explanation:
        'James shows very limited fit with 8.9% overall (Rank #6). HR management background with no technical ML skills. System correctly deprioritises non-technical profiles for this role.',
    },
  ],
  fairness_summary: {
    overall_score: 0.94,
    bias_flags: [],
    recommendations: [
      'Rankings are based purely on skills and semantic relevance — no demographic data used.',
      'Note: Dr. Sarah Okonkwo (Rank #4) is a potential hidden gem — high semantic score despite lower keyword match.',
      'Consider blind review for shortlisted candidates to further reduce unconscious bias.',
    ],
  },
};

export default function Home() {
  const { results, isProcessing, processResumes, setResults, jobTitle, jobDescription, files } =
    useScreeningStore();
  const [step, setStep] = useState<'upload' | 'results'>('upload');

  const canProcess =
    jobTitle.trim().length > 0 && jobDescription.trim().length > 0 && files.length > 0;

  const handleProcess = async () => {
    await processResumes();
    setStep('results');
  };

  const handleDemo = () => {
    setResults(DEMO_RESULTS as any);
    setStep('results');
  };

  return (
    <main className="min-h-screen px-4 pb-8 md:px-8 pt-0 bg-white text-black">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-10 flex flex-col items-center"
      >
        <div className="inline-flex items-center gap-2 border-2 border-black px-4 py-1 mb-6 mt-4 font-sans">
          <Activity className="w-4 h-4" />
          <span className="text-xs font-bold tracking-widest uppercase">
            AI-POWERED · NLP · SEMANTIC MATCHING
          </span>
        </div>

        <h1 className="text-6xl md:text-8xl lg:text-9xl font-black tracking-tighter leading-none uppercase mb-2 text-black">
          AI RESUME
        </h1>
        <div className="bg-black text-white px-4 md:px-8 py-2 md:py-4 transform -rotate-1 mb-6 inline-block">
          <h1 className="text-6xl md:text-8xl lg:text-9xl font-black tracking-tighter leading-none uppercase">
            SCREENER
          </h1>
        </div>

        <div className="w-full max-w-4xl h-1 bg-black mb-4"></div>
        <p className="text-lg md:text-xl font-medium tracking-widest uppercase text-black">
          Rank candidates in seconds · Understand why, not just who
        </p>
        <div className="w-full max-w-4xl h-1 bg-black mt-4"></div>
      </motion.header>

      {/* Content */}
      {isProcessing ? (
        <LoadingScreen />
      ) : step === 'upload' ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="max-w-6xl mx-auto space-y-6"
        >
          {/* Demo banner */}
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="border-4 border-black p-4 bg-gray-50 flex flex-col md:flex-row items-center justify-between gap-4"
          >
            <div>
              <p className="font-black tracking-widest uppercase text-sm text-black">
                🚀 Want to see it in action instantly?
              </p>
              <p className="text-xs font-bold tracking-widest text-gray-500 mt-1">
                Load 6 pre-built candidate profiles for a Senior ML Engineer role — including a
                &quot;hidden gem&quot; demo
              </p>
            </div>
            <button
              onClick={handleDemo}
              className="flex items-center gap-2 px-6 py-3 border-4 border-black bg-white font-black tracking-widest uppercase text-sm hover:bg-black hover:text-white transition-colors flex-shrink-0"
            >
              <FlaskConical className="w-4 h-4" /> LOAD DEMO
            </button>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            <FileUpload />
            <JobDescriptionForm />
          </div>

          {!canProcess && (
            <p className="text-center text-sm font-bold tracking-widest uppercase text-gray-400">
              Upload resumes + enter job title & description to continue
            </p>
          )}

          <button
            onClick={handleProcess}
            disabled={!canProcess}
            className={`brutalist-button w-full py-6 text-xl md:text-2xl mt-4 ${
              !canProcess
                ? 'opacity-40 cursor-not-allowed hover:translate-y-0 hover:translate-x-0 hover:shadow-brutal'
                : ''
            }`}
          >
            PROCESS RESUMES <Zap className="w-6 h-6 ml-2" />
          </button>
        </motion.div>
      ) : (
        <ResultsView onBack={() => setStep('upload')} />
      )}

      {/* Footer */}
      <footer className="mt-16 text-center border-t-4 border-black pt-6">
        <p className="text-xs font-bold tracking-widest uppercase text-gray-400">
          sentence-transformers · FAISS · spaCy · FastAPI · Next.js · Docker
        </p>
      </footer>
    </main>
  );
}
