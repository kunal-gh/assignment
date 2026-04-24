'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Zap } from 'lucide-react';
import FileUpload from '@/components/FileUpload';
import JobDescriptionForm from '@/components/JobDescriptionForm';
import ResultsView from '@/components/ResultsView';
import LoadingScreen from '@/components/LoadingScreen';
import { useScreeningStore } from '@/store/screeningStore';

export default function Home() {
  const { results, isProcessing, processResumes, jobTitle, jobDescription, files } =
    useScreeningStore();
  const [step, setStep] = useState<'upload' | 'results'>('upload');

  const canProcess =
    jobTitle.trim().length > 0 && jobDescription.trim().length > 0 && files.length > 0;

  const handleProcess = async () => {
    await processResumes();
    setStep('results');
  };

  return (
    <main className="min-h-screen px-4 pb-8 md:px-8 pt-0 bg-white text-black">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-10 flex flex-col items-center pt-6"
      >
        <h1 className="text-3xl md:text-4xl lg:text-5xl font-black tracking-tighter leading-none uppercase mb-2 text-black">
          AI RESUME
        </h1>
        <div className="bg-black text-white px-4 py-2 transform -rotate-1 mb-4 inline-block">
          <h1 className="text-3xl md:text-4xl lg:text-5xl font-black tracking-tighter leading-none uppercase">
            SCREENER
          </h1>
        </div>

        <div className="w-full max-w-4xl h-1 bg-black mb-3"></div>
        <p className="text-sm md:text-base font-medium tracking-widest uppercase text-black">
          Rank candidates in seconds · Understand why, not just who
        </p>
        <div className="w-full max-w-4xl h-1 bg-black mt-3"></div>
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
          {/* Helper text at top */}
          {!canProcess && (
            <p className="text-center text-sm font-bold tracking-widest uppercase text-gray-400">
              Upload resumes + enter job title &amp; description to continue
            </p>
          )}

          <div className="grid md:grid-cols-2 gap-8">
            <FileUpload />
            <JobDescriptionForm />
          </div>

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
