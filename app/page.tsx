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
  const { isProcessing, processResumes, jobTitle, jobDescription, files } = useScreeningStore();
  const [step, setStep] = useState<'upload' | 'results'>('upload');

  const canProcess =
    jobTitle.trim().length > 0 && jobDescription.trim().length > 0 && files.length > 0;

  const handleProcess = async () => {
    await processResumes();
    setStep('results');
  };

  return (
    <main className="h-screen flex flex-col px-4 md:px-8 bg-white text-black overflow-hidden">
      {/* Header — compact */}
      <motion.header
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center flex flex-col items-center pt-4 pb-3 flex-shrink-0"
      >
        <div className="flex items-center gap-3 mb-1">
          <h1 className="text-4xl md:text-5xl font-black tracking-tighter leading-none uppercase text-black">
            AI RESUME
          </h1>
          <div className="bg-black text-white px-3 py-1 transform -rotate-1 inline-block">
            <span className="text-4xl md:text-5xl font-black tracking-tighter leading-none uppercase">
              SCREENER
            </span>
          </div>
        </div>
        <div className="w-full max-w-5xl h-[2px] bg-black mb-1 mt-2"></div>
        <p className="text-xs font-medium tracking-widest uppercase text-black">
          Rank candidates in seconds · Understand why, not just who
        </p>
        <div className="w-full max-w-5xl h-[2px] bg-black mt-1"></div>
      </motion.header>

      {/* Content */}
      {isProcessing ? (
        <div className="flex-1 overflow-auto">
          <LoadingScreen />
        </div>
      ) : step === 'upload' ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex-1 flex flex-col max-w-7xl w-full mx-auto min-h-0"
        >
          {/* Helper hint */}
          {!canProcess && (
            <p className="text-center text-xs font-bold tracking-widest uppercase text-gray-400 py-1 flex-shrink-0">
              Upload resumes + enter job title &amp; description to continue
            </p>
          )}

          {/* Cards — fill remaining space */}
          <div className="grid md:grid-cols-2 gap-4 flex-1 min-h-0 py-2">
            <FileUpload />
            <JobDescriptionForm />
          </div>

          {/* Process button */}
          <div className="flex-shrink-0 pb-3 pt-2">
            <button
              onClick={handleProcess}
              disabled={!canProcess}
              className={`brutalist-button w-full py-4 text-lg ${
                !canProcess
                  ? 'opacity-40 cursor-not-allowed hover:translate-y-0 hover:translate-x-0 hover:shadow-brutal'
                  : ''
              }`}
            >
              PROCESS RESUMES <Zap className="w-5 h-5 ml-2" />
            </button>
          </div>
        </motion.div>
      ) : (
        <div className="flex-1 overflow-auto">
          <ResultsView onBack={() => setStep('upload')} />
        </div>
      )}
    </main>
  );
}
