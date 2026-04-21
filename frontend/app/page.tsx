'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import FileUpload from '@/components/FileUpload';
import JobDescriptionForm from '@/components/JobDescriptionForm';
import ResultsView from '@/components/ResultsView';
import LoadingScreen from '@/components/LoadingScreen';
import { useScreeningStore } from '@/store/screeningStore';

export default function Home() {
  const { results, isProcessing } = useScreeningStore();
  const [step, setStep] = useState<'upload' | 'results'>('upload');

  return (
    <main className="min-h-screen p-4 md:p-8">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <h1 className="text-5xl md:text-7xl font-bold text-gradient mb-4">
          AI Resume Screener
        </h1>
        <p className="text-gray-600 text-lg md:text-xl max-w-2xl mx-auto">
          Intelligent candidate ranking powered by semantic AI
        </p>
      </motion.header>

      {/* Main Content */}
      {isProcessing ? (
        <LoadingScreen />
      ) : step === 'upload' ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          className="max-w-6xl mx-auto"
        >
          <div className="grid md:grid-cols-2 gap-8">
            {/* Job Description */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
            >
              <JobDescriptionForm />
            </motion.div>

            {/* File Upload */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <FileUpload />
            </motion.div>
          </div>

          {/* Process Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mt-8 text-center"
          >
            <button
              onClick={() => setStep('results')}
              className="btn-primary text-lg px-12 py-4"
            >
              🚀 Process Resumes
            </button>
          </motion.div>
        </motion.div>
      ) : (
        <ResultsView onBack={() => setStep('upload')} />
      )}

      {/* Footer */}
      <motion.footer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="text-center mt-16 text-gray-500 text-sm"
      >
        <p>Built with Next.js, Preact & AI • 100% Free & Open Source</p>
      </motion.footer>
    </main>
  );
}
