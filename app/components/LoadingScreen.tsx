'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useScreeningStore } from '@/store/screeningStore';
import { Clock } from 'lucide-react';

export default function LoadingScreen() {
  const { progress, statusMessage } = useScreeningStore();

  const showColdStartWarning = progress < 20;

  return (
    <div className="flex items-center justify-center min-h-[60vh] p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="brutalist-card p-10 max-w-lg w-full"
      >
        {/* Cold start notice — shown at the start */}
        <AnimatePresence>
          {showColdStartWarning && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="flex items-start gap-3 border-4 border-black bg-yellow-50 p-4 mb-6"
            >
              <Clock className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-xs font-black tracking-widest uppercase text-black mb-1">
                  First request may take 30–60s
                </p>
                <p className="text-xs font-medium text-gray-600 leading-relaxed">
                  The AI backend (sentence-transformers + FAISS) runs on Render free tier
                  and spins down after 15 min of inactivity. It&apos;s waking up now —
                  subsequent requests will be fast.
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Status message */}
        <motion.p
          key={statusMessage}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-sm font-black tracking-widest uppercase text-black mb-6 text-center min-h-[20px]"
        >
          {statusMessage || 'Processing...'}
        </motion.p>

        {/* Progress bar */}
        <div className="h-4 border-2 border-black bg-gray-100 overflow-hidden mb-3">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
            className="h-full bg-black"
          />
        </div>

        <p className="text-center text-black font-black tracking-widest text-lg mb-6">
          {progress}%
        </p>

        {/* Pipeline steps */}
        <div className="space-y-1 mb-6">
          {[
            { label: 'PDF text extraction (PyMuPDF)', done: progress >= 25 },
            { label: 'spaCy NER + skill extraction', done: progress >= 40 },
            { label: 'sentence-transformers embeddings', done: progress >= 55 },
            { label: 'FAISS cosine similarity', done: progress >= 70 },
            { label: 'Hybrid scoring + explanations', done: progress >= 82 },
            { label: 'Fairness analysis', done: progress >= 90 },
          ].map(({ label, done }) => (
            <div key={label} className="flex items-center gap-2">
              <span className={`text-xs font-bold ${done ? 'text-black' : 'text-gray-300'}`}>
                {done ? '✓' : '○'}
              </span>
              <span className={`text-xs font-medium tracking-wide ${done ? 'text-black' : 'text-gray-300'}`}>
                {label}
              </span>
            </div>
          ))}
        </div>

        {/* Dots */}
        <div className="flex justify-center space-x-3">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              animate={{ scale: [1, 1.5, 1], opacity: [0.3, 1, 0.3] }}
              transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
              className="w-3 h-3 bg-black border border-black rounded-full"
            />
          ))}
        </div>
      </motion.div>
    </div>
  );
}
