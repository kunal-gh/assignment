'use client';

import { motion } from 'framer-motion';
import { useScreeningStore } from '@/store/screeningStore';

const STAGES = [
  { threshold: 0,  label: 'Initializing pipeline...', icon: '🚀' },
  { threshold: 15, label: 'Parsing resumes...', icon: '📄' },
  { threshold: 35, label: 'Extracting skills...', icon: '🔍' },
  { threshold: 50, label: 'Generating embeddings...', icon: '🧠' },
  { threshold: 65, label: 'Calculating scores...', icon: '📊' },
  { threshold: 80, label: 'Analyzing fairness...', icon: '⚖️' },
  { threshold: 92, label: 'Finalizing results...', icon: '✨' },
];

export default function LoadingScreen() {
  const { progress } = useScreeningStore();

  const currentStage = STAGES.reduce(
    (acc, stage) => (progress >= stage.threshold ? stage : acc),
    STAGES[0]
  );

  return (
    <div className="flex items-center justify-center min-h-[60vh] p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="brutalist-card p-10 max-w-md w-full"
      >
        {/* Animated icon */}
        <motion.div
          key={currentStage.icon}
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="text-6xl text-center mb-6"
        >
          {currentStage.icon}
        </motion.div>

        {/* Stage label */}
        <motion.h2
          key={currentStage.label}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-xl font-black text-center text-black mb-6 tracking-widest uppercase"
        >
          {currentStage.label}
        </motion.h2>

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

        <p className="text-center text-xs font-bold tracking-widest uppercase text-gray-400 mt-4">
          Semantic AI · NLP · FAISS Vector Search
        </p>
      </motion.div>
    </div>
  );
}
