'use client';

import { motion } from 'framer-motion';
import { useScreeningStore } from '@/store/screeningStore';

export default function LoadingScreen() {
  const { progress } = useScreeningStore();

  const stages = [
    { threshold: 0, label: 'Initializing...', icon: '🚀' },
    { threshold: 20, label: 'Parsing resumes...', icon: '📄' },
    { threshold: 40, label: 'Generating embeddings...', icon: '🧠' },
    { threshold: 60, label: 'Calculating scores...', icon: '📊' },
    { threshold: 80, label: 'Analyzing fairness...', icon: '⚖️' },
    { threshold: 95, label: 'Finalizing results...', icon: '✨' },
  ];

  const currentStage = stages.reduce((acc, stage) => 
    progress >= stage.threshold ? stage : acc
  , stages[0]);

  return (
    <div className="fixed inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center z-50">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="glass-card p-12 max-w-md w-full mx-4"
      >
        {/* Icon */}
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          className="text-8xl text-center mb-8"
        >
          {currentStage.icon}
        </motion.div>

        {/* Stage Label */}
        <motion.h2
          key={currentStage.label}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-2xl font-bold text-center text-gray-800 mb-8"
        >
          {currentStage.label}
        </motion.h2>

        {/* Progress Bar */}
        <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden mb-4">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary-500 to-primary-600 rounded-full"
          />
        </div>

        {/* Progress Percentage */}
        <p className="text-center text-gray-600 font-medium">
          {progress}%
        </p>

        {/* Animated Dots */}
        <div className="flex justify-center space-x-2 mt-6">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              animate={{
                scale: [1, 1.5, 1],
                opacity: [0.3, 1, 0.3],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                delay: i * 0.2,
              }}
              className="w-2 h-2 bg-primary-500 rounded-full"
            />
          ))}
        </div>
      </motion.div>
    </div>
  );
}
