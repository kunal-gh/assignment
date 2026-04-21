'use client';

import { motion } from 'framer-motion';
import { useScreeningStore } from '@/store/screeningStore';

export default function JobDescriptionForm() {
  const {
    jobTitle,
    jobDescription,
    setJobTitle,
    setJobDescription,
    semanticWeight,
    setSemanticWeight,
    includeFairness,
    setIncludeFairness,
  } = useScreeningStore();

  const skillWeight = 1 - semanticWeight;

  return (
    <div className="glass-card p-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">
        💼 Job Description
      </h2>

      {/* Job Title */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Job Title
        </label>
        <input
          type="text"
          value={jobTitle}
          onChange={(e) => setJobTitle(e.target.value)}
          placeholder="e.g., Senior Software Engineer"
          className="input-field"
        />
      </div>

      {/* Job Description */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Description
        </label>
        <textarea
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
          placeholder="Enter the full job description including requirements, responsibilities, and qualifications..."
          rows={8}
          className="textarea-field"
        />
      </div>

      {/* Configuration */}
      <div className="space-y-6 pt-6 border-t border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">
          ⚙️ Configuration
        </h3>

        {/* Semantic Weight Slider */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <label className="text-sm font-medium text-gray-700">
              Scoring Weights
            </label>
            <div className="text-sm text-gray-600">
              <span className="font-medium text-primary-600">
                {(semanticWeight * 100).toFixed(0)}%
              </span>
              {' semantic / '}
              <span className="font-medium text-primary-600">
                {(skillWeight * 100).toFixed(0)}%
              </span>
              {' skills'}
            </div>
          </div>

          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={semanticWeight}
            onChange={(e) => setSemanticWeight(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-500"
          />

          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>More semantic</span>
            <span>More skills</span>
          </div>
        </div>

        {/* Fairness Analysis Toggle */}
        <div className="flex items-center justify-between">
          <div>
            <label className="text-sm font-medium text-gray-700">
              Fairness Analysis
            </label>
            <p className="text-xs text-gray-500 mt-1">
              Detect potential bias in rankings
            </p>
          </div>

          <button
            onClick={() => setIncludeFairness(!includeFairness)}
            className={`
              relative inline-flex h-6 w-11 items-center rounded-full
              transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2
              ${includeFairness ? 'bg-primary-500' : 'bg-gray-300'}
            `}
          >
            <motion.span
              animate={{ x: includeFairness ? 20 : 2 }}
              transition={{ type: 'spring', stiffness: 500, damping: 30 }}
              className="inline-block h-5 w-5 transform rounded-full bg-white shadow-lg"
            />
          </button>
        </div>
      </div>

      {/* Tips */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="mt-6 p-4 bg-primary-50 rounded-xl border border-primary-100"
      >
        <p className="text-sm text-primary-800">
          <span className="font-semibold">💡 Tip:</span> Include specific technical skills,
          experience requirements, and responsibilities for better matching.
        </p>
      </motion.div>
    </div>
  );
}
