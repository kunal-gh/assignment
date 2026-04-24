'use client';

import { Briefcase, Settings, CheckSquare, Square } from 'lucide-react';
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
    <div className="brutalist-card p-4 flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 pb-3 border-b-4 border-black flex-shrink-0">
        <Briefcase className="w-5 h-5" />
        <h2 className="text-base font-black tracking-widest uppercase text-black">
          JOB DESCRIPTION
        </h2>
      </div>

      {/* Inputs */}
      <div className="flex flex-col gap-3 flex-1 min-h-0">
        <div className="flex-shrink-0">
          <label className="block text-xs font-black tracking-widest uppercase mb-1 text-black">TITLE</label>
          <input
            type="text"
            value={jobTitle}
            onChange={(e) => setJobTitle(e.target.value)}
            placeholder="e.g., Senior Software Engineer"
            className="brutalist-input text-sm py-2"
          />
        </div>

        <div className="flex-1 flex flex-col min-h-0">
          <label className="block text-xs font-black tracking-widest uppercase mb-1 text-black">DESCRIPTION</label>
          <textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Enter job requirements, skills, responsibilities..."
            className="brutalist-input flex-1 resize-none text-sm min-h-0"
          />
        </div>
      </div>

      {/* Configuration */}
      <div className="mt-3 pt-3 border-t-4 border-black flex-shrink-0">
        <div className="flex items-center gap-2 mb-3">
          <Settings className="w-4 h-4" />
          <h3 className="text-xs font-black tracking-widest uppercase text-black">CONFIGURATION</h3>
        </div>

        {/* Scoring Weights */}
        <div className="mb-3">
          <div className="flex justify-between items-center mb-1">
            <label className="text-xs font-black tracking-widest uppercase text-black">SCORING WEIGHTS</label>
            <div className="bg-black text-white px-2 py-0.5 text-xs font-black tracking-widest">
              {(semanticWeight * 100).toFixed(0)}% SEM / {(skillWeight * 100).toFixed(0)}% SKL
            </div>
          </div>
          <p className="text-xs text-gray-500 mb-2">
            Semantic = meaning match. Skills = keyword match. Slide to balance both.
          </p>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={semanticWeight}
            onChange={(e) => setSemanticWeight(parseFloat(e.target.value))}
            style={{ transition: 'none' }}
            className="w-full h-2 bg-white border-2 border-black rounded-none appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
              [&::-webkit-slider-thumb]:bg-black [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer
              [&::-webkit-slider-thumb]:transition-none
              [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4
              [&::-moz-range-thumb]:bg-black [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0"
          />
          <div className="flex justify-between text-xs font-bold text-gray-400 mt-1 uppercase">
            <span>← Semantic</span>
            <span>Skills →</span>
          </div>
        </div>

        {/* Fairness Toggle */}
        <div
          className="flex items-center justify-between border-2 border-black p-3 cursor-pointer hover:bg-gray-50 transition-colors"
          onClick={() => setIncludeFairness(!includeFairness)}
        >
          <div>
            <p className="text-xs font-black tracking-widest uppercase text-black">FAIRNESS ANALYSIS</p>
            <p className="text-xs text-gray-500 mt-0.5">
              Checks rankings for bias patterns using the four-fifths rule.
            </p>
          </div>
          <div className="text-black ml-3 flex-shrink-0">
            {includeFairness ? <CheckSquare className="w-6 h-6" /> : <Square className="w-6 h-6" />}
          </div>
        </div>
      </div>
    </div>
  );
}
