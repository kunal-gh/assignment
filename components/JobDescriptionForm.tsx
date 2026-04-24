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
    <div className="brutalist-card p-6 flex flex-col h-full">
      <div className="flex items-center gap-3 mb-6 pb-4 border-b-4 border-black">
        <Briefcase className="w-6 h-6" />
        <h2 className="text-xl font-black tracking-widest uppercase text-black">
          JOB DESCRIPTION
        </h2>
      </div>

      <div className="space-y-5 flex-1">
        {/* Job Title */}
        <div>
          <label className="block text-sm font-black tracking-widest uppercase mb-2 text-black">TITLE</label>
          <input
            type="text"
            value={jobTitle}
            onChange={(e) => setJobTitle(e.target.value)}
            placeholder="e.g., Senior Software Engineer"
            className="brutalist-input"
          />
        </div>

        {/* Job Description */}
        <div className="flex-1 flex flex-col">
          <label className="block text-sm font-black tracking-widest uppercase mb-2 text-black">DESCRIPTION</label>
          <textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Enter job requirements, skills, responsibilities..."
            className="brutalist-input min-h-[140px] resize-none"
          />
        </div>
      </div>

      {/* Configuration */}
      <div className="mt-6 pt-6 border-t-4 border-black">
        <div className="flex items-center gap-2 mb-5">
          <Settings className="w-5 h-5" />
          <h3 className="text-sm font-black tracking-widest uppercase text-black">CONFIGURATION</h3>
        </div>

        {/* Scoring Weights */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <label className="text-sm font-black tracking-widest uppercase text-black">SCORING WEIGHTS</label>
            <div className="bg-black text-white px-2 py-1 text-xs font-black tracking-widest">
              {(semanticWeight * 100).toFixed(0)}% SEM / {(skillWeight * 100).toFixed(0)}% SKL
            </div>
          </div>
          <p className="text-sm text-gray-500 mb-3">
            How much weight to give AI understanding vs. exact skill keywords when scoring candidates. Higher semantic = rewards candidates who express the same ideas differently.
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
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5
              [&::-webkit-slider-thumb]:bg-black [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer
              [&::-webkit-slider-thumb]:transition-none
              [&::-moz-range-thumb]:w-5 [&::-moz-range-thumb]:h-5
              [&::-moz-range-thumb]:bg-black [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0"
          />
          <div className="flex justify-between text-sm font-bold text-gray-400 mt-2 uppercase">
            <span>← AI Meaning</span>
            <span>Exact Skills →</span>
          </div>
        </div>

        {/* Fairness Toggle */}
        <div
          className="flex items-center justify-between border-4 border-black p-4 cursor-pointer hover:bg-gray-50 transition-colors"
          onClick={() => setIncludeFairness(!includeFairness)}
        >
          <div>
            <p className="text-sm font-black tracking-widest uppercase text-black mb-1">FAIRNESS ANALYSIS</p>
            <p className="text-sm text-gray-500">
              Flags if any group of candidates is being systematically ranked lower — helps catch unintended bias in results.
            </p>
          </div>
          <div className="text-black ml-4 flex-shrink-0">
            {includeFairness ? <CheckSquare className="w-7 h-7" /> : <Square className="w-7 h-7" />}
          </div>
        </div>
      </div>
    </div>
  );
}
