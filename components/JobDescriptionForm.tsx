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
      <div className="flex items-center gap-3 mb-6 pb-4 border-b-4 border-black border-solid">
        <Briefcase className="w-6 h-6" />
        <h2 className="text-xl font-black tracking-widest uppercase text-black">
          JOB DESCRIPTION
        </h2>
      </div>

      <div className="space-y-6 flex-1">
        {/* Job Title */}
        <div>
          <label className="brutalist-label">TITLE</label>
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
          <label className="brutalist-label">DESCRIPTION</label>
          <textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Enter job requirements, skills, responsibilities..."
            className="brutalist-input flex-1 min-h-[120px] resize-none"
          />
        </div>
      </div>

      {/* Configuration */}
      <div className="mt-8 pt-6 border-t-4 border-black">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-5 h-5" />
          <h3 className="text-sm font-black tracking-widest uppercase text-black">
            CONFIGURATION
          </h3>
        </div>

        {/* Semantic Weight Slider */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-1">
            <label className="brutalist-label mb-0">SCORING WEIGHTS</label>
            <div className="bg-black text-white px-2 py-1 text-xs font-black tracking-widest">
              {(semanticWeight * 100).toFixed(0)}% SEM / {(skillWeight * 100).toFixed(0)}% SKL
            </div>
          </div>

          {/* Explanation */}
          <p className="text-xs text-gray-500 font-medium mb-3">
            Semantic = how well the resume meaning matches the JD. Skills = exact keyword matches. Slide left for deeper meaning analysis, right for strict skill matching.
          </p>

          <div className="relative pt-1">
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
                [&::-moz-range-thumb]:bg-black [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0
                [&::-moz-range-thumb]:cursor-pointer"
            />
          </div>

          <div className="flex justify-between text-xs font-bold tracking-widest text-gray-400 mt-2 uppercase">
            <span>← Semantic (meaning)</span>
            <span>Skills (keywords) →</span>
          </div>
        </div>

        {/* Fairness Analysis Toggle */}
        <div
          className="flex items-center justify-between border-4 border-black p-4 cursor-pointer hover:bg-gray-50 transition-colors"
          onClick={() => setIncludeFairness(!includeFairness)}
        >
          <div>
            <label className="brutalist-label mb-1 cursor-pointer">FAIRNESS ANALYSIS</label>
            <p className="text-xs text-gray-500 font-medium">
              Checks if rankings show bias patterns across candidate groups using the four-fifths rule.
            </p>
          </div>
          <div className="text-black ml-4 flex-shrink-0">
            {includeFairness ? (
              <CheckSquare className="w-8 h-8" />
            ) : (
              <Square className="w-8 h-8" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
