'use client';

import { motion } from 'framer-motion';
import { Candidate } from '@/store/screeningStore';
import { Mail, CheckCircle, XCircle, Clock } from 'lucide-react';

interface CandidateCardProps {
  candidate: Candidate;
  index: number;
}

export default function CandidateCard({ candidate, index }: CandidateCardProps) {
  const getScoreGrade = (score: number) => {
    if (score >= 0.9) return 'A+';
    if (score >= 0.8) return 'A';
    if (score >= 0.7) return 'B+';
    if (score >= 0.6) return 'B';
    if (score >= 0.5) return 'C+';
    if (score >= 0.4) return 'C';
    return 'D';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08 }}
      className="brutalist-card p-6"
    >
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start justify-between mb-6 border-b-4 border-black pb-6 gap-6">
        <div className="flex items-center space-x-6">
          <div className={`
            w-16 h-16 flex items-center justify-center font-black text-2xl border-4 border-black flex-shrink-0
            ${candidate.rank <= 3 ? 'bg-black text-white shadow-brutal-sm' : 'bg-white text-black'}
          `}>
            #{candidate.rank}
          </div>
          <div>
            <h3 className="text-2xl font-black text-black tracking-widest uppercase mb-1">
              {candidate.name}
            </h3>
            {candidate.email && (
              <p className="text-sm font-bold tracking-widest uppercase text-gray-500 flex items-center gap-2">
                <Mail className="w-4 h-4" /> {candidate.email}
              </p>
            )}
            <p className="text-sm font-bold tracking-widest text-gray-400 flex items-center gap-2 mt-1">
              <Clock className="w-4 h-4" /> {candidate.years_experience} yrs experience
            </p>
          </div>
        </div>

        {/* Overall Score */}
        <div className="flex items-center gap-4 flex-shrink-0">
          <div className="text-right">
            <p className="text-xs font-black tracking-widest uppercase text-gray-500 mb-1">
              OVERALL MATCH
            </p>
            <p className="text-3xl font-black text-black">
              {(candidate.hybrid_score * 100).toFixed(1)}%
            </p>
          </div>
          <div className="w-16 h-16 border-4 border-black bg-black text-white flex items-center justify-center font-black text-2xl shadow-brutal-sm">
            {getScoreGrade(candidate.hybrid_score)}
          </div>
        </div>
      </div>

      {/* Score Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="border-4 border-black p-4 bg-gray-50">
          <p className="text-xs font-black tracking-widest uppercase text-black mb-2">
            SEMANTIC MATCH
          </p>
          <div className="flex items-center space-x-4">
            <div className="flex-1 h-4 border-2 border-black bg-white overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${candidate.semantic_score * 100}%` }}
                transition={{ duration: 1, delay: index * 0.08 + 0.3 }}
                className="h-full bg-black"
              />
            </div>
            <span className="text-lg font-black text-black w-12 text-right">
              {(candidate.semantic_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        <div className="border-4 border-black p-4 bg-gray-50">
          <p className="text-xs font-black tracking-widest uppercase text-black mb-2">
            SKILL MATCH
          </p>
          <div className="flex items-center space-x-4">
            <div className="flex-1 h-4 border-2 border-black bg-white overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${candidate.skill_score * 100}%` }}
                transition={{ duration: 1, delay: index * 0.08 + 0.4 }}
                className="h-full bg-black"
              />
            </div>
            <span className="text-lg font-black text-black w-12 text-right">
              {(candidate.skill_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Skills */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {candidate.matched_skills.length > 0 && (
          <div>
            <p className="text-xs font-black tracking-widest uppercase text-black mb-2 flex items-center gap-1">
              <CheckCircle className="w-4 h-4 text-green-600" /> MATCHED SKILLS
            </p>
            <div className="flex flex-wrap gap-2">
              {candidate.matched_skills.slice(0, 6).map((skill, i) => (
                <span
                  key={i}
                  className="px-2 py-1 border-2 border-black bg-white text-black text-xs font-bold tracking-wide uppercase"
                >
                  {skill}
                </span>
              ))}
              {candidate.matched_skills.length > 6 && (
                <span className="px-2 py-1 border-2 border-gray-300 text-gray-500 text-xs font-bold">
                  +{candidate.matched_skills.length - 6}
                </span>
              )}
            </div>
          </div>
        )}

        {candidate.missing_skills.length > 0 && (
          <div>
            <p className="text-xs font-black tracking-widest uppercase text-black mb-2 flex items-center gap-1">
              <XCircle className="w-4 h-4 text-red-500" /> MISSING SKILLS
            </p>
            <div className="flex flex-wrap gap-2">
              {candidate.missing_skills.slice(0, 6).map((skill, i) => (
                <span
                  key={i}
                  className="px-2 py-1 border-2 border-red-300 bg-red-50 text-red-700 text-xs font-bold tracking-wide uppercase"
                >
                  {skill}
                </span>
              ))}
              {candidate.missing_skills.length > 6 && (
                <span className="px-2 py-1 border-2 border-gray-300 text-gray-500 text-xs font-bold">
                  +{candidate.missing_skills.length - 6}
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Explanation */}
      {candidate.explanation && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: index * 0.08 + 0.5 }}
          className="mt-4 p-4 border-l-4 border-black bg-gray-50"
        >
          <p className="text-xs font-black tracking-widest uppercase text-black mb-1">AI ANALYSIS</p>
          <p className="text-sm text-gray-700 leading-relaxed">
            {candidate.explanation}
          </p>
        </motion.div>
      )}
    </motion.div>
  );
}
