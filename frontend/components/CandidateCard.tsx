'use client';

import { motion } from 'framer-motion';
import { Candidate } from '@/store/screeningStore';

interface CandidateCardProps {
  candidate: Candidate;
  index: number;
}

export default function CandidateCard({ candidate, index }: CandidateCardProps) {
  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-500';
    if (score >= 0.6) return 'bg-blue-500';
    if (score >= 0.4) return 'bg-yellow-500';
    return 'bg-gray-400';
  };

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
      transition={{ delay: index * 0.1 }}
      className="candidate-card"
    >
      <div className="flex items-start justify-between mb-4">
        {/* Rank Badge */}
        <div className="flex items-center space-x-4">
          <div className={`
            w-12 h-12 rounded-full flex items-center justify-center font-bold text-xl
            ${candidate.rank <= 3 ? 'bg-gradient-to-br from-yellow-400 to-yellow-600 text-white' : 'bg-gray-200 text-gray-700'}
          `}>
            #{candidate.rank}
          </div>

          {/* Name & Email */}
          <div>
            <h3 className="text-xl font-bold text-gray-800">
              {candidate.name}
            </h3>
            {candidate.email && (
              <p className="text-sm text-gray-500">
                📧 {candidate.email}
              </p>
            )}
          </div>
        </div>

        {/* Overall Score */}
        <div className="text-right">
          <div className={`
            score-circle ${getScoreColor(candidate.hybrid_score)}
            text-white
          `}>
            {getScoreGrade(candidate.hybrid_score)}
          </div>
          <p className="text-sm text-gray-600 mt-2">
            {(candidate.hybrid_score * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Score Breakdown */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-primary-50 rounded-xl p-3">
          <p className="text-xs text-primary-600 font-medium mb-1">
            Semantic Match
          </p>
          <div className="flex items-center space-x-2">
            <div className="flex-1 h-2 bg-white rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${candidate.semantic_score * 100}%` }}
                transition={{ duration: 1, delay: index * 0.1 + 0.3 }}
                className="h-full bg-primary-500"
              />
            </div>
            <span className="text-sm font-bold text-primary-700">
              {(candidate.semantic_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        <div className="bg-purple-50 rounded-xl p-3">
          <p className="text-xs text-purple-600 font-medium mb-1">
            Skill Match
          </p>
          <div className="flex items-center space-x-2">
            <div className="flex-1 h-2 bg-white rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${candidate.skill_score * 100}%` }}
                transition={{ duration: 1, delay: index * 0.1 + 0.4 }}
                className="h-full bg-purple-500"
              />
            </div>
            <span className="text-sm font-bold text-purple-700">
              {(candidate.skill_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Skills */}
      <div className="space-y-3">
        {/* Matched Skills */}
        {candidate.matched_skills.length > 0 && (
          <div>
            <p className="text-xs font-medium text-gray-600 mb-2">
              ✅ Matched Skills ({candidate.matched_skills.length})
            </p>
            <div className="flex flex-wrap gap-2">
              {candidate.matched_skills.slice(0, 8).map((skill, i) => (
                <span
                  key={i}
                  className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium"
                >
                  {skill}
                </span>
              ))}
              {candidate.matched_skills.length > 8 && (
                <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-xs font-medium">
                  +{candidate.matched_skills.length - 8} more
                </span>
              )}
            </div>
          </div>
        )}

        {/* Missing Skills */}
        {candidate.missing_skills.length > 0 && (
          <div>
            <p className="text-xs font-medium text-gray-600 mb-2">
              ⚠️ Missing Skills ({candidate.missing_skills.length})
            </p>
            <div className="flex flex-wrap gap-2">
              {candidate.missing_skills.slice(0, 5).map((skill, i) => (
                <span
                  key={i}
                  className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-xs font-medium"
                >
                  {skill}
                </span>
              ))}
              {candidate.missing_skills.length > 5 && (
                <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-xs font-medium">
                  +{candidate.missing_skills.length - 5} more
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Experience */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-sm text-gray-600">
          <span className="font-medium">Experience:</span> {candidate.years_experience} years
        </p>
      </div>

      {/* Explanation */}
      {candidate.explanation && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          transition={{ delay: index * 0.1 + 0.5 }}
          className="mt-4 p-4 bg-gray-50 rounded-xl"
        >
          <p className="text-sm text-gray-700 leading-relaxed">
            {candidate.explanation}
          </p>
        </motion.div>
      )}
    </motion.div>
  );
}
