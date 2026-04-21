'use client';

import { motion } from 'framer-motion';
import { useScreeningStore } from '@/store/screeningStore';
import CandidateCard from './CandidateCard';
import AnalyticsCharts from './AnalyticsCharts';
import { useState } from 'react';

interface ResultsViewProps {
  onBack: () => void;
}

export default function ResultsView({ onBack }: ResultsViewProps) {
  const { results } = useScreeningStore();
  const [activeTab, setActiveTab] = useState<'candidates' | 'analytics'>('candidates');

  if (!results) {
    return (
      <div className="text-center py-20">
        <p className="text-gray-500 text-lg">No results available</p>
        <button onClick={onBack} className="btn-primary mt-4">
          Go Back
        </button>
      </div>
    );
  }

  const exportToCSV = () => {
    const headers = ['Rank', 'Name', 'Email', 'Overall Score', 'Semantic Score', 'Skill Score', 'Experience'];
    const rows = results.candidates.map(c => [
      c.rank,
      c.name,
      c.email || '',
      (c.hybrid_score * 100).toFixed(1),
      (c.semantic_score * 100).toFixed(1),
      (c.skill_score * 100).toFixed(1),
      c.years_experience
    ]);

    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `screening_results_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6 mb-8"
      >
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold text-gray-800 mb-2">
              📊 Screening Results
            </h2>
            <p className="text-gray-600">
              {results.job_title} • {results.successfully_parsed} candidates processed
            </p>
          </div>

          <button
            onClick={onBack}
            className="btn-secondary"
          >
            ← Back
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="bg-primary-50 rounded-xl p-4">
            <p className="text-sm text-primary-600 font-medium mb-1">
              Total Resumes
            </p>
            <p className="text-3xl font-bold text-primary-700">
              {results.total_resumes}
            </p>
          </div>

          <div className="bg-green-50 rounded-xl p-4">
            <p className="text-sm text-green-600 font-medium mb-1">
              Successfully Parsed
            </p>
            <p className="text-3xl font-bold text-green-700">
              {results.successfully_parsed}
            </p>
          </div>

          <div className="bg-blue-50 rounded-xl p-4">
            <p className="text-sm text-blue-600 font-medium mb-1">
              Processing Time
            </p>
            <p className="text-3xl font-bold text-blue-700">
              {results.processing_time_seconds.toFixed(1)}s
            </p>
          </div>

          <div className="bg-purple-50 rounded-xl p-4">
            <p className="text-sm text-purple-600 font-medium mb-1">
              Top Score
            </p>
            <p className="text-3xl font-bold text-purple-700">
              {results.candidates.length > 0 
                ? (results.candidates[0].hybrid_score * 100).toFixed(1) + '%'
                : 'N/A'
              }
            </p>
          </div>
        </div>
      </motion.div>

      {/* Tabs */}
      <div className="flex space-x-2 mb-6">
        <button
          onClick={() => setActiveTab('candidates')}
          className={`
            px-6 py-3 rounded-xl font-medium transition-all duration-200
            ${activeTab === 'candidates'
              ? 'bg-primary-500 text-white shadow-lg'
              : 'bg-white text-gray-600 hover:bg-gray-50'
            }
          `}
        >
          👥 Candidates ({results.candidates.length})
        </button>

        <button
          onClick={() => setActiveTab('analytics')}
          className={`
            px-6 py-3 rounded-xl font-medium transition-all duration-200
            ${activeTab === 'analytics'
              ? 'bg-primary-500 text-white shadow-lg'
              : 'bg-white text-gray-600 hover:bg-gray-50'
            }
          `}
        >
          📈 Analytics
        </button>

        <button
          onClick={exportToCSV}
          className="ml-auto btn-secondary"
        >
          📥 Export CSV
        </button>
      </div>

      {/* Content */}
      {activeTab === 'candidates' ? (
        <div className="space-y-4">
          {results.candidates.map((candidate, index) => (
            <CandidateCard
              key={candidate.rank}
              candidate={candidate}
              index={index}
            />
          ))}
        </div>
      ) : (
        <AnalyticsCharts results={results} />
      )}

      {/* Fairness Summary */}
      {results.fairness_summary && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-card p-6 mt-8"
        >
          <h3 className="text-2xl font-bold text-gray-800 mb-4">
            ⚖️ Fairness Analysis
          </h3>

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <p className="text-sm text-gray-600 mb-2">Overall Fairness Score</p>
              <div className="flex items-center space-x-4">
                <div className="flex-1 h-4 bg-gray-200 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${results.fairness_summary.overall_score * 100}%` }}
                    transition={{ duration: 1, delay: 0.6 }}
                    className="h-full bg-green-500"
                  />
                </div>
                <span className="text-2xl font-bold text-gray-800">
                  {(results.fairness_summary.overall_score * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            <div>
              {results.fairness_summary.bias_flags.length > 0 ? (
                <div>
                  <p className="text-sm text-gray-600 mb-2">⚠️ Potential Issues</p>
                  <ul className="space-y-1">
                    {results.fairness_summary.bias_flags.map((flag, i) => (
                      <li key={i} className="text-sm text-red-600">
                        • {flag}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <div className="flex items-center space-x-2 text-green-600">
                  <span className="text-2xl">✅</span>
                  <p className="font-medium">No significant bias detected</p>
                </div>
              )}
            </div>
          </div>

          {results.fairness_summary.recommendations.length > 0 && (
            <div className="mt-6 p-4 bg-blue-50 rounded-xl">
              <p className="text-sm font-medium text-blue-800 mb-2">
                💡 Recommendations
              </p>
              <ul className="space-y-1">
                {results.fairness_summary.recommendations.map((rec, i) => (
                  <li key={i} className="text-sm text-blue-700">
                    • {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
