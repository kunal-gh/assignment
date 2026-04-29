'use client';

import { motion } from 'framer-motion';
import { useScreeningStore } from '@/store/screeningStore';
import CandidateCard from './CandidateCard';
import AnalyticsCharts from './AnalyticsCharts';
import { useState } from 'react';
import { BarChart2, Users, Download, ArrowLeft, AlertTriangle } from 'lucide-react';

interface ResultsViewProps {
  onBack: () => void;
}

export default function ResultsView({ onBack }: ResultsViewProps) {
  const { results, error } = useScreeningStore();
  const [activeTab, setActiveTab] = useState<'candidates' | 'analytics'>('candidates');

  // Error state
  if (error) {
    return (
      <div className="max-w-2xl mx-auto text-center py-20">
        <div className="brutalist-card p-8">
          <AlertTriangle className="w-16 h-16 mx-auto mb-4 text-red-500" />
          <h2 className="text-2xl font-black tracking-widest uppercase mb-4">Processing Error</h2>
          <p className="text-gray-600 mb-6 font-medium">{error}</p>
          <button onClick={onBack} className="brutalist-button py-4 px-8 mx-auto">
            <ArrowLeft className="w-5 h-5 mr-2" /> TRY AGAIN
          </button>
        </div>
      </div>
    );
  }

  // No results
  if (!results) {
    return (
      <div className="text-center py-20">
        <p className="text-black font-bold text-xl uppercase tracking-widest mb-6">No results available</p>
        <button onClick={onBack} className="brutalist-button py-4 px-8 mx-auto">
          <ArrowLeft className="w-5 h-5 mr-2" /> GO BACK
        </button>
      </div>
    );
  }

  const exportToCSV = () => {
    const headers = ['Rank', 'Name', 'Email', 'Overall Score (%)', 'Semantic Score (%)', 'Skill Score (%)', 'Experience (yrs)', 'Matched Skills', 'Missing Skills'];
    const rows = results.candidates.map((c) => [
      c.rank,
      c.name,
      c.email || '',
      (c.hybrid_score * 100).toFixed(1),
      (c.semantic_score * 100).toFixed(1),
      (c.skill_score * 100).toFixed(1),
      c.years_experience,
      c.matched_skills.join('; '),
      c.missing_skills.join('; '),
    ]);

    const csv = [headers, ...rows].map((row) => row.map((v) => `"${v}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `screening_${results.job_title.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="brutalist-card p-6 mb-8"
      >
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-3xl font-black text-black tracking-widest uppercase mb-2">
              SCREENING RESULTS
            </h2>
            <p className="text-black font-bold tracking-widest uppercase text-sm bg-gray-100 inline-block px-3 py-1 border-2 border-black">
              {results.job_title} · {results.successfully_parsed} RESUMES PROCESSED
            </p>
          </div>
          <button onClick={onBack} className="brutalist-button px-6 py-3 text-sm md:w-auto w-full">
            <ArrowLeft className="w-5 h-5 mr-2" /> NEW SCREENING
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          {[
            { label: 'TOTAL RESUMES', value: results.total_resumes, dark: false },
            { label: 'PARSED OK', value: results.successfully_parsed, dark: false },
            { label: 'TIME TAKEN', value: `${results.processing_time_seconds.toFixed(1)}s`, dark: false },
            {
              label: 'TOP MATCH',
              value: results.candidates.length > 0
                ? `${(results.candidates[0].hybrid_score * 100).toFixed(1)}%`
                : 'N/A',
              dark: true,
            },
          ].map(({ label, value, dark }) => (
            <div key={label} className={`border-4 border-black p-4 ${dark ? 'bg-black text-white' : 'bg-white'}`}>
              <p className={`text-xs font-black tracking-widest uppercase mb-1 ${dark ? 'text-white' : 'text-black'}`}>
                {label}
              </p>
              <p className={`text-3xl font-black ${dark ? 'text-white' : 'text-black'}`}>{value}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Tabs */}
      <div className="flex flex-col md:flex-row gap-4 mb-8 border-b-4 border-black pb-4">
        <button
          onClick={() => setActiveTab('candidates')}
          className={`px-6 py-3 font-black tracking-widest uppercase text-sm border-4 border-black flex items-center justify-center transition-all duration-100 ${
            activeTab === 'candidates' ? 'bg-black text-white shadow-brutal' : 'bg-white text-black hover:bg-gray-100'
          }`}
        >
          <Users className="w-5 h-5 mr-2" /> CANDIDATES ({results.candidates.length})
        </button>

        <button
          onClick={() => setActiveTab('analytics')}
          className={`px-6 py-3 font-black tracking-widest uppercase text-sm border-4 border-black flex items-center justify-center transition-all duration-100 ${
            activeTab === 'analytics' ? 'bg-black text-white shadow-brutal' : 'bg-white text-black hover:bg-gray-100'
          }`}
        >
          <BarChart2 className="w-5 h-5 mr-2" /> ANALYTICS
        </button>

        <button
          onClick={exportToCSV}
          className="md:ml-auto px-6 py-3 font-black tracking-widest uppercase text-sm border-4 border-black bg-white text-black hover:bg-gray-100 flex items-center justify-center transition-all duration-100"
        >
          <Download className="w-5 h-5 mr-2" /> EXPORT CSV
        </button>
      </div>

      {/* Content */}
      {activeTab === 'candidates' ? (
        <div className="space-y-6">
          {results.candidates.map((candidate, index) => (
            <CandidateCard key={candidate.rank} candidate={candidate} index={index} />
          ))}
        </div>
      ) : (
        <AnalyticsCharts results={results} />
      )}

      {/* Fairness */}
      {results.fairness_summary && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="brutalist-card p-6 mt-8"
        >
          <h3 className="text-xl font-black tracking-widest uppercase text-black mb-4">
            ⚖️ FAIRNESS ANALYSIS
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <p className="text-xs font-black tracking-widest uppercase text-black mb-2">OVERALL FAIRNESS SCORE</p>
              <div className="flex items-center space-x-4">
                <div className="flex-1 h-4 border-2 border-black bg-white overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${results.fairness_summary.overall_score * 100}%` }}
                    transition={{ duration: 1 }}
                    className="h-full bg-black"
                  />
                </div>
                <span className="text-2xl font-black text-black">
                  {(results.fairness_summary.overall_score * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            <div>
              {results.fairness_summary.bias_flags.length > 0 ? (
                <div>
                  <p className="text-xs font-black tracking-widest uppercase text-black mb-2">⚠️ POTENTIAL ISSUES</p>
                  <ul className="space-y-1">
                    {results.fairness_summary.bias_flags.map((flag, i) => (
                      <li key={i} className="text-sm text-red-600 font-medium">• {flag}</li>
                    ))}
                  </ul>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <span className="text-2xl">✅</span>
                  <p className="font-black tracking-widest uppercase text-sm">No significant bias detected</p>
                </div>
              )}
            </div>
          </div>
          {results.fairness_summary.recommendations.length > 0 && (
            <div className="mt-4 p-4 border-l-4 border-black bg-gray-50">
              <p className="text-xs font-black tracking-widest uppercase text-black mb-2">💡 RECOMMENDATIONS</p>
              <ul className="space-y-1">
                {results.fairness_summary.recommendations.map((rec, i) => (
                  <li key={i} className="text-sm text-gray-700">• {rec}</li>
                ))}
              </ul>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
