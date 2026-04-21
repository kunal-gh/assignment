'use client';

import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { ScreeningResults } from '@/store/screeningStore';

interface AnalyticsChartsProps {
  results: ScreeningResults;
}

export default function AnalyticsCharts({ results }: AnalyticsChartsProps) {
  // Prepare data for score distribution chart
  const scoreData = results.candidates.slice(0, 10).map(c => ({
    name: c.name.length > 15 ? c.name.substring(0, 15) + '...' : c.name,
    'Overall': (c.hybrid_score * 100).toFixed(1),
    'Semantic': (c.semantic_score * 100).toFixed(1),
    'Skills': (c.skill_score * 100).toFixed(1),
  }));

  // Prepare data for top 3 comparison radar
  const radarData = [
    { metric: 'Overall', ...Object.fromEntries(results.candidates.slice(0, 3).map((c, i) => [`Candidate ${i + 1}`, c.hybrid_score * 100])) },
    { metric: 'Semantic', ...Object.fromEntries(results.candidates.slice(0, 3).map((c, i) => [`Candidate ${i + 1}`, c.semantic_score * 100])) },
    { metric: 'Skills', ...Object.fromEntries(results.candidates.slice(0, 3).map((c, i) => [`Candidate ${i + 1}`, c.skill_score * 100])) },
  ];

  // Calculate statistics
  const scores = results.candidates.map(c => c.hybrid_score);
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  const maxScore = Math.max(...scores);
  const minScore = Math.min(...scores);

  return (
    <div className="space-y-8">
      {/* Statistics Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6"
      >
        <h3 className="text-2xl font-bold text-gray-800 mb-6">
          📊 Statistical Summary
        </h3>

        <div className="grid grid-cols-3 gap-6">
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-2">Average Score</p>
            <p className="text-4xl font-bold text-primary-600">
              {(avgScore * 100).toFixed(1)}%
            </p>
          </div>

          <div className="text-center">
            <p className="text-sm text-gray-600 mb-2">Highest Score</p>
            <p className="text-4xl font-bold text-green-600">
              {(maxScore * 100).toFixed(1)}%
            </p>
          </div>

          <div className="text-center">
            <p className="text-sm text-gray-600 mb-2">Lowest Score</p>
            <p className="text-4xl font-bold text-gray-600">
              {(minScore * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </motion.div>

      {/* Score Distribution Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-card p-6"
      >
        <h3 className="text-2xl font-bold text-gray-800 mb-6">
          📈 Score Distribution (Top 10)
        </h3>

        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={scoreData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="name" stroke="#6b7280" />
            <YAxis stroke="#6b7280" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: 'none',
                borderRadius: '12px',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
              }}
            />
            <Legend />
            <Bar dataKey="Overall" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
            <Bar dataKey="Semantic" fill="#6366f1" radius={[8, 8, 0, 0]} />
            <Bar dataKey="Skills" fill="#a78bfa" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Top 3 Comparison Radar */}
      {results.candidates.length >= 3 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-card p-6"
        >
          <h3 className="text-2xl font-bold text-gray-800 mb-6">
            🎯 Top 3 Candidates Comparison
          </h3>

          <ResponsiveContainer width="100%" height={400}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#e5e7eb" />
              <PolarAngleAxis dataKey="metric" stroke="#6b7280" />
              <PolarRadiusAxis stroke="#6b7280" />
              <Radar name={results.candidates[0].name} dataKey="Candidate 1" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
              <Radar name={results.candidates[1].name} dataKey="Candidate 2" stroke="#6366f1" fill="#6366f1" fillOpacity={0.6} />
              <Radar name={results.candidates[2].name} dataKey="Candidate 3" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.6} />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* Skills Analysis */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass-card p-6"
      >
        <h3 className="text-2xl font-bold text-gray-800 mb-6">
          🔧 Skills Coverage Analysis
        </h3>

        <div className="space-y-4">
          {/* Most Common Matched Skills */}
          <div>
            <p className="text-sm font-medium text-gray-600 mb-3">
              Most Common Matched Skills
            </p>
            <div className="flex flex-wrap gap-2">
              {(() => {
                const skillCounts: Record<string, number> = {};
                results.candidates.forEach(c => {
                  c.matched_skills.forEach(skill => {
                    skillCounts[skill] = (skillCounts[skill] || 0) + 1;
                  });
                });
                
                return Object.entries(skillCounts)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 10)
                  .map(([skill, count]) => (
                    <div
                      key={skill}
                      className="px-4 py-2 bg-green-100 text-green-700 rounded-full text-sm font-medium"
                    >
                      {skill} ({count})
                    </div>
                  ));
              })()}
            </div>
          </div>

          {/* Most Common Missing Skills */}
          <div>
            <p className="text-sm font-medium text-gray-600 mb-3">
              Most Common Missing Skills
            </p>
            <div className="flex flex-wrap gap-2">
              {(() => {
                const skillCounts: Record<string, number> = {};
                results.candidates.forEach(c => {
                  c.missing_skills.forEach(skill => {
                    skillCounts[skill] = (skillCounts[skill] || 0) + 1;
                  });
                });
                
                return Object.entries(skillCounts)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 10)
                  .map(([skill, count]) => (
                    <div
                      key={skill}
                      className="px-4 py-2 bg-red-100 text-red-700 rounded-full text-sm font-medium"
                    >
                      {skill} ({count})
                    </div>
                  ));
              })()}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
