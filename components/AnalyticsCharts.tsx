'use client';

import { motion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar,
} from 'recharts';
import { ScreeningResults } from '@/store/screeningStore';

interface Props {
  results: ScreeningResults;
}

export default function AnalyticsCharts({ results }: Props) {
  const top10 = results.candidates.slice(0, 10);

  // Bar chart data
  const barData = top10.map((c) => ({
    name: c.name.length > 14 ? c.name.substring(0, 14) + '…' : c.name,
    Overall: parseFloat((c.hybrid_score * 100).toFixed(1)),
    Semantic: parseFloat((c.semantic_score * 100).toFixed(1)),
    Skills: parseFloat((c.skill_score * 100).toFixed(1)),
  }));

  // Radar data for top 3
  const top3 = results.candidates.slice(0, 3);
  const radarData = ['Overall', 'Semantic', 'Skills'].map((metric) => {
    const entry: Record<string, string | number> = { metric };
    top3.forEach((c, i) => {
      const score =
        metric === 'Overall' ? c.hybrid_score :
        metric === 'Semantic' ? c.semantic_score : c.skill_score;
      entry[`#${i + 1} ${c.name.split(' ')[0]}`] = parseFloat((score * 100).toFixed(1));
    });
    return entry;
  });

  const radarKeys = top3.map((c, i) => `#${i + 1} ${c.name.split(' ')[0]}`);
  const radarColors = ['#000000', '#555555', '#999999'];

  // Stats
  const scores = results.candidates.map((c) => c.hybrid_score);
  const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
  const max = Math.max(...scores);
  const min = Math.min(...scores);

  // Skill frequency
  const matchedCounts: Record<string, number> = {};
  const missingCounts: Record<string, number> = {};
  results.candidates.forEach((c) => {
    c.matched_skills.forEach((s) => { matchedCounts[s] = (matchedCounts[s] || 0) + 1; });
    c.missing_skills.forEach((s) => { missingCounts[s] = (missingCounts[s] || 0) + 1; });
  });
  const topMatched = Object.entries(matchedCounts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const topMissing = Object.entries(missingCounts).sort((a, b) => b[1] - a[1]).slice(0, 8);

  return (
    <div className="space-y-8">
      {/* Stats summary */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="brutalist-card p-6">
        <h3 className="text-xl font-black tracking-widest uppercase text-black mb-6">📊 STATISTICAL SUMMARY</h3>
        <div className="grid grid-cols-3 gap-6 text-center">
          {[
            { label: 'Average Score', value: `${(avg * 100).toFixed(1)}%`, color: 'text-black' },
            { label: 'Highest Score', value: `${(max * 100).toFixed(1)}%`, color: 'text-black' },
            { label: 'Lowest Score', value: `${(min * 100).toFixed(1)}%`, color: 'text-gray-500' },
          ].map(({ label, value, color }) => (
            <div key={label} className="border-4 border-black p-4">
              <p className="text-xs font-black tracking-widest uppercase text-gray-500 mb-2">{label}</p>
              <p className={`text-3xl font-black ${color}`}>{value}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Score distribution bar chart */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="brutalist-card p-6">
        <h3 className="text-xl font-black tracking-widest uppercase text-black mb-6">📈 SCORE DISTRIBUTION (TOP 10)</h3>
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={barData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="name" stroke="#374151" tick={{ fontSize: 11, fontWeight: 700 }} angle={-35} textAnchor="end" />
            <YAxis stroke="#374151" tick={{ fontSize: 11 }} domain={[0, 100]} unit="%" />
            <Tooltip formatter={(v: number) => `${v}%`} contentStyle={{ border: '2px solid black', borderRadius: 0 }} />
            <Legend wrapperStyle={{ fontWeight: 700, fontSize: 12 }} />
            <Bar dataKey="Overall" fill="#000000" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Semantic" fill="#555555" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Skills" fill="#999999" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Radar chart — top 3 */}
      {top3.length >= 2 && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="brutalist-card p-6">
          <h3 className="text-xl font-black tracking-widest uppercase text-black mb-6">🎯 TOP CANDIDATES COMPARISON</h3>
          <ResponsiveContainer width="100%" height={360}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#e5e7eb" />
              <PolarAngleAxis dataKey="metric" tick={{ fontWeight: 700, fontSize: 13 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
              {radarKeys.map((key, i) => (
                <Radar key={key} name={key} dataKey={key} stroke={radarColors[i]} fill={radarColors[i]} fillOpacity={0.25} />
              ))}
              <Legend wrapperStyle={{ fontWeight: 700, fontSize: 12 }} />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* Skills coverage */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="brutalist-card p-6">
        <h3 className="text-xl font-black tracking-widest uppercase text-black mb-6">🔧 SKILLS COVERAGE</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <p className="text-xs font-black tracking-widest uppercase text-black mb-3">MOST COMMON MATCHED</p>
            <div className="flex flex-wrap gap-2">
              {topMatched.map(([skill, count]) => (
                <span key={skill} className="px-3 py-1 border-2 border-black bg-white text-black text-xs font-bold uppercase">
                  {skill} ({count})
                </span>
              ))}
              {topMatched.length === 0 && <p className="text-sm text-gray-400">No matched skills data</p>}
            </div>
          </div>
          <div>
            <p className="text-xs font-black tracking-widest uppercase text-black mb-3">MOST COMMON MISSING</p>
            <div className="flex flex-wrap gap-2">
              {topMissing.map(([skill, count]) => (
                <span key={skill} className="px-3 py-1 border-2 border-red-400 bg-red-50 text-red-700 text-xs font-bold uppercase">
                  {skill} ({count})
                </span>
              ))}
              {topMissing.length === 0 && <p className="text-sm text-gray-400">No missing skills data</p>}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
