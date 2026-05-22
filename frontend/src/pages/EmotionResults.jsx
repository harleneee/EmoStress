import React, { useContext } from 'react';
import { AppContext } from '../App';
import { Link } from 'react-router-dom';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend
} from 'recharts';

const EMOTION_COLORS = {
  happy: '#10b981',
  neutral: '#6366f1',
  sad: '#64748b',
  fear: '#f59e0b',
  anger: '#ef4444',
  disgust: '#8b5cf6',
};

const STRESS_COLORS = {
  low: '#10b981',
  high: '#ef4444',
};

export default function EmotionResults() {
  const { analysisData } = useContext(AppContext);

  if (!analysisData) {
    return (
      <div className="text-center mt-8">
        <h1 className="page-title">Emotion Results</h1>
        <p className="page-subtitle mb-8">No data available. Please run an analysis first.</p>
        <Link to="/upload" className="btn btn-primary">Go to Input & Analyze</Link>
      </div>
    );
  }

  const { emotion, stress_level, emotion_probabilities, stress_probabilities } = analysisData;

  const emotionChartData = Object.entries(emotion_probabilities || {})
    .map(([key, val]) => ({
      name: key.charAt(0).toUpperCase() + key.slice(1),
      key,
      probability: parseFloat((val * 100).toFixed(1)),
      isMax: key === emotion
    }))
    .sort((a, b) => b.probability - a.probability);

  const stressChartData = Object.entries(stress_probabilities || {})
    .map(([key, val]) => ({
      name: key.charAt(0).toUpperCase() + key.slice(1) + ' Stress',
      key,
      probability: parseFloat((val * 100).toFixed(1)),
    }));

  const stressColor = STRESS_COLORS[stress_level] || '#6366f1';
  const emotionColor = EMOTION_COLORS[emotion] || 'var(--primary)';

  return (
    <div>
      <h1 className="page-title">Emotion Results</h1>
      <p className="page-subtitle">Probability distributions from both classification models.</p>

      {/* Top summary row */}
      <div className="grid-2 mb-8">
        <div className="card" style={{ textAlign: 'center', borderTop: `4px solid ${emotionColor}` }}>
          <p className="text-muted mb-2" style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Top Detected Emotion</p>
          <div style={{ fontSize: '2.5rem', fontWeight: 800, textTransform: 'capitalize', color: emotionColor }}>{emotion}</div>
          <div className="mt-2" style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
            Confidence: <strong>{emotionChartData[0]?.probability}%</strong>
          </div>
        </div>

        <div className="card" style={{ textAlign: 'center', borderTop: `4px solid ${stressColor}` }}>
          <p className="text-muted mb-2" style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Estimated Stress Level</p>
          <div style={{ fontSize: '2.5rem', fontWeight: 800, textTransform: 'capitalize', color: stressColor }}>{stress_level}</div>
          <div className="mt-2" style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
            Confidence: <strong>{stressChartData.find(d => d.key === stress_level)?.probability || '–'}%</strong>
          </div>
        </div>
      </div>

      {/* Emotion Bar Chart */}
      <div className="card mb-8">
        <h3 className="mb-6">Emotion Probability Distribution</h3>
        <div style={{ width: '100%', height: 300 }}>
          <ResponsiveContainer>
            <BarChart data={emotionChartData} layout="vertical" margin={{ top: 5, right: 40, left: 70, bottom: 5 }}>
              <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 12 }} />
              <YAxis dataKey="name" type="category" tick={{ fontSize: 13, fontWeight: 500 }} />
              <Tooltip formatter={value => [`${value}%`, 'Confidence']} />
              <Bar dataKey="probability" radius={[0, 6, 6, 0]} maxBarSize={32}>
                {emotionChartData.map((entry) => (
                  <Cell
                    key={entry.key}
                    fill={EMOTION_COLORS[entry.key] || '#6366f1'}
                    opacity={entry.isMax ? 1 : 0.4}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Stress Bar Chart */}
      <div className="card mb-8">
        <h3 className="mb-6">Stress Level Confidence</h3>
        <div style={{ width: '100%', height: 160 }}>
          <ResponsiveContainer>
            <BarChart data={stressChartData} layout="vertical" margin={{ top: 5, right: 40, left: 100, bottom: 5 }}>
              <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 12 }} />
              <YAxis dataKey="name" type="category" tick={{ fontSize: 13, fontWeight: 500 }} />
              <Tooltip formatter={value => [`${value}%`, 'Confidence']} />
              <Bar dataKey="probability" radius={[0, 6, 6, 0]} maxBarSize={32}>
                {stressChartData.map(entry => (
                  <Cell key={entry.key} fill={STRESS_COLORS[entry.key] || '#6366f1'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ textAlign: 'center' }}>
        <Link to="/report" className="btn btn-primary" style={{ marginRight: '1rem' }}>View Full Report</Link>
        <Link to="/logic" className="btn btn-secondary" style={{ backgroundColor: 'white' }}>How Stress is Calculated</Link>
      </div>
    </div>
  );
}
