import React, { useContext } from 'react';
import { AppContext } from '../App';
import { Link } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function EmotionResults() {
  const { analysisData } = useContext(AppContext);

  if (!analysisData) {
    return (
      <div className="text-center mt-8">
        <h1 className="page-title">Emotion Probabilities</h1>
        <p className="page-subtitle mb-8">No data available. Please upload signals first.</p>
        <Link to="/upload" className="btn btn-primary">Go to Upload Page</Link>
      </div>
    );
  }

  const { emotion, emotion_probabilities } = analysisData;

  // Convert dictionary to array for recharts
  const chartData = Object.keys(emotion_probabilities).map(key => ({
    name: key.charAt(0).toUpperCase() + key.slice(1),
    probability: (emotion_probabilities[key] * 100).toFixed(1),
    isMax: key === emotion
  }));

  // Sort by probability descending
  chartData.sort((a, b) => b.probability - a.probability);

  return (
    <div>
      <h1 className="page-title">Emotion Probabilities</h1>
      <p className="page-subtitle">Confidence scores outputted by the Random Forest emotion classification model.</p>

      <div className="card mb-8">
        <h3 className="mb-4 text-muted">Detected Emotion: <span style={{ color: 'var(--primary)', fontWeight: 'bold', textTransform: 'capitalize' }}>{emotion}</span></h3>
        <p className="mb-8">The model is <strong>{chartData[0].probability}%</strong> confident that the current state is {emotion}.</p>
        
        <div style={{ width: '100%', height: 400 }}>
          <ResponsiveContainer>
            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <XAxis type="number" domain={[0, 100]} tickFormatter={(val) => `${val}%`} />
              <YAxis dataKey="name" type="category" width={80} />
              <Tooltip formatter={(value) => `${value}%`} />
              <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.isMax ? 'var(--primary)' : 'var(--primary-light)'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
