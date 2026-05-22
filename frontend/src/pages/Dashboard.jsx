import React, { useContext } from 'react';
import { AppContext } from '../App';
import { Link } from 'react-router-dom';
import { Activity, Brain, ArrowRight, AlertCircle } from 'lucide-react';

const EMOTION_COLORS = {
  happy:   '#10b981',
  neutral: '#6366f1',
  sad:     '#64748b',
  fear:    '#f59e0b',
  anger:   '#ef4444',
  disgust: '#8b5cf6',
};

const EMOTION_EMOJI = {
  happy:   '😊',
  neutral: '😐',
  sad:     '😢',
  fear:    '😨',
  anger:   '😠',
  disgust: '🤢',
};

const STRESS_COLORS = { high: 'var(--danger)', low: 'var(--success)', moderate: 'var(--warning)' };

export default function Dashboard() {
  const { analysisData } = useContext(AppContext);

  if (!analysisData) {
    return (
      <div style={{ textAlign: 'center', marginTop: '4rem' }}>
        <AlertCircle size={56} color="var(--text-muted)" style={{ margin: '0 auto 1rem' }} />
        <h1 className="page-title">Analysis Dashboard</h1>
        <p className="page-subtitle mb-8">No analysis data yet. Run an analysis first.</p>
        <Link to="/upload" className="btn btn-primary">Go to Input & Analyze</Link>
      </div>
    );
  }

  const { emotion, stress_level, emotion_probabilities, features, emotion_source, model_used } = analysisData;
  const formatValue = (val) => val != null ? Number(val).toFixed(2) : '—';
  const emotionColor = EMOTION_COLORS[emotion] || 'var(--primary)';
  const stressColor  = STRESS_COLORS[stress_level] || 'var(--primary)';
  
  const isHrIbiModel = emotion_source === 'hr_ibi_model';
  const isDemo = model_used === 'demo';

  // Top emotion probability
  const topProb = emotion_probabilities?.[emotion];
  const confidencePct = topProb != null ? (topProb * 100).toFixed(1) : null;

  return (
    <div>
      <h1 className="page-title">Analysis Dashboard</h1>
      <p className="page-subtitle">Results from the Emotion Classifier models.</p>

      {/* Info: Demo mode */}
      {isDemo && (
        <div style={{
          display: 'flex', alignItems: 'flex-start', gap: '0.75rem',
          backgroundColor: '#eff6ff', border: '1px solid #bfdbfe',
          borderLeft: '4px solid #3b82f6', borderRadius: '8px',
          padding: '0.85rem 1.25rem', marginBottom: '1.5rem'
        }}>
          <AlertCircle size={20} color="#2563eb" style={{ flexShrink: 0, marginTop: '0.1rem' }} />
          <div>
            <strong style={{ color: '#1e40af', display: 'block', marginBottom: '0.2rem' }}>
              Demo Mode Active
            </strong>
            <span style={{ fontSize: '0.875rem', color: '#1e3a8a', lineHeight: '1.5' }}>
              No files were uploaded. These results use sample data to demonstrate the system.
            </span>
          </div>
        </div>
      )}

      {/* PRIMARY: Emotion — full-width hero card */}
      <div className="card mb-6" style={{
        borderTop: `6px solid ${emotionColor}`,
        background: `linear-gradient(135deg, ${emotionColor}10 0%, var(--surface) 100%)`,
        padding: '2.5rem',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1.5rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <Brain size={20} color={emotionColor} />
              <span style={{ fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: emotionColor }}>
                Primary Result — Detected Emotion
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <span style={{ fontSize: '4.5rem', lineHeight: 1 }}>{EMOTION_EMOJI[emotion] || '🧠'}</span>
              <div>
                <div style={{ fontSize: '3.5rem', fontWeight: 900, textTransform: 'capitalize', color: emotionColor, lineHeight: 1 }}>
                  {emotion}
                </div>
                {confidencePct && (
                  <div style={{ fontSize: '1rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    Model confidence: <strong style={{ color: emotionColor }}>{confidencePct}%</strong>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Mini probability bars */}
          {emotion_probabilities && (
            <div style={{ minWidth: '220px' }}>
              <p style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                All Emotion Probabilities
              </p>
              {Object.entries(emotion_probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([emo, prob]) => (
                  <div key={emo} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
                    <span style={{ width: '60px', fontSize: '0.8rem', textTransform: 'capitalize', color: emo === emotion ? EMOTION_COLORS[emo] : 'var(--text-muted)', fontWeight: emo === emotion ? 700 : 400 }}>
                      {emo}
                    </span>
                    <div style={{ flex: 1, height: '8px', backgroundColor: 'var(--border)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{ width: `${(prob * 100).toFixed(0)}%`, height: '100%', backgroundColor: EMOTION_COLORS[emo] || '#6366f1', opacity: emo === emotion ? 1 : 0.4, borderRadius: '4px' }} />
                    </div>
                    <span style={{ fontSize: '0.75rem', width: '38px', textAlign: 'right', color: 'var(--text-muted)' }}>
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              <Link to="/emotions" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', color: 'var(--primary)', fontSize: '0.85rem', fontWeight: 600, marginTop: '0.75rem' }}>
                Full chart <ArrowRight size={14} />
              </Link>
            </div>
          )}
        </div>
      </div>

      {/* SECONDARY: Stress — compact banner */}
      <div className="card mb-8" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1.25rem 1.75rem', borderLeft: `4px solid ${stressColor}`, flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
            Interpretation — Estimated Stress Level
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ fontSize: '2rem', fontWeight: 800, textTransform: 'capitalize', color: stressColor }}>{stress_level}</span>
            <span className={`badge badge-${stress_level === 'high' ? 'high' : 'low'}`} style={{ fontSize: '0.8rem' }}>
              {stress_level === 'high' ? 'Elevated Stress' : 'Low / Baseline Stress'}
            </span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Derived directly from the primary detected emotional state. <Link to="/logic" style={{ color: stressColor, fontWeight: 600 }}>How is this mapped?</Link>
          </p>
        </div>
      </div>

      {/* Physiological Feature Summary */}
      <h2 className="mb-4" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.1rem' }}>
        <Activity size={20} color="var(--primary)" /> Extracted Physiological Features
      </h2>
      <div className="grid-4 mb-6">
        {[
          { label: 'HR Mean', value: formatValue(features?.hr_mean), unit: 'bpm' },
          { label: 'HR Std Dev', value: formatValue(features?.hr_std), unit: '' },
          { label: 'IBI Mean', value: formatValue(features?.ibi_mean), unit: 's' },
          { label: 'IBI RMSSD', value: formatValue(features?.ibi_rmssd), unit: 's' },
          { label: 'SDNN', value: formatValue(features?.ibi_sdnn), unit: 's' },
          { label: 'pNN50', value: formatValue(features?.ibi_pnn50), unit: '' },
          { label: 'HR > 80 ratio', value: formatValue(features?.hr_above_80_ratio), unit: '' },
          { label: 'HR < 60 ratio', value: formatValue(features?.hr_below_60_ratio), unit: '' },
        ].map(f => (
          <div className="card" key={f.label} style={{ padding: '1rem' }}>
            <h4 className="text-muted mb-1" style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{f.label}</h4>
            <div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{f.value} <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{f.unit}</span></div>
          </div>
        ))}
      </div>

      <div className="card" style={{ backgroundColor: 'var(--primary-light)', borderColor: 'var(--primary)', color: 'var(--primary)', padding: '1rem 1.5rem' }}>
        <p style={{ margin: 0, fontSize: '0.9rem' }}>
          <strong>Note:</strong> The ECG Emotion model extracts 30 features, while the HR+IBI Emotion model extracts 243 features.{' '}
          <Link to="/report" style={{ textDecoration: 'underline', fontWeight: 'bold', color: 'var(--primary)' }}>See the full Report</Link> for all extracted values.
        </p>
      </div>
    </div>
  );
}
