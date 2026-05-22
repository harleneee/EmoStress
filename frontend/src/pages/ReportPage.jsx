import React, { useContext } from 'react';
import { AppContext } from '../App';
import { Link } from 'react-router-dom';
import { FileText, Printer, Clock, AlertCircle, Brain, Activity, HeartPulse } from 'lucide-react';

const EMOTION_COLORS = {
  happy: '#10b981', neutral: '#6366f1', sad: '#64748b',
  fear: '#f59e0b', anger: '#ef4444', disgust: '#8b5cf6',
};
const EMOTION_EMOJI = {
  happy: '😊', neutral: '😐', sad: '😢', fear: '😨', anger: '😠', disgust: '🤢',
};
const STRESS_COLORS = { high: '#ef4444', low: '#10b981', moderate: '#f59e0b' };

export default function ReportPage() {
  const { analysisData, uploadedFiles } = useContext(AppContext);

  if (!analysisData) {
    return (
      <div>
        <h1 className="page-title">Analysis Report</h1>
        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
          <AlertCircle size={48} color="var(--text-muted)" style={{ margin: '0 auto 1rem' }} />
          <h3>No Analysis Data Available</h3>
          <p className="text-muted mt-2">Please run an analysis on the Input & Analyze page first.</p>
          <Link to="/upload" className="btn btn-primary" style={{ marginTop: '1.5rem', display: 'inline-flex' }}>
            Go to Input & Analyze
          </Link>
        </div>
      </div>
    );
  }

  // Correct field names matching backend response
  const {
    emotion,
    stress_level,
    emotion_probabilities,
    stress_probabilities,
    features,
    model_used,
  } = analysisData;

  const dateStr = new Date().toLocaleString();
  const emotionColor = EMOTION_COLORS[emotion] || 'var(--primary)';
  const stressColor  = STRESS_COLORS[stress_level] || 'var(--primary)';
  const topEmotionPct = emotion_probabilities?.[emotion]
    ? (emotion_probabilities[emotion] * 100).toFixed(1)
    : null;

  const hasFiles = uploadedFiles && Object.keys(uploadedFiles).length > 0;

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 className="page-title mb-1" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <FileText size={28} color="var(--primary)" /> Analysis Report
          </h1>
          <p className="text-muted" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
            <Clock size={14} /> Generated on {dateStr}
            {model_used && (
              <span style={{ marginLeft: '1rem', backgroundColor: model_used === 'demo' ? '#fef3c7' : '#d1fae5', color: model_used === 'demo' ? '#92400e' : '#065f46', padding: '0.15rem 0.6rem', borderRadius: '99px', fontSize: '0.75rem', fontWeight: 600 }}>
                {model_used === 'demo' ? 'Demo Mode' : 'Live Model'}
              </span>
            )}
          </p>
        </div>
        <button
          className="btn btn-primary no-print"
          onClick={() => window.print()}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <Printer size={18} /> Print / Save PDF
        </button>
      </div>

      {/* Primary Result: Emotion */}
      <div className="card mb-6" style={{
        borderTop: `6px solid ${emotionColor}`,
        background: `linear-gradient(135deg, ${emotionColor}12 0%, var(--surface) 100%)`,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
          <Brain size={18} color={emotionColor} />
          <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: emotionColor }}>
            Primary Finding — Detected Emotion
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', marginBottom: '1.5rem' }}>
          <span style={{ fontSize: '3.5rem' }}>{EMOTION_EMOJI[emotion] || '🧠'}</span>
          <div>
            <div style={{ fontSize: '3rem', fontWeight: 900, textTransform: 'capitalize', color: emotionColor, lineHeight: 1 }}>
              {emotion || 'Unknown'}
            </div>
            {topEmotionPct && (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
                Model confidence: <strong style={{ color: emotionColor }}>{topEmotionPct}%</strong>
              </div>
            )}
          </div>
        </div>

        {/* Emotion probability bars */}
        {emotion_probabilities && (
          <>
            <p style={{ fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
              All Emotion Probabilities
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '0.6rem' }}>
              {Object.entries(emotion_probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([emo, prob]) => {
                  const pct = (prob * 100).toFixed(1);
                  const isTop = emo === emotion;
                  return (
                    <div key={emo} style={{
                      backgroundColor: isTop ? emotionColor + '18' : 'var(--surface)',
                      border: `1px solid ${isTop ? emotionColor : 'var(--border)'}`,
                      borderRadius: '8px', padding: '0.6rem 0.85rem',
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                        <span style={{ fontWeight: isTop ? 700 : 500, textTransform: 'capitalize', color: isTop ? EMOTION_COLORS[emo] : 'var(--text-main)', fontSize: '0.875rem' }}>
                          {EMOTION_EMOJI[emo]} {emo}
                        </span>
                        <span style={{ fontWeight: 700, color: isTop ? EMOTION_COLORS[emo] : 'var(--text-muted)', fontSize: '0.875rem' }}>{pct}%</span>
                      </div>
                      <div style={{ height: '6px', backgroundColor: 'var(--border)', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${pct}%`, height: '100%', backgroundColor: EMOTION_COLORS[emo] || '#6366f1', opacity: isTop ? 1 : 0.45, borderRadius: '3px' }} />
                      </div>
                    </div>
                  );
                })}
            </div>
          </>
        )}
      </div>

      {/* Secondary Result: Stress */}
      <div className="card mb-6" style={{ borderLeft: `5px solid ${stressColor}` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
          <HeartPulse size={18} color={stressColor} />
          <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
            Secondary Feature — Estimated Stress Level
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
          <div style={{ fontSize: '2.5rem', fontWeight: 800, textTransform: 'capitalize', color: stressColor }}>
            {stress_level || 'Unknown'}
          </div>
          {stress_probabilities && (
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              {Object.entries(stress_probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([cls, prob]) => (
                  <div key={cls} style={{
                    backgroundColor: (STRESS_COLORS[cls] || '#6366f1') + '18',
                    border: `1px solid ${STRESS_COLORS[cls] || '#6366f1'}`,
                    borderRadius: '8px', padding: '0.5rem 1rem', textAlign: 'center', minWidth: '100px',
                  }}>
                    <div style={{ textTransform: 'capitalize', fontWeight: 600, color: STRESS_COLORS[cls] || '#6366f1', fontSize: '0.875rem' }}>{cls}</div>
                    <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>{(prob * 100).toFixed(1)}%</div>
                  </div>
                ))}
            </div>
          )}
        </div>
      </div>

      {/* Session Details */}
      <div className="card mb-6" style={{ borderTop: '4px solid var(--border)' }}>
        <h3 className="mb-4">Session Details</h3>
        <div style={{ display: 'flex', gap: '0.5rem', flexDirection: 'column' }}>
          <p style={{ fontSize: '0.875rem', fontWeight: 600, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Analyzed Files</p>
          {hasFiles ? (
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              {uploadedFiles.hr     && <li style={{ marginBottom: '0.25rem' }}>📄 {uploadedFiles.hr} <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>(Heart Rate CSV)</span></li>}
              {uploadedFiles.ibi    && <li style={{ marginBottom: '0.25rem' }}>📄 {uploadedFiles.ibi} <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>(IBI CSV)</span></li>}
              {uploadedFiles.ecgExp && <li style={{ marginBottom: '0.25rem' }}>📄 {uploadedFiles.ecgExp} <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>(ECG Experiment)</span></li>}
              {uploadedFiles.ecgSleep && <li>📄 {uploadedFiles.ecgSleep} <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>(ECG Sleep)</span></li>}
            </ul>
          ) : (
            <p style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Demo Mode — no files uploaded</p>
          )}
        </div>
      </div>

      {/* Extracted Features */}
      {features && Object.keys(features).length > 0 && (
        <div className="card mb-6">
          <h3 className="mb-2" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={20} color="var(--primary)" /> Extracted Physiological Features
          </h3>
          <p className="text-muted mb-4" style={{ fontSize: '0.875rem' }}>
            Key statistical features extracted from the uploaded signals and used as input to the classification models.
          </p>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
            gap: '0.6rem',
            backgroundColor: 'var(--surface)',
            padding: '1.25rem',
            borderRadius: '8px',
            border: '1px solid var(--border)',
          }}>
            {Object.entries(features).map(([key, value]) => {
              const displayVal = typeof value === 'number'
                ? (Number.isInteger(value) ? value : Number(value).toFixed(4))
                : String(value);
              return (
                <div key={key} style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem', borderBottom: '1px dashed var(--border)', paddingBottom: '0.3rem' }}>
                  <span className="text-muted" style={{ fontSize: '0.82rem' }}>{key}</span>
                  <span style={{ fontWeight: 600, fontSize: '0.82rem' }}>{displayVal}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Academic Disclaimer */}
      <div style={{ padding: '1rem 1.25rem', backgroundColor: '#fef9c3', border: '1px solid #fde68a', borderRadius: '8px', fontSize: '0.82rem', color: '#78350f' }}>
        <strong>Academic Disclaimer:</strong> This report was generated by an experimental research prototype for Empathic Computing purposes only. Results should not be used for medical or clinical diagnosis.
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        @media print {
          .sidebar, .no-print { display: none !important; }
          .main-content { margin-left: 0 !important; padding: 1rem !important; }
          .card { border: 1px solid #ddd !important; box-shadow: none !important; break-inside: avoid; margin-bottom: 1rem; }
          body { background-color: white !important; }
        }
      `}} />
    </div>
  );
}
