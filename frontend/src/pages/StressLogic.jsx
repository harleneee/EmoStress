import React from 'react';
import { HeartPulse, ShieldAlert, Smile, Zap, Activity } from 'lucide-react';

export default function StressLogic() {
  return (
    <div>
      <h1 className="page-title">Classification Logic</h1>
      <p className="page-subtitle">
        EmoStress operates using two distinct emotion recognition models depending on the available physiological data. Stress is derived dynamically from these emotional states.
      </p>

      {/* PRIMARY: HR + IBI Emotion Classifier */}
      <div className="card mb-8" style={{ borderLeft: '6px solid var(--primary)', background: 'linear-gradient(135deg, var(--primary-light) 0%, var(--surface) 100%)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <div style={{ background: 'var(--primary)', padding: '0.5rem', borderRadius: '8px', color: 'white', display: 'flex' }}>
            <HeartPulse size={22} />
          </div>
          <div>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--primary)' }}>Primary Model</span>
            <h2 style={{ margin: 0, color: 'var(--primary)' }}>Emotion Recognition (HR + IBI)</h2>
          </div>
        </div>
        <p style={{ lineHeight: '1.7', marginBottom: '1.5rem' }}>
          The primary model for the EmoStress web application is a robust <strong>Extra Trees Pipeline</strong> trained on the <strong>ECSMP HR + IBI Dataset</strong>. It extracts <strong>243 baseline-calibrated features</strong> from standard Heart Rate (HR) and Inter-Beat Interval (IBI) signals to predict six basic emotions, achieving an impressive <strong>82.15% test accuracy</strong> without the need for clinical ECG equipment.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: '0.75rem' }}>
          {[
            { label: 'Happy',   color: '#10b981', emoji: '😊', note: 'Positive, calm arousal' },
            { label: 'Neutral', color: '#6366f1', emoji: '😐', note: 'Baseline state' },
            { label: 'Sad',     color: '#64748b', emoji: '😢', note: 'Low energy, negative' },
            { label: 'Fear',    color: '#f59e0b', emoji: '😨', note: 'High arousal, threat' },
            { label: 'Anger',   color: '#ef4444', emoji: '😠', note: 'High arousal, hostile' },
            { label: 'Disgust', color: '#8b5cf6', emoji: '🤢', note: 'Aversion response' },
          ].map(e => (
            <div key={e.label} style={{ backgroundColor: 'white', border: `2px solid ${e.color}30`, borderRadius: '10px', padding: '0.85rem', textAlign: 'center' }}>
              <div style={{ fontSize: '1.75rem', marginBottom: '0.25rem' }}>{e.emoji}</div>
              <div style={{ fontWeight: 700, color: e.color, marginBottom: '0.2rem' }}>{e.label}</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{e.note}</div>
            </div>
          ))}
        </div>

        <div style={{ marginTop: '1.25rem', padding: '0.85rem 1.25rem', backgroundColor: 'white', borderRadius: '8px', border: '1px solid var(--border)' }}>
          <p style={{ margin: 0, fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
            <strong style={{ color: 'var(--primary)' }}>Key HR/IBI Features (243):</strong> Windowed statistics (Mean, Median, Std, Quantiles), peak interval distributions, and Heart Rate Variability (HRV) metrics such as RMSSD, pNN50, and SDNN across time.
          </p>
        </div>
      </div>

      {/* SECONDARY: ECG + GSR Emotion Classifier */}
      <div className="card mb-8" style={{ borderLeft: '6px solid var(--success)', background: 'linear-gradient(135deg, #f0fdf4 0%, var(--surface) 100%)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <div style={{ background: 'var(--success)', padding: '0.5rem', borderRadius: '8px', color: 'white', display: 'flex' }}>
            <Activity size={22} />
          </div>
          <div>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--success)' }}>Secondary Model</span>
            <h2 style={{ margin: 0, color: 'var(--success)' }}>Emotion Recognition (ECG + GSR)</h2>
          </div>
        </div>
        <p style={{ lineHeight: '1.7', marginBottom: '1.5rem' }}>
          When raw electrocardiogram (ECG) and galvanic skin response (GSR) binary files are available, EmoStress utilizes a secondary <strong>Random Forest Classifier</strong> trained on the <strong>ECG and GSR Emotion Dataset</strong>. This model extracts <strong>30 features</strong> to classify the same emotional states with a slightly higher <strong>83.33% test accuracy</strong>.
        </p>
      </div>

      {/* STRESS DERIVATION */}
      <div className="card" style={{ borderLeft: '4px solid var(--text-muted)', opacity: 0.95 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <div style={{ background: 'var(--surface)', padding: '0.5rem', borderRadius: '8px', color: 'var(--text-muted)', display: 'flex', border: '1px solid var(--border)' }}>
            <Zap size={22} />
          </div>
          <div>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>Interpretation Layer</span>
            <h2 style={{ margin: 0, color: 'var(--text-main)' }}>Stress Level Derivation</h2>
          </div>
        </div>
        <p style={{ lineHeight: '1.7', marginBottom: '1.5rem' }}>
          EmoStress no longer uses an isolated stress classification model. Instead, stress levels are interpreted dynamically as a <strong>secondary mapping</strong> derived straight from the predicted emotional state.
        </p>

        <div className="grid-2" style={{ gap: '1rem' }}>
          <div className="card" style={{ borderTop: '3px solid var(--success)', backgroundColor: 'white' }}>
            <h3 style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Smile size={18} /> Low Stress
            </h3>
            <p className="text-muted mt-2" style={{ fontSize: '0.85rem', lineHeight: '1.5' }}>
              Maps to physiological states of rest or positive/neutral arousal.
            </p>
            <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span className="badge badge-low">Happy</span>
              <span className="badge badge-low">Neutral</span>
            </div>
          </div>
          <div className="card" style={{ borderTop: '3px solid var(--danger)', backgroundColor: 'white' }}>
            <h3 style={{ color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <ShieldAlert size={18} /> Moderate to High Stress
            </h3>
            <p className="text-muted mt-2" style={{ fontSize: '0.85rem', lineHeight: '1.5' }}>
              Maps to emotional states exhibiting sympathetic nervous system activation (fight-or-flight) or aversive responses.
            </p>
            <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span className="badge" style={{ backgroundColor: '#fef08a', color: '#854d0e', border: '1px solid #fde047' }}>Sad</span>
              <span className="badge badge-high">Disgust</span>
              <span className="badge badge-high">Anger</span>
              <span className="badge badge-high">Fear</span>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
