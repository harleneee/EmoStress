import React from 'react';
import { Brain, ArrowRight, HeartPulse, ShieldAlert, Smile, Frown } from 'lucide-react';

export default function StressLogic() {
  return (
    <div>
      <h1 className="page-title">Stress Interpretation Logic</h1>
      <p className="page-subtitle">How does the system map detected emotions and physiological signals to a final stress level?</p>

      <div className="card mb-8">
        <h2 className="mb-4" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--primary)' }}>
          <Brain size={24} /> Emotion-to-Stress Mapping
        </h2>
        <p className="mb-6" style={{ lineHeight: '1.7' }}>
          Because emotional states are closely tied to sympathetic nervous system arousal, the primary driver for our stress estimation is the detected emotion. The Random Forest model categorizes physiological features into one of six basic emotions. We then apply a rule-based heuristic to estimate the corresponding stress level:
        </p>

        <div className="grid-3">
          {/* Low Stress */}
          <div className="card" style={{ borderTop: '4px solid var(--success)', backgroundColor: 'var(--surface)' }}>
            <h3 style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Smile size={20} /> Low Stress
            </h3>
            <div style={{ margin: '1rem 0' }}>
              <span className="badge badge-low" style={{ marginRight: '0.5rem' }}>Neutral</span>
              <span className="badge badge-low">Happy</span>
            </div>
            <p className="text-muted" style={{ fontSize: '0.875rem', lineHeight: '1.5' }}>
              These emotional states typically correspond to a relaxed or positive physiological baseline, characterized by stable Heart Rate and high Heart Rate Variability (HRV).
            </p>
          </div>

          {/* Moderate Stress */}
          <div className="card" style={{ borderTop: '4px solid var(--warning)', backgroundColor: 'var(--surface)' }}>
            <h3 style={{ color: 'var(--warning)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Frown size={20} /> Moderate Stress
            </h3>
            <div style={{ margin: '1rem 0' }}>
              <span className="badge badge-moderate">Sad</span>
            </div>
            <p className="text-muted" style={{ fontSize: '0.875rem', lineHeight: '1.5' }}>
              Sadness induces mild physiological arousal and cognitive load, leading to slight deviations in cardiovascular patterns, interpreted as moderate stress.
            </p>
          </div>

          {/* High Stress */}
          <div className="card" style={{ borderTop: '4px solid var(--danger)', backgroundColor: 'var(--surface)' }}>
            <h3 style={{ color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <ShieldAlert size={20} /> High Stress
            </h3>
            <div style={{ margin: '1rem 0' }}>
              <span className="badge badge-high" style={{ marginRight: '0.5rem' }}>Fear</span>
              <span className="badge badge-high" style={{ marginRight: '0.5rem' }}>Anger</span>
              <span className="badge badge-high">Disgust</span>
            </div>
            <p className="text-muted" style={{ fontSize: '0.875rem', lineHeight: '1.5' }}>
              These intense negative emotions trigger a strong "fight-or-flight" response, characterized by elevated Heart Rate, reduced HRV, and rapid ECG peaks.
            </p>
          </div>
        </div>
      </div>

      <div className="card" style={{ backgroundColor: 'var(--primary-light)', borderLeft: '4px solid var(--primary)' }}>
        <h2 className="mb-4" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--primary)' }}>
          <HeartPulse size={24} /> Physiological Modifiers
        </h2>
        <p style={{ lineHeight: '1.7', marginBottom: 0 }}>
          While the baseline stress level is derived directly from the emotion classification, the secondary Random Forest model evaluates the <strong>raw physiological features</strong> (such as IBI RMSSD and ECG peak rates) to fine-tune the confidence probability of that stress state. For example, if the system detects "Fear" but the Heart Rate remains uncharacteristically low (e.g., 60 BPM), the confidence score for "High Stress" will be penalized.
        </p>
      </div>
    </div>
  );
}
