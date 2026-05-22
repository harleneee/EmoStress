import React from 'react';
import { Database, Activity, Heart, Moon, Zap, BarChart2 } from 'lucide-react';

export default function DatasetOverview() {
  return (
    <div>
      <h1 className="page-title">Dataset Overview</h1>
      <p className="page-subtitle">Learn about the ECSMP dataset used to train the EmoStress machine learning models.</p>

      <div className="card mb-8">
        <h2 className="mb-4" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--primary)' }}>
          <Database size={24} /> The ECSMP Dataset
        </h2>
        <p className="mb-4" style={{ lineHeight: '1.7' }}>
          This project relies on the <strong>ECSMP (Emotion, Cognition, Sleep, and Multi-model Physiological Signals)</strong> dataset.
          The dataset was carefully collected from college students during emotion-induction experiments and sleep tracking.
          By combining these diverse physiological markers, the dataset allows for highly contextual machine learning models.
        </p>
      </div>

      <h2 className="mb-4" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Activity size={24} color="var(--secondary)" /> Selected Physiological Signals
      </h2>
      <div className="grid-2 mb-8">
        <div className="card">
          <h4 className="mb-2" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Heart size={20} color="var(--primary)" /> PPG-derived Heart Rate (HR)
          </h4>
          <p className="text-muted" style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
            Extracted from the Empatica E4 wearable device. Continuous Heart Rate provides a baseline for sympathetic nervous system arousal (the "fight-or-flight" response).
          </p>
        </div>

        <div className="card">
          <h4 className="mb-2" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={20} color="var(--primary)" /> Inter-Beat Interval (IBI)
          </h4>
          <p className="text-muted" style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
            Also extracted from the Empatica E4, the IBI represents the time interval between consecutive heartbeats. This is crucial for calculating Heart Rate Variability (HRV), a key indicator of stress.
          </p>
        </div>

        <div className="card">
          <h4 className="mb-2" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Zap size={20} color="var(--secondary)" /> ECG Experiment Data
          </h4>
          <p className="text-muted" style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
            Electrical activity of the heart recorded via chest sensors during the emotion-induction video experiments. Provides high-fidelity insights into cardiac responses to stimuli.
          </p>
        </div>

        <div className="card">
          <h4 className="mb-2" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Moon size={20} color="var(--secondary)" /> ECG Sleep Data
          </h4>
          <p className="text-muted" style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
            Heart activity recorded during the participant's sleep prior to the experiment. This helps establish a personalized baseline for restfulness and cognitive load.
          </p>
        </div>
      </div>

      <div className="card" style={{ backgroundColor: 'var(--primary-light)', borderLeft: '4px solid var(--primary)' }}>
        <h2 className="mb-4" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--primary)' }}>
          <BarChart2 size={24} /> Six Emotion Classes
        </h2>
        <p className="mb-4" style={{ lineHeight: '1.7' }}>
          During the experiment, subjects were shown six emotion-inducing videos (Video1–Video6). The signal streams were segmented into equal parts and labeled according to the video order. The models were trained to detect these six distinct emotional states:
        </p>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
          <span className="badge badge-low" style={{ fontSize: '1rem', padding: '0.5rem 1rem' }}>Neutral</span>
          <span className="badge badge-high" style={{ fontSize: '1rem', padding: '0.5rem 1rem' }}>Fear</span>
          <span className="badge badge-moderate" style={{ fontSize: '1rem', padding: '0.5rem 1rem' }}>Sad</span>
          <span className="badge badge-low" style={{ fontSize: '1rem', padding: '0.5rem 1rem' }}>Happy</span>
          <span className="badge badge-high" style={{ fontSize: '1rem', padding: '0.5rem 1rem' }}>Anger</span>
          <span className="badge badge-high" style={{ fontSize: '1rem', padding: '0.5rem 1rem' }}>Disgust</span>
        </div>
      </div>

    </div>
  );
}
