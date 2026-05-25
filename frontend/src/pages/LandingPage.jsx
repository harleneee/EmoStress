import React from 'react';
import { Link } from 'react-router-dom';
import { Brain, HeartPulse, Activity, ShieldCheck, ArrowRight } from 'lucide-react';

export default function LandingPage() {
  return (
    <div style={{ paddingBottom: '4rem' }}>
      {/* Hero Section */}
      <div className="card hero-card" style={{ padding: '5rem 2rem', textAlign: 'center', marginBottom: '4rem' }}>
        <div className="hero-glow"></div>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem', position: 'relative', zIndex: 1 }}>
          <div style={{ background: 'linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%)', padding: '1.25rem', borderRadius: '50%', color: 'white', boxShadow: 'var(--shadow-glow)' }}>
            <Brain size={56} />
          </div>
        </div>
        <h1 style={{ fontSize: '3.5rem', fontWeight: 800, color: 'var(--text-main)', marginBottom: '1.25rem', letterSpacing: '-0.03em', position: 'relative', zIndex: 1 }}>
          Decoding Stress Through Physiology
        </h1>
        <p style={{ fontSize: '1.25rem', color: 'var(--text-muted)', maxWidth: '800px', margin: '0 auto 3rem', lineHeight: '1.7', position: 'relative', zIndex: 1 }}>
          EmoStress is an advanced academic research prototype that utilizes machine learning to detect emotional states and estimate stress levels in college students using multimodal physiological signals.
        </p>
        <div style={{ display: 'flex', gap: '1.25rem', justifyContent: 'center', position: 'relative', zIndex: 1 }}>
          <Link to="/upload" className="btn btn-primary" style={{ padding: '1.1rem 2.5rem', fontSize: '1.1rem' }}>
            Start Analysis <ArrowRight size={20} />
          </Link>
          <Link to="/dataset" className="btn btn-secondary" style={{ padding: '1.1rem 2.5rem', fontSize: '1.1rem' }}>
            Learn About the Dataset
          </Link>
        </div>
      </div>

      {/* Features Section */}
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <h2 style={{ fontSize: '2.25rem', fontWeight: 800, marginBottom: '0.75rem', letterSpacing: '-0.02em' }}>Multimodal Signal Processing</h2>
        <p className="text-muted" style={{ maxWidth: '600px', margin: '0 auto', fontSize: '1.1rem' }}>
          Our Random Forest models analyze high-fidelity data streams to ensure accurate emotion classification.
        </p>
      </div>

      <div className="grid-3 mb-8">
        <div className="card" style={{ textAlign: 'center' }}>
          <div style={{ color: 'var(--primary)', marginBottom: '1.25rem', display: 'flex', justifyContent: 'center' }}>
            <Activity size={48} />
          </div>
          <h3 style={{ marginBottom: '1rem', fontSize: '1.25rem' }}>Electrocardiogram (ECG)</h3>
          <p className="text-muted" style={{ fontSize: '0.95rem', lineHeight: '1.6' }}>
            Extracts complex cardiac features from both emotion-induction experiments and baseline sleep tracking to map the autonomic nervous system.
          </p>
        </div>

        <div className="card" style={{ textAlign: 'center' }}>
          <div style={{ color: 'var(--secondary)', marginBottom: '1.25rem', display: 'flex', justifyContent: 'center' }}>
            <HeartPulse size={48} />
          </div>
          <h3 style={{ marginBottom: '1rem', fontSize: '1.25rem' }}>Heart Rate (HR)</h3>
          <p className="text-muted" style={{ fontSize: '0.95rem', lineHeight: '1.6' }}>
            Utilizes PPG-derived continuous heart rate data from the Empatica E4 wearable to monitor real-time cardiovascular arousal.
          </p>
        </div>

        <div className="card" style={{ textAlign: 'center' }}>
          <div style={{ color: 'var(--primary)', marginBottom: '1.25rem', display: 'flex', justifyContent: 'center' }}>
            <Activity size={48} />
          </div>
          <h3 style={{ marginBottom: '1rem', fontSize: '1.25rem' }}>Inter-Beat Interval (IBI)</h3>
          <p className="text-muted" style={{ fontSize: '0.95rem', lineHeight: '1.6' }}>
            Measures the precise time between consecutive heartbeats to calculate Heart Rate Variability (HRV), a critical biomarker for psychological stress.
          </p>
        </div>
      </div>

      {/* How it Works / Disclaimer */}
      <div className="card" style={{ backgroundColor: 'var(--surface)', display: 'flex', gap: '2rem', alignItems: 'center' }}>
        <div style={{ flex: 1 }}>
          <h2 style={{ fontSize: '1.75rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <ShieldCheck color="var(--success)" size={32} /> Safe & Secure
          </h2>
          <p style={{ lineHeight: '1.7', color: 'var(--text-muted)' }}>
            This web application processes all physiological features securely via a localized FastAPI backend. It is designed specifically for academic research within the domain of <strong>Empathic Computing</strong>. By analyzing physiological markers, the system predicts one of six basic emotions, which is then heuristically mapped to a final estimated stress level.
          </p>
          <p style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#b45309', fontWeight: 600 }}>
            *Not intended for clinical or medical diagnosis.
          </p>
        </div>
        <div style={{ flex: 1, display: 'flex', justifyContent: 'center' }}>
          <img 
            src="https://illustrations.popsy.co/amber/student-going-to-school.svg" 
            alt="Student Illustration" 
            style={{ maxWidth: '100%', height: '250px', opacity: 0.9 }}
          />
        </div>
      </div>
    </div>
  );
}
