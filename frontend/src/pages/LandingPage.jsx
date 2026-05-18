import React from 'react';
import { Link } from 'react-router-dom';
import { Brain, HeartPulse, Activity, ShieldCheck, ArrowRight } from 'lucide-react';

export default function LandingPage() {
  return (
    <div style={{ paddingBottom: '4rem' }}>
      {/* Hero Section */}
      <div 
        className="card" 
        style={{ 
          background: 'linear-gradient(135deg, var(--primary-light) 0%, var(--surface) 100%)',
          border: '1px solid var(--primary-light)',
          padding: '4rem 2rem',
          textAlign: 'center',
          marginBottom: '3rem'
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }}>
          <div style={{ background: 'var(--primary)', padding: '1rem', borderRadius: '50%', color: 'white', boxShadow: 'var(--shadow-md)' }}>
            <Brain size={48} />
          </div>
        </div>
        <h1 style={{ fontSize: '3rem', fontWeight: 800, color: 'var(--text-main)', marginBottom: '1rem', letterSpacing: '-0.025em' }}>
          Decoding Stress Through Physiology
        </h1>
        <p style={{ fontSize: '1.25rem', color: 'var(--text-muted)', maxWidth: '800px', margin: '0 auto 2.5rem', lineHeight: '1.6' }}>
          EmoStress is an advanced academic research prototype that utilizes machine learning to detect emotional states and estimate stress levels in college students using multimodal physiological signals.
        </p>
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
          <Link to="/upload" className="btn btn-primary" style={{ padding: '1rem 2rem', fontSize: '1.1rem' }}>
            Start Analysis <ArrowRight size={20} />
          </Link>
          <Link to="/dataset" className="btn btn-secondary" style={{ padding: '1rem 2rem', fontSize: '1.1rem', backgroundColor: 'white' }}>
            Learn About the Dataset
          </Link>
        </div>
      </div>

      {/* Features Section */}
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <h2 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '0.5rem' }}>Multimodal Signal Processing</h2>
        <p className="text-muted" style={{ maxWidth: '600px', margin: '0 auto' }}>
          Our Random Forest models analyze high-fidelity data streams to ensure accurate emotion classification.
        </p>
      </div>

      <div className="grid-3 mb-8">
        <div className="card" style={{ textAlign: 'center', transition: 'transform 0.2s', cursor: 'default' }} onMouseOver={e => e.currentTarget.style.transform = 'translateY(-5px)'} onMouseOut={e => e.currentTarget.style.transform = 'translateY(0)'}>
          <div style={{ color: 'var(--primary)', marginBottom: '1rem', display: 'flex', justifyContent: 'center' }}>
            <Activity size={40} />
          </div>
          <h3 style={{ marginBottom: '1rem' }}>Electrocardiogram (ECG)</h3>
          <p className="text-muted" style={{ fontSize: '0.95rem', lineHeight: '1.6' }}>
            Extracts complex cardiac features from both emotion-induction experiments and baseline sleep tracking to map the autonomic nervous system.
          </p>
        </div>

        <div className="card" style={{ textAlign: 'center', transition: 'transform 0.2s', cursor: 'default' }} onMouseOver={e => e.currentTarget.style.transform = 'translateY(-5px)'} onMouseOut={e => e.currentTarget.style.transform = 'translateY(0)'}>
          <div style={{ color: 'var(--secondary)', marginBottom: '1rem', display: 'flex', justifyContent: 'center' }}>
            <HeartPulse size={40} />
          </div>
          <h3 style={{ marginBottom: '1rem' }}>Heart Rate (HR)</h3>
          <p className="text-muted" style={{ fontSize: '0.95rem', lineHeight: '1.6' }}>
            Utilizes PPG-derived continuous heart rate data from the Empatica E4 wearable to monitor real-time cardiovascular arousal.
          </p>
        </div>

        <div className="card" style={{ textAlign: 'center', transition: 'transform 0.2s', cursor: 'default' }} onMouseOver={e => e.currentTarget.style.transform = 'translateY(-5px)'} onMouseOut={e => e.currentTarget.style.transform = 'translateY(0)'}>
          <div style={{ color: 'var(--primary)', marginBottom: '1rem', display: 'flex', justifyContent: 'center' }}>
            <Activity size={40} />
          </div>
          <h3 style={{ marginBottom: '1rem' }}>Inter-Beat Interval (IBI)</h3>
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
