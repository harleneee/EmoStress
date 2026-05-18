import React, { useContext } from 'react';
import { AppContext } from '../App';
import { Link } from 'react-router-dom';
import { Activity, Brain, Heart, ArrowRight } from 'lucide-react';

export default function Dashboard() {
  const { analysisData } = useContext(AppContext);

  if (!analysisData) {
    return (
      <div className="text-center mt-8">
        <h1 className="page-title">Analysis Dashboard</h1>
        <p className="page-subtitle mb-8">No data available. Please upload signals first.</p>
        <Link to="/upload" className="btn btn-primary">Go to Upload Page</Link>
      </div>
    );
  }

  const { emotion, stress_level, confidence, features } = analysisData;
  
  // Format numbers
  const formatValue = (val) => val ? Number(val).toFixed(2) : "0.00";

  return (
    <div>
      <h1 className="page-title">Analysis Dashboard</h1>
      <p className="page-subtitle">Overview of predicted emotion, stress level, and extracted physiological features.</p>

      {/* Main Results */}
      <div className="grid-2 mb-8">
        <div className="card" style={{ textAlign: 'center', borderTop: '4px solid var(--primary)' }}>
          <h3 className="mb-4 text-muted">Detected Emotion</h3>
          <div style={{ fontSize: '3rem', fontWeight: 'bold', textTransform: 'capitalize', color: 'var(--primary)' }}>
            {emotion}
          </div>
          <Link to="/emotions" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', color: 'var(--primary)', marginTop: '1rem', fontWeight: 600 }}>
            View Probabilities <ArrowRight size={16} />
          </Link>
        </div>

        <div className="card" style={{ textAlign: 'center', borderTop: '4px solid var(--danger)' }}>
          <h3 className="mb-4 text-muted">Estimated Stress Level</h3>
          <div style={{ fontSize: '3rem', fontWeight: 'bold', textTransform: 'capitalize', color: stress_level === 'high' ? 'var(--danger)' : stress_level === 'moderate' ? 'var(--warning)' : 'var(--success)' }}>
            {stress_level}
          </div>
          <Link to="/logic" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', color: 'var(--text-muted)', marginTop: '1rem', fontWeight: 600 }}>
            How is this calculated? <ArrowRight size={16} />
          </Link>
        </div>
      </div>

      <h2 className="mb-4" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Activity size={24} color="var(--primary)" /> Extracted Features Summary
      </h2>
      
      <div className="grid-4 mb-8">
        <div className="card">
          <h4 className="text-muted mb-2">HR Mean</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 600 }}>{formatValue(features.hr_mean)} bpm</div>
        </div>
        <div className="card">
          <h4 className="text-muted mb-2">HR Std Dev</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 600 }}>{formatValue(features.hr_std)}</div>
        </div>
        <div className="card">
          <h4 className="text-muted mb-2">IBI Mean</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 600 }}>{formatValue(features.ibi_mean)} s</div>
        </div>
        <div className="card">
          <h4 className="text-muted mb-2">IBI RMSSD</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 600 }}>{formatValue(features.ibi_rmssd)} s</div>
        </div>
      </div>
      
      <div className="card" style={{ backgroundColor: 'var(--primary-light)', borderColor: 'var(--primary)', color: 'var(--primary)' }}>
        <p><strong>Note:</strong> Over 48 statistical features were extracted from the ECG, HR, and IBI signals to generate these predictions using the Random Forest models. <Link to="/report" style={{ textDecoration: 'underline', fontWeight: 'bold' }}>See the Report page</Link> for full details.</p>
      </div>

    </div>
  );
}
