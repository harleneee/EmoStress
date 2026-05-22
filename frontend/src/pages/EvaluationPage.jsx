import React, { useState, useEffect } from 'react';
import { Activity, HeartPulse, CheckCircle, ChevronDown, ChevronUp } from 'lucide-react';

const BACKEND = 'http://localhost:8000';

export default function EvaluationPage() {
  const [showEcgMatrix, setShowEcgMatrix] = useState(true);
  const [showHrIbiMatrix, setShowHrIbiMatrix] = useState(true);

  const [ecgReport, setEcgReport] = useState("");
  const [hrIbiReport, setHrIbiReport] = useState("");

  useEffect(() => {
    fetch(`${BACKEND}/evaluation/ECG_emotion_classification_report.txt`)
      .then(res => res.text())
      .then(text => setEcgReport(text))
      .catch(err => console.error("Could not fetch ECG report", err));

    fetch(`${BACKEND}/evaluation/HR_IBI_final_best_classification_report.txt`)
      .then(res => res.text())
      .then(text => setHrIbiReport(text))
      .catch(err => console.error("Could not fetch HR+IBI report", err));
  }, []);

  const emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad'];

  return (
    <div>
      <h1 className="page-title">Model Evaluation</h1>
      <p className="page-subtitle">Performance metrics and confusion matrices from the final trained emotion recognition models.</p>

      {/* Model Summary Cards */}
      <div className="grid-2 mb-8">
        {/* ECG + GSR Emotion Classifier */}
        <div className="card" style={{ borderTop: '4px solid var(--primary)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Activity size={20} color="var(--primary)" /> ECG + GSR Emotion Classifier
            </h3>
            <span style={{ color: 'var(--success)', display:'flex', alignItems:'center', gap:'0.25rem', fontSize:'0.85rem', fontWeight:600 }}>
              <CheckCircle size={14}/> Available
            </span>
          </div>
          <p className="text-muted mt-2" style={{ fontSize: '0.875rem' }}>
            Dataset: ECG + GSR Emotion Dataset
          </p>
          <div style={{ display: 'flex', gap: '2rem', marginTop: '1rem' }}>
            <div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--primary)' }}>83.33%</div>
              <div className="text-muted" style={{ fontSize: '0.8rem' }}>Test Accuracy</div>
            </div>
            <div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--secondary)' }}>30</div>
              <div className="text-muted" style={{ fontSize: '0.8rem' }}>Features</div>
            </div>
          </div>
          <div style={{ marginTop: '1rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {emotions.map(c => (
              <span key={c} className="badge badge-low" style={{ fontSize: '0.75rem', textTransform: 'capitalize' }}>{c}</span>
            ))}
          </div>
        </div>

        {/* ECSMP HR + IBI Emotion Classifier */}
        <div className="card" style={{ borderTop: '4px solid var(--success)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <HeartPulse size={20} color="var(--success)" /> ECSMP HR + IBI Emotion Classifier
            </h3>
            <span style={{ color: 'var(--success)', display:'flex', alignItems:'center', gap:'0.25rem', fontSize:'0.85rem', fontWeight:600 }}>
              <CheckCircle size={14}/> Available
            </span>
          </div>
          <p className="text-muted mt-2" style={{ fontSize: '0.875rem' }}>
            Dataset: ECSMP HR + IBI Dataset
          </p>
          <div style={{ display: 'flex', gap: '2rem', marginTop: '1rem' }}>
            <div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--success)' }}>82.15%</div>
              <div className="text-muted" style={{ fontSize: '0.8rem' }}>Test Accuracy</div>
            </div>
            <div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--warning)' }}>243</div>
              <div className="text-muted" style={{ fontSize: '0.8rem' }}>Features</div>
            </div>
          </div>
          <div style={{ marginTop: '1rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {emotions.map(c => (
              <span key={c} className="badge badge-low" style={{ fontSize: '0.75rem', textTransform: 'capitalize' }}>{c}</span>
            ))}
          </div>
        </div>
      </div>

      {/* ECG Classification Details */}
      <div className="card mb-8">
        <button
          onClick={() => setShowEcgMatrix(!showEcgMatrix)}
          style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
        >
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={20} color="var(--primary)" /> ECG + GSR Emotion Model — Performance
          </h3>
          {showEcgMatrix ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>

        {showEcgMatrix && (
          <div style={{ marginTop: '1.5rem', overflowX: 'auto' }}>
            <div className="grid-2" style={{ gap: '1.5rem', marginBottom: '1.5rem' }}>
              <div>
                <h4 className="mb-3 text-muted" style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Confusion Matrix</h4>
                <img
                  src={`${BACKEND}/evaluation/ECG_emotion_confusion_matrix_normalized.png`}
                  alt="ECG Emotion Confusion Matrix"
                  style={{ width: '100%', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}
                  onError={e => { e.target.style.display='none'; e.target.nextSibling.style.display='block'; }}
                />
                <div style={{ display:'none', padding:'1rem', textAlign:'center', color:'var(--text-muted)', border:'1px dashed var(--border)', borderRadius:'8px' }}>
                  Image not found in backend/evaluation
                </div>
              </div>
              <div>
                <h4 className="mb-3 text-muted" style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Feature Importance</h4>
                <img
                  src={`${BACKEND}/evaluation/ECG_emotion_feature_importance.png`}
                  alt="ECG Emotion Feature Importance"
                  style={{ width: '100%', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}
                  onError={e => { e.target.style.display='none'; e.target.nextSibling.style.display='block'; }}
                />
                <div style={{ display:'none', padding:'1rem', textAlign:'center', color:'var(--text-muted)', border:'1px dashed var(--border)', borderRadius:'8px' }}>
                  Image not found in backend/evaluation
                </div>
              </div>
            </div>
            
            <h4 className="mb-3 text-muted" style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Classification Report</h4>
            {ecgReport ? (
              <pre style={{ backgroundColor: 'var(--surface)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.85rem', overflowX: 'auto' }}>
                {ecgReport}
              </pre>
            ) : (
              <p className="text-muted" style={{ fontSize: '0.85rem' }}>Loading report...</p>
            )}
          </div>
        )}
      </div>

      {/* HR+IBI Classification Details */}
      <div className="card mb-8">
        <button
          onClick={() => setShowHrIbiMatrix(!showHrIbiMatrix)}
          style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
        >
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <HeartPulse size={20} color="var(--success)" /> ECSMP HR + IBI Emotion Model — Performance
          </h3>
          {showHrIbiMatrix ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>

        {showHrIbiMatrix && (
          <div style={{ marginTop: '1.5rem', overflowX: 'auto' }}>
            <div style={{ marginBottom: '1.5rem' }}>
              <h4 className="mb-3 text-muted" style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Confusion Matrix</h4>
              <img
                src={`${BACKEND}/evaluation/HR_IBI_final_best_confusion_matrix_normalized.png`}
                alt="HR+IBI Emotion Confusion Matrix"
                style={{ maxWidth: '600px', width: '100%', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', display: 'block', margin: '0 auto' }}
                onError={e => { e.target.style.display='none'; e.target.nextSibling.style.display='block'; }}
              />
              <div style={{ display:'none', padding:'1rem', textAlign:'center', color:'var(--text-muted)', border:'1px dashed var(--border)', borderRadius:'8px' }}>
                Image not found in backend/evaluation
              </div>
            </div>

            <h4 className="mb-3 text-muted" style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Classification Report</h4>
            {hrIbiReport ? (
              <pre style={{ backgroundColor: 'var(--surface)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.85rem', overflowX: 'auto' }}>
                {hrIbiReport}
              </pre>
            ) : (
              <p className="text-muted" style={{ fontSize: '0.85rem' }}>Loading report...</p>
            )}
            
            <p className="text-muted mt-4" style={{ fontSize: '0.85rem' }}>
              <strong>Note:</strong> This pipeline model uses <strong>243 baseline-calibrated features</strong> extracted exclusively from Heart Rate and Inter-Beat Interval (IBI) signals.
            </p>
          </div>
        )}
      </div>

    </div>
  );
}
