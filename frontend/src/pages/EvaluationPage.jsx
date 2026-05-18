import React, { useState, useEffect } from 'react';
import { Image as ImageIcon, AlertTriangle } from 'lucide-react';

export default function EvaluationPage() {
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/model-info')
      .then(res => res.json())
      .then(data => setModelInfo(data))
      .catch(err => console.error("Could not fetch model info", err));
  }, []);

  return (
    <div>
      <h1 className="page-title">Model Evaluation</h1>
      <p className="page-subtitle">Confusion matrices and classification reports from the original Random Forest training.</p>

      {modelInfo && (!modelInfo.emotion_model_loaded || !modelInfo.stress_model_loaded) && (
        <div className="card mb-8" style={{ backgroundColor: 'var(--warning-light)', color: '#b45309', borderColor: '#fde68a' }}>
          <strong style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertTriangle size={20} /> Warning: Missing Models
          </strong>
          <p className="mt-2">
            The FastAPI backend could not locate the <code>.joblib</code> models in the <code>backend/models</code> directory. 
            The system is currently running in a mock fallback mode. Please place the `.joblib` files in the backend.
          </p>
        </div>
      )}

      <div className="grid-2 mb-8">
        <div className="card">
          <h3 className="mb-4" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <ImageIcon size={20} color="var(--primary)" /> Emotion Confusion Matrix
          </h3>
          <img 
            src="http://localhost:8000/evaluation/emotion_confusion_matrix_normalized.png" 
            alt="Emotion Confusion Matrix" 
            style={{ width: '100%', borderRadius: 'var(--radius-md)' }}
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.nextSibling.style.display = 'block';
            }}
          />
          <div style={{ display: 'none', padding: '2rem', textAlign: 'center', backgroundColor: 'var(--background)', color: 'var(--text-muted)' }}>
            Image not found in backend/evaluation
          </div>
        </div>

        <div className="card">
          <h3 className="mb-4" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <ImageIcon size={20} color="var(--danger)" /> Stress Confusion Matrix
          </h3>
          <img 
            src="http://localhost:8000/evaluation/stress_confusion_matrix_normalized.png" 
            alt="Stress Confusion Matrix" 
            style={{ width: '100%', borderRadius: 'var(--radius-md)' }}
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.nextSibling.style.display = 'block';
            }}
          />
          <div style={{ display: 'none', padding: '2rem', textAlign: 'center', backgroundColor: 'var(--background)', color: 'var(--text-muted)' }}>
            Image not found in backend/evaluation
          </div>
        </div>
      </div>

      <div className="grid-2 mb-8">
        <div className="card">
          <h3 className="mb-4">Emotion Classification Report</h3>
          <a href="http://localhost:8000/evaluation/emotion_classification_report.txt" target="_blank" rel="noreferrer" className="btn btn-secondary">
            View Raw Report Text
          </a>
        </div>
        
        <div className="card">
          <h3 className="mb-4">Stress Classification Report</h3>
          <a href="http://localhost:8000/evaluation/stress_classification_report.txt" target="_blank" rel="noreferrer" className="btn btn-secondary">
            View Raw Report Text
          </a>
        </div>
      </div>

    </div>
  );
}
