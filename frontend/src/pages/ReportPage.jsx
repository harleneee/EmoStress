import React, { useContext } from 'react';
import { AppContext } from '../App';
import { FileText, Printer, Download, Clock, AlertCircle } from 'lucide-react';

export default function ReportPage() {
  const { analysisData, uploadedFiles } = useContext(AppContext);

  const handlePrint = () => {
    window.print();
  };

  if (!analysisData) {
    return (
      <div>
        <h1 className="page-title">Session Report</h1>
        <div className="card text-center py-8">
          <AlertCircle size={48} color="var(--text-muted)" style={{ margin: '0 auto 1rem' }} />
          <h3>No Analysis Data Available</h3>
          <p className="text-muted mt-2">Please upload physiological signals on the Input & Analyze page to generate a report.</p>
        </div>
      </div>
    );
  }

  const { emotion, stress, features } = analysisData;
  const dateStr = new Date().toLocaleString();

  return (
    <div className="report-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div>
          <h1 className="page-title mb-1" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <FileText size={28} color="var(--primary)" /> Analysis Report
          </h1>
          <p className="text-muted" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Clock size={16} /> Generated on {dateStr}
          </p>
        </div>
        
        <div style={{ display: 'flex', gap: '1rem' }} className="no-print">
          <button className="btn btn-primary" onClick={handlePrint} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Printer size={18} /> Print / Save PDF
          </button>
        </div>
      </div>

      <div className="card mb-6" style={{ borderTop: '4px solid var(--primary)' }}>
        <h3 className="mb-4">Session Details</h3>
        <div className="grid-2">
          <div>
            <p className="text-muted mb-1" style={{ fontSize: '0.875rem', fontWeight: 600, textTransform: 'uppercase' }}>Analyzed Files</p>
            <ul style={{ listStyleType: 'none', padding: 0 }}>
              {uploadedFiles?.hr && <li>• {uploadedFiles.hr}</li>}
              {uploadedFiles?.ibi && <li>• {uploadedFiles.ibi}</li>}
              {uploadedFiles?.ecgExp && <li>• {uploadedFiles.ecgExp}</li>}
              {uploadedFiles?.ecgSleep && <li>• {uploadedFiles.ecgSleep}</li>}
              {(!uploadedFiles || Object.keys(uploadedFiles).length === 0) && <li>• No files tracked (Demo Mode)</li>}
            </ul>
          </div>
          <div>
            <p className="text-muted mb-1" style={{ fontSize: '0.875rem', fontWeight: 600, textTransform: 'uppercase' }}>Primary Findings</p>
            <p style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
              Detected Emotion: <strong style={{ textTransform: 'capitalize', color: 'var(--primary)' }}>{emotion?.prediction || 'Unknown'}</strong>
            </p>
            <p style={{ fontSize: '1.1rem' }}>
              Estimated Stress: <strong style={{ textTransform: 'capitalize', color: stress?.prediction === 'high' ? 'var(--danger)' : stress?.prediction === 'moderate' ? 'var(--warning)' : 'var(--success)' }}>
                {stress?.prediction || 'Unknown'}
              </strong>
            </p>
          </div>
        </div>
      </div>

      <div className="card mb-6">
        <h3 className="mb-4">Emotion Probabilities</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '1rem' }}>
          {emotion?.probabilities && Object.entries(emotion.probabilities).map(([emo, prob]) => (
            <div key={emo} style={{ backgroundColor: 'var(--surface)', padding: '1rem', borderRadius: '8px', textAlign: 'center', border: '1px solid var(--border)' }}>
              <div style={{ textTransform: 'capitalize', fontWeight: 600, marginBottom: '0.5rem' }}>{emo}</div>
              <div style={{ fontSize: '1.25rem', color: 'var(--primary)' }}>{(prob * 100).toFixed(1)}%</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3 className="mb-4">Extracted Physiological Features</h3>
        <p className="text-muted mb-4" style={{ fontSize: '0.9rem' }}>
          The following raw statistical features were extracted from the signals and fed into the Random Forest classification models.
        </p>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', 
          gap: '0.75rem',
          backgroundColor: 'var(--surface)',
          padding: '1.5rem',
          borderRadius: '8px',
          border: '1px solid var(--border)'
        }}>
          {features && Object.entries(features).map(([key, value]) => {
            // Format value: if it's a number, fix to 4 decimals max
            const displayVal = typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(4)) : value;
            return (
              <div key={key} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px dashed var(--border)', paddingBottom: '0.25rem' }}>
                <span className="text-muted" style={{ fontSize: '0.85rem' }}>{key}</span>
                <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>{displayVal}</span>
              </div>
            );
          })}
        </div>
      </div>
      
      <style dangerouslySetInnerHTML={{__html: `
        @media print {
          .sidebar, .topbar, .no-print { display: none !important; }
          .main-content { margin-left: 0 !important; padding: 0 !important; }
          .card { border: 1px solid #ddd !important; box-shadow: none !important; break-inside: avoid; }
          body { background-color: white !important; }
        }
      `}} />
    </div>
  );
}
