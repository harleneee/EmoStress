import React, { useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Activity, Zap, Info, ChevronDown, ChevronUp } from 'lucide-react';
import { AppContext } from '../App';

export default function UploadPage() {
  const { setAnalysisData, setUploadedFiles } = useContext(AppContext);
  const navigate = useNavigate();
  
  const [files, setFilesState] = useState({
    hr: null,
    ibi: null,
    ecgExp: null,
    ecgSleep: null
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleFileChange = (e, type) => {
    if (e.target.files && e.target.files[0]) {
      setFilesState({ ...files, [type]: e.target.files[0] });
    }
  };

  const handleAnalyze = async (sampleProfile = null) => {
    setLoading(true);
    setError(null);
    
    // Store uploaded files for report generation
    setUploadedFiles({
      hr: sampleProfile ? 'Sample_HR.csv' : files.hr?.name,
      ibi: sampleProfile ? 'Sample_IBI.csv' : files.ibi?.name,
      ecgExp: sampleProfile ? 'Sample_ECG_Exp.bin' : files.ecgExp?.name,
      ecgSleep: sampleProfile ? 'Sample_ECG_Sleep.bin' : files.ecgSleep?.name
    });

    const formData = new FormData();
    if (sampleProfile) {
      formData.append('sample_profile', sampleProfile);
    } else {
      if (files.hr) formData.append('hr_file', files.hr);
      if (files.ibi) formData.append('ibi_file', files.ibi);
      if (files.ecgExp) formData.append('ecg_exp_file', files.ecgExp);
      if (files.ecgSleep) formData.append('ecg_sleep_file', files.ecgSleep);
    }

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      setAnalysisData(data);
      navigate('/dashboard');
    } catch (err) {
      console.error(err);
      setError("Failed to connect to the backend server. Make sure FastAPI is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 className="page-title">Input & Analyze</h1>
      <p className="page-subtitle">Test the stress detection model using your own sensor data or our pre-loaded samples.</p>

      {error && (
        <div className="card mb-8" style={{ backgroundColor: 'var(--danger-light)', color: 'var(--danger)', borderColor: 'var(--danger)' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* DEMO SECTION */}
      <div className="card mb-8" style={{ borderLeft: '4px solid var(--primary)', backgroundColor: 'var(--primary-light)' }}>
        <h3 className="mb-4" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--primary)' }}>
          <Zap size={20} /> Quick Test (Recommended)
        </h3>
        <p className="mb-4 text-main">
          Don't have sensor files right now? Use our pre-recorded dataset profiles to instantly see how the model interprets different physiological states.
        </p>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button 
            className="btn btn-primary" 
            onClick={() => handleAnalyze('relaxed')}
            disabled={loading}
          >
            Load Subject A (Relaxed)
          </button>
          <button 
            className="btn btn-secondary" 
            onClick={() => handleAnalyze('stressed')}
            disabled={loading}
            style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--primary)' }}
          >
            Load Subject B (Stressed)
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', margin: '2rem 0' }}>
        <div style={{ flex: 1, height: '1px', backgroundColor: 'var(--border)' }}></div>
        <span style={{ padding: '0 1rem', color: 'var(--text-muted)', fontWeight: 600 }}>OR UPLOAD YOUR DATA</span>
        <div style={{ flex: 1, height: '1px', backgroundColor: 'var(--border)' }}></div>
      </div>

      {/* STANDARD UPLOAD SECTION */}
      <div className="grid-2 mb-4">
        <div className="card">
          <h3 className="mb-2" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={20} color="var(--primary)" /> Heart Rate (HR.csv)
          </h3>
          <p className="text-muted mb-4" style={{ fontSize: '0.875rem' }}>Exported from Empatica E4 or standard smartwatches.</p>
          <input type="file" accept=".csv" onChange={(e) => handleFileChange(e, 'hr')} className="w-full mb-4" />
          {files.hr && <span className="badge badge-low">Selected: {files.hr.name}</span>}
        </div>

        <div className="card">
          <h3 className="mb-2" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={20} color="var(--primary)" /> Inter-Beat Interval (IBI.csv)
          </h3>
          <p className="text-muted mb-4" style={{ fontSize: '0.875rem' }}>Provides Heart Rate Variability (HRV) insights.</p>
          <input type="file" accept=".csv" onChange={(e) => handleFileChange(e, 'ibi')} className="w-full mb-4" />
          {files.ibi && <span className="badge badge-low">Selected: {files.ibi.name}</span>}
        </div>
      </div>

      {/* ADVANCED UPLOAD SECTION */}
      <div className="card mb-8" style={{ padding: '1rem 1.5rem', cursor: 'pointer' }} onClick={() => setShowAdvanced(!showAdvanced)}>
        <h4 style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Info size={18} /> Advanced ECG Inputs (Optional)</span>
          {showAdvanced ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </h4>
        
        {showAdvanced && (
          <div className="grid-2 mt-4 pt-4" style={{ borderTop: '1px solid var(--border)' }} onClick={(e) => e.stopPropagation()}>
            <div>
              <label className="mb-2 block font-semibold text-sm">ECG Experiment (.bin)</label>
              <input type="file" accept=".bin" onChange={(e) => handleFileChange(e, 'ecgExp')} className="w-full mb-4" />
            </div>
            <div>
              <label className="mb-2 block font-semibold text-sm">ECG Sleep (.bin)</label>
              <input type="file" accept=".bin" onChange={(e) => handleFileChange(e, 'ecgSleep')} className="w-full mb-4" />
            </div>
          </div>
        )}
      </div>

      <div className="text-center">
        <button 
          className="btn btn-primary" 
          onClick={() => handleAnalyze()}
          disabled={loading || (!files.hr && !files.ibi)}
          style={{ fontSize: '1.1rem', padding: '1rem 2rem' }}
        >
          {loading ? (
            <>Processing Data...</>
          ) : (
            <><Upload size={20} /> Analyze Custom Data</>
          )}
        </button>
        {(!files.hr && !files.ibi) && (
          <p className="text-muted mt-2" style={{ fontSize: '0.875rem' }}>Upload at least HR or IBI data to analyze.</p>
        )}
      </div>
    </div>
  );
}
