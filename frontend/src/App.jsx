import React, { useState, createContext } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { Activity, Brain, Upload, BarChart2, User, FileText, Heart, Moon, Image as ImageIcon } from 'lucide-react';

export const AppContext = createContext();

import LandingPage from './pages/LandingPage';
import DatasetOverview from './pages/DatasetOverview';
import UploadPage from './pages/UploadPage';
import Dashboard from './pages/Dashboard';
import EmotionResults from './pages/EmotionResults';
import StressLogic from './pages/StressLogic';
import ProfilePage from './pages/ProfilePage';
import ReportPage from './pages/ReportPage';
import EvaluationPage from './pages/EvaluationPage';

function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState({});

  return (
    <AppContext.Provider value={{ analysisData, setAnalysisData, uploadedFiles, setUploadedFiles }}>
      <Router>
        <div className="app-container">
          <aside className="sidebar">
            <div className="sidebar-logo">
              <Brain size={32} />
              <span>EmoStress</span>
            </div>
            
            <nav>
              <NavLink to="/" className={({isActive}) => isActive ? "nav-link active" : "nav-link"} end>
                <Activity size={20} /> Home
              </NavLink>
              <NavLink to="/dataset" className={({isActive}) => isActive ? "nav-link active" : "nav-link"}>
                <FileText size={20} /> Dataset Info
              </NavLink>
              <NavLink to="/upload" className={({isActive}) => isActive ? "nav-link active" : "nav-link"}>
                <Upload size={20} /> Input & Analyze
              </NavLink>
              <NavLink to="/dashboard" className={({isActive}) => isActive ? "nav-link active" : "nav-link"}>
                <BarChart2 size={20} /> Dashboard
              </NavLink>
              <NavLink to="/emotions" className={({isActive}) => isActive ? "nav-link active" : "nav-link"}>
                <Heart size={20} /> Emotion Results
              </NavLink>
              <NavLink to="/logic" className={({isActive}) => isActive ? "nav-link active" : "nav-link"}>
                <Brain size={20} /> Stress Logic
              </NavLink>
              <NavLink to="/evaluation" className={({isActive}) => isActive ? "nav-link active" : "nav-link"}>
                <ImageIcon size={20} /> Model Evaluation
              </NavLink>
              <NavLink to="/report" className={({isActive}) => isActive ? "nav-link active" : "nav-link"}>
                <FileText size={20} /> Generate Report
              </NavLink>
            </nav>

            <div className="disclaimer" style={{ marginTop: 'auto', fontSize: '11px', textAlign: 'left', lineHeight: '1.4' }}>
              <strong>Important Disclaimer:</strong><br/>
              This system is developed for Empathic Computing academic research purposes only. The stress and emotion results are based on experimental physiological signal patterns and should not be used as medical or psychological diagnosis.
            </div>
          </aside>

          <main className="main-content">
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/dataset" element={<DatasetOverview />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/emotions" element={<EmotionResults />} />
              <Route path="/logic" element={<StressLogic />} />
              <Route path="/evaluation" element={<EvaluationPage />} />
              <Route path="/profile" element={<ProfilePage />} />
              <Route path="/report" element={<ReportPage />} />
            </Routes>
          </main>
        </div>
      </Router>
    </AppContext.Provider>
  );
}

export default App;
