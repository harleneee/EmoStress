# EmoStress: A Web-Based Stress and Emotion Detection System

A full-stack web application designed to detect stress and emotion in college students using physiological signals (ECG, HR, IBI) from the ECSMP dataset.

## Architecture
- **Backend**: Python FastAPI 
- **Frontend**: React (Vite)
- **Machine Learning**: Random Forest classification models loaded via `joblib`.

---

## How to Run the Application

You need two terminals running simultaneously to run both the frontend and backend.

### 1. Start the FastAPI Backend
Open a terminal and navigate to the backend folder:
```bash
cd emostress-app/backend
```
Activate the virtual environment:
- **Windows**: `.\venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

Start the server:
```bash
uvicorn main:app --reload --port 8000
```
*The API will be available at `http://localhost:8000`*

### 2. Start the React Frontend
Open a **new** terminal and navigate to the frontend folder:
```bash
cd emostress-app/frontend
```
Install dependencies (if you haven't already):
```bash
npm install
```
Start the development server:
```bash
npm run dev
```
*The web app will open automatically, or you can access it at `http://localhost:5173`*

---

## Important Disclaimer
*This system is developed for Empathic Computing academic research purposes only. The stress and emotion results are based on experimental physiological signal patterns and should not be used as medical or psychological diagnosis.*
