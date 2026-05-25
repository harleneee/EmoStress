# Phase 2: System Architecture and Web Development Methodology

This phase details the technical methodology utilized to develop the EmoStress web application. The objective was to create an accessible, full-stack software prototype that bridges the gap between raw physiological machine learning models and end-user interpretability. The system was divided into a high-performance Python backend and a modern, responsive React frontend.

## 2.1 Backend Development (Python & FastAPI)

The backend was engineered to serve as the computational core of the EmoStress system. Instead of relying on heavy web frameworks, **FastAPI** was selected for its high performance, asynchronous capabilities, and native support for Python-based machine learning environments.

### Model Integration and Serialization
The trained machine learning models—specifically the Extra Trees classifiers for both the ECG+GSR and HR+IBI datasets—were serialized using the `joblib` library. This allowed the backend to load the pre-trained weights, feature columns, and scaling metadata directly into memory upon server startup, completely eliminating the need to retrain models during runtime.

### Dynamic Routing Logic
A critical component of the backend methodology was the implementation of a dynamic data router. Because users may possess different types of wearable sensors, the backend was programmed to inspect the uploaded CSV files before processing:
1. **Primary Route:** If the backend detects the presence of `ECG.csv` (and optionally `GSR.csv`), it routes the data through the 30-feature extraction pipeline and utilizes the 83.33% accurate primary Extra Trees model.
2. **Secondary Route:** If ECG data is absent, but `HR.csv` and `IBI.csv` are uploaded, the system gracefully falls back to the 243-feature extraction pipeline, utilizing the 82.15% accurate secondary Extra Trees model.

### Psychological-to-Physiological Stress Mapping
To satisfy the dual requirements of emotion detection and stress estimation, the backend implements a deterministic mapping dictionary. Once the Extra Trees model predicts one of the six base emotions (Anger, Disgust, Fear, Happy, Neutral, Sad), a heuristic logic layer translates the emotional valence and arousal into a physiological stress level ("Low", "Moderate", "High"). For example, high-arousal negative emotions like Fear and Anger are strictly mapped to "High" stress, ensuring programmatic consistency between psychological state and physiological arousal.

## 2.2 Frontend Development (React & Vite)

The client-facing interface was developed using **React**, initialized via the **Vite** build tool for optimal rendering speed and hot-module replacement during development.

### Component-Based Architecture
The frontend was structured using a modular component methodology. Distinct pages were created using `react-router-dom` to separate concerns:
* **Upload Page:** Handles the drag-and-drop file inputs, securely packaging the CSV files into a `FormData` object and transmitting them via HTTP POST requests to the FastAPI backend.
* **Analysis Dashboard:** The primary visualizer that receives the JSON response from the backend and conditionally renders the detected emotion and stress level using dynamic color-coding.
* **Educational Pages:** Static components (Dataset Info, Classification Logic, Model Evaluation) were integrated to maintain absolute academic transparency regarding how the system operates.

## 2.3 UI Modernization and Aesthetic Design

To ensure the prototype met the visual standards of a modern AI health-tech application, a comprehensive UI design methodology was applied using pure CSS.

* **Color Theory & Theming:** A cohesive palette was established using Indigo (`#6366F1`), Teal (`#14B8A6`), and deep Purple (`#8B5CF6`). These cool, clinical colors were chosen to evoke a sense of professional medical technology and calm objectivity.
* **Modern Dashboard Styling:** The application utilizes "glassmorphism" principles, soft linear gradients, and extremely subtle drop-shadows (`box-shadow: 0 4px 15px rgba(0,0,0,0.03)`). Hard borders were removed in favor of heavily rounded cards (`24px` border-radius) to create a friendly, accessible aesthetic.
* **Responsive Layout:** The application relies on CSS Grid and Flexbox to maintain a strictly constrained `1200px` max-width content wrapper. This prevents the dashboard metrics from stretching awkwardly on ultra-wide monitors, while media queries ensure the sidebar collapses cleanly on smaller laptop screens.
