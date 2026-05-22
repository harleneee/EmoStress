# Chapter 3: Methodology

## 3.1 Data Collection

To perform accurate physiological emotion recognition, the EmoStress system utilizes two distinct datasets that capture objective bodily responses associated with emotional and stress states. These physiological signals bypass subjective self-reporting biases and capture direct autonomic nervous system responses. 

### A. ECG + GSR Emotion Dataset
The primary emotion classifier was trained using the ECG + GSR Emotion Dataset. This dataset focuses on two critical physiological markers:
- **Electrocardiogram (ECG):** Reflects the electrical activity of the heart. The autonomic nervous system modulates heart activity during emotional arousal, making ECG a strong indicator of both valence (positive/negative emotion) and arousal levels.
- **Galvanic Skin Response (GSR):** Also known as electrodermal activity, GSR measures continuous variations in the electrical characteristics of the skin. Emotional arousal triggers sweat gland activity, directly altering skin conductance.
- **Labels:** The dataset categorized emotional states into six classes: *anger, disgust, fear, happy, neutral,* and *sad*.

### B. ECSMP HR + IBI Dataset
The secondary classifier was trained using the ECSMP dataset, specifically utilizing Heart Rate (HR) and Inter-Beat Interval (IBI) data derived from wearable recordings. 
- **Heart Rate (HR):** Represents the number of heartbeats per minute. Shifts in HR are often the most immediate physiological response to stressors and emotional stimuli.
- **Inter-Beat Interval (IBI):** Represents the exact time interval between successive heartbeats. IBI is critical because it allows for the calculation of Heart Rate Variability (HRV), a proven biomarker for emotional regulation and psychological stress.
- **Labels:** This dataset also categorized emotional states into the same six classes: *anger, disgust, fear, happy, neutral,* and *sad*.

## 3.2 Data Processing

Raw physiological signals contain noise and are highly dimensional; therefore, they cannot be fed directly into machine learning models. A robust data processing pipeline was developed for both models to transform raw signals into meaningful numerical features.

### Processing for ECG + GSR Emotion Classifier
1. **Feature Extraction:** Raw and pre-processed ECG and GSR signals were processed to extract a concentrated set of numerical features. These include signal-level statistical metrics (e.g., mean, standard deviation, peak frequencies) and physiological characteristics.
2. **Feature Dimensionality:** The final extraction resulted in a compact array of **30 numerical features**.
3. **Training:** These 30 processed features were directly used to train the Random Forest emotion classifier.

### Processing for ECSMP HR + IBI Emotion Classifier
1. **Segmentation and Extraction:** Raw `HR.csv` and `IBI.csv` data streams were loaded and segmented using sliding windows (120-second windows with a 15-second step size). Raw time-series data is not directly passed to the model. Instead, extensive statistical and HRV-related features are extracted from the HR and IBI arrays.
2. **Baseline Calibration:** The extracted features were calibrated against individual baseline readings to account for physiological differences between subjects.
3. **Feature Dimensionality:** This comprehensive extraction process yielded a highly detailed array of **243 numerical features**.
4. **Pipeline Construction:** The processed features were aligned with the trained model's designated feature columns. The final system utilizes an Extra Trees Pipeline bundled dictionary containing the model (`best_model_name`), the model Pipeline object (`model`), the exact required `feature_columns`, `labels`, an `emotion_to_stress` mapping for secondary interpretation, `settings`, and evaluation `metrics`.

## 3.3 Models Used

The EmoStress system leverages two independent final models, dynamically choosing the appropriate classifier based on the data provided by the user. 

### A. ECG + GSR Emotion Classifier
- **Algorithm:** Random Forest
- **Test Accuracy:** 83.33%
- **Feature Count:** 30
- **Emotion Classes:** anger, disgust, fear, happy, neutral, sad

**Suitability:** Random Forest is an ensemble learning method constructed from multiple decision trees. It is highly suitable for this task because it effectively handles non-linear relationships inherent in physiological data, mitigates the risk of overfitting through bagging, and natively supports tabular extracted features. Furthermore, it allows for clear feature importance analysis, ensuring model interpretability.

**Model Architecture:**
```mermaid
graph LR
    A[ECG & GSR Input] --> B[Feature Extraction]
    B --> C[30 Extracted Features]
    C --> D[Random Forest Classifier]
    D --> E[6 Emotion Classes]
    E --> F[Emotion Result]
```

![ECG Emotion Confusion Matrix](../evaluation/ECG_emotion_confusion_matrix_normalized.png)
![ECG Feature Importance](../evaluation/ECG_emotion_feature_importance.png)

### B. ECSMP HR + IBI Emotion Classifier
- **Algorithm:** Extra Trees (Extremely Randomized Trees) Pipeline
- **Test Accuracy:** 82.15%
- **Feature Count:** 243
- **Emotion Classes:** anger, disgust, fear, happy, neutral, sad

**Suitability:** The Extra Trees algorithm is an ensemble tree-based method similar to Random Forest but utilizes randomized splits at the nodes rather than seeking the optimum split. This randomization makes it exceptionally robust for high-dimensional feature spaces (such as the 243 features extracted here) and significantly reduces variance and overfitting. It performed best among all candidate models evaluated for the ECSMP dataset.

**Model Architecture:**
```mermaid
graph LR
    A[HR.csv + IBI.csv] --> B[Windowed Segmentation]
    B --> C[HR/IBI/HRV Feature Extraction]
    C --> D[243 Extracted Features]
    D --> E[Extra Trees Pipeline]
    E --> F[6 Emotion Classes]
    F --> G[Emotion Result]
    G --> H[Stress Interpretation]
```

![HR+IBI Emotion Confusion Matrix](../evaluation/HR_IBI_final_best_confusion_matrix_normalized.png)

---

# Chapter 4: Discussion and Analysis of Results

This section analyzes the performance of the trained classifiers, examining what the results indicate regarding the feasibility of physiological emotion recognition.

## 4.1 ECG + GSR Emotion Classifier Results
The primary ECG + GSR model achieved a strong test accuracy of **83.33%**, successfully distinguishing between the six complex emotion categories. The classification report indicates exceptional performance for the "Happy" class (1.00 F1-score) and strong results for "Disgust" (0.89 F1-score). 

The confusion matrix suggests minor confusion boundaries between certain high-arousal negative emotions, which is expected as physiological responses (like elevated heart rate and skin conductance) during anger and fear can exhibit similarities. However, the overall high accuracy confirms that the combination of cardiovascular (ECG) and electrodermal (GSR) signals provides a robust, multi-modal foundation for emotion detection. Because emotions directly stimulate the autonomic nervous system, the physical manifestations captured by these 30 features act as highly reliable predictors.

## 4.2 ECSMP HR + IBI Emotion Classifier Results
The secondary ECSMP model, functioning exclusively on HR and IBI inputs, achieved an impressive **82.15%** test accuracy utilizing the Extra Trees algorithm. According to the classification report, it performs with high consistency across classes, notably achieving an 0.88 F1-score for "Neutral". 

This result is highly significant. It demonstrates that even in the absence of explicit ECG waveform data or GSR sweat measurements, basic heart rhythm metrics (HR) combined with the precise timing between beats (IBI) are sufficient to capture the physiological arousal patterns required to classify six distinct emotions. The high feature count (243) is necessary here, as the model must derive complex, subtle HRV frequency and time-domain patterns from simpler inputs to match the performance of multi-modal ECG/GSR data.

Furthermore, this HR + IBI model is exceptionally practical for real-world application. Within the EmoStress web interface, users are much more likely to have access to exported HR.csv and IBI.csv files from standard smartwatches than they are to have clinical ECG/GSR equipment.

## 4.3 Comparison of the Two Models
When evaluating the two approaches:
- **ECG + GSR Model:** 83.33% accuracy, 30 features, Random Forest
- **HR + IBI Model:** 82.15% accuracy, 243 features, Extra Trees

Both models proved highly effective, comfortably exceeding the 80% accuracy threshold. The ECG + GSR model achieved a slightly higher accuracy with far fewer features (30 vs 243). This efficiency implies that raw ECG and GSR data inherently contain dense, easily separable emotional markers. 

Conversely, the HR + IBI model requires massive feature extraction (243 features) to uncover the hidden emotional markers within simpler heart rate data, yet it still performs nearly as well. Ultimately, while the ECG + GSR model is slightly more accurate theoretically, the HR + IBI model is more practical and accessible for standard users of the EmoStress website.

It is also important to note that within the EmoStress system workflow, stress is no longer treated as an isolated classification target. Instead, both models operate strictly as emotion classifiers. Stress levels are interpreted as a secondary, derivative layer based directly on the primary emotion output (e.g., mapping "Fear" or "Anger" to High Stress, and "Happy" or "Neutral" to Low Stress).

---

# Chapter 5: Conclusion and Future Work

## 5.1 Conclusion
The EmoStress project successfully demonstrates that physiological signals can be reliably used for objective emotion recognition, supporting the broader goals of empathic computing. By bypassing subjective self-reporting, the system accurately interprets human emotional states directly from autonomic nervous system responses.

The final implemented system utilizes two trained emotion classifiers:
1. A primary **ECG + GSR Emotion Classifier** that achieved 83.33% accuracy.
2. A highly accessible **ECSMP HR + IBI Emotion Classifier** that achieved 82.15% accuracy.

Both models proved capable of classifying states of anger, disgust, fear, happy, neutral, and sad. By integrating these models into a cohesive web application, EmoStress provides a seamless platform where users can upload physiological data, receive instant emotional classification, and view derived stress interpretations, successfully bridging the gap between raw biological signals and actionable empathic insights.

## 5.2 Future Work
While the current EmoStress system is highly functional, several avenues for future research and development remain:

- **Dataset Expansion:** Improving the overall dataset size and ensuring a perfectly balanced distribution of classes will help models generalize better to unseen data.
- **Real-World Validation:** Testing the models on data collected in natural, real-world environments—rather than strictly controlled lab conditions—will validate the robustness of the system.
- **Live Integration:** Transitioning from static CSV file uploads to real-time data streaming via API integrations with smartwatches or specialized devices (like the Empatica E4).
- **Personalized Baselines:** Enhancing the HR/IBI preprocessing pipeline to include deep, personalized baseline calibration per user, accounting for individual resting heart rates and unique physiological quirks.
- **Expanded Emotion Models:** Broadening the classification scope beyond the base six emotions to include nuanced affective states such as frustration, excitement, or cognitive load.
- **Explainability:** Enhancing the web interface with Explainable AI (XAI) features so that users can visually understand exactly which physiological features (e.g., a sudden drop in HRV) led to a specific emotion or stress prediction.
