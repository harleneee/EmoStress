# Phase 3: EmoStress Paper Draft

## Abstract
This study presents the development and evaluation of EmoStress, an Artificial Emotional Intelligence (AEI) Python prototype designed to perform objective physiological emotion recognition. Traditional emotion recognition systems often rely on subjective self-reporting or facial analysis, which can be prone to bias or conscious manipulation. To address this limitation, EmoStress utilizes autonomic nervous system signals to classify six distinct emotional states: anger, disgust, fear, happiness, neutrality, and sadness. The system integrates two independently trained Extra Trees algorithms to accommodate different physiological sensor inputs. The primary model, utilizing 30 features extracted from Electrocardiogram (ECG) and Galvanic Skin Response (GSR) data, achieved a classification accuracy of 83.33%. To ensure robust accessibility for standard wearable devices lacking full ECG capabilities, a secondary model was developed utilizing Heart Rate (HR) and Inter-Beat Interval (IBI) data. By extracting a high-dimensional space of 243 Heart Rate Variability (HRV) and statistical features, this alternative pipeline achieved a highly comparable accuracy of 82.15%. The findings demonstrate that non-invasive physiological signals contain highly separable biological markers capable of distinguishing complex human emotions with significant precision. The deployment of such systems carries profound societal implications, particularly in mental health monitoring, adaptive learning environments, and workplace stress management. However, these benefits must be carefully weighed against critical ethical considerations regarding biometric data privacy, algorithmic bias, and the necessity of continuous user consent.

---

## System Description

The EmoStress system is a fully functional Python prototype that translates raw biological signals into structured emotional insights. The system architecture relies on a dynamic backend designed to evaluate whatever physiological data the user uploads, seamlessly routing the data to the appropriate machine learning pipeline. 

### Core Libraries Used
The prototype was engineered using a robust stack of industry-standard Python libraries:
* **Scikit-Learn:** Served as the core machine learning library for training the Extra Trees (Extremely Randomized Trees) classifiers. It also handled the complex preprocessing pipelines, cross-validation, and performance metric calculations.
* **Pandas and NumPy:** Utilized extensively for managing the high-dimensional tabular datasets, handling missing values, and performing the mathematical matrix operations required during HRV feature extraction.
* **FastAPI and Uvicorn:** Provided the framework to deploy the trained models as an asynchronous local web server. FastAPI handles incoming CSV/binary data streams and instantly returns JSON-formatted emotional predictions.
* **Joblib:** Used to serialize and bundle the trained model weights, label encoders, and specific feature column metadata into unified `.joblib` dictionary files for seamless, lightweight deployment.

### Model Accuracy and Implementation
The prototype integrates two specific models:
1. **ECG and GSR Classifier (Primary):** Reached **83.33% accuracy**. It utilizes 30 specific waveform features. The system defaults to this model whenever ECG data is detected, as electrical cardiac activity provides the highest fidelity for emotion mapping.
2. **HR and IBI Classifier (Secondary):** Reached **82.15% accuracy**. Utilizing an expansive 243-feature extraction pipeline, this model serves as a highly robust alternative when users only possess standard smartwatch data (Heart Rate and Inter-Beat Intervals).

*(Instructor Note: Insert screenshots of the EmoStress React UI here showing the Dashboard and Upload screens. You should also insert the classification report texts and the confusion matrix images from your `backend/evaluation` folder to visually prove these accuracies).*

---

## Impact & Discussion

The development of the EmoStress prototype demonstrates a significant leap forward in Empathic Computing, carrying profound societal impacts. By translating invisible physiological stress and emotional arousal into readable data, society can begin to address mental well-being proactively rather than reactively. 

In clinical and therapeutic settings, systems like EmoStress offer healthcare professionals an objective, continuous stream of patient data, circumventing the psychological barriers patients face when asked to verbally express negative valence emotions like fear or sadness. In the workplace, this prototype presents a novel solution to the growing epidemic of occupational burnout. By identifying sustained periods of high-arousal negative emotions, organizations could theoretically intervene before chronic stress causes long-term health deterioration.

However, the societal impact is not entirely positive. The ability to monitor an individual's internal emotional state using wearable devices introduces severe risks of corporate surveillance and emotional profiling. If deployed irresponsibly, institutions could use such AEI systems to monitor employee productivity or emotional compliance, violating fundamental rights to mental privacy.

---

## Conclusion

The findings of this study prove that physiological emotion recognition is highly viable and accurate. However, the deployment of Artificial Emotional Intelligence (AEI) systems like EmoStress must be governed by strict ethical guidelines to prevent the exploitation of biometric data.

The following recommendations are provided for the ethical deployment of AEI:
1. **Explicit and Continuous Consent:** Users must actively opt-in to physiological monitoring. Furthermore, systems must be designed to allow users to effortlessly revoke access to their biometric data at any time without penalty.
2. **Data Minimization and Edge Computing:** AEI systems should adhere to strict data minimization policies. Whenever possible, physiological processing should occur locally on the user's edge device (e.g., their smartphone), transmitting only the anonymized emotional classification rather than the raw heartbeat and electrodermal data.
3. **Mitigation of Algorithmic Bias:** Physiological responses to emotion can vary across different demographics, ages, and medical backgrounds. AEI models must be trained on diverse, highly representative datasets to prevent systemic misinterpretation of minority physiological baselines.
4. **"Human in the Loop" Integration:** AEI should never be used as a standalone diagnostic or punitive tool. Emotion recognition algorithms must act strictly as supplemental decision-support systems, requiring human empathy and psychological context to make final evaluations.
