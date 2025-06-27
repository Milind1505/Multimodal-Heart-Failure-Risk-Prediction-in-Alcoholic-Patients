Multimodal Heart Failure Risk Prediction in Alcoholic Patients

Overview

This project presents a production-ready multimodal deep learning pipeline designed to predict heart failure risk in alcoholic patients by integrating structured tabular health metadata with ECG plot image data. It encapsulates real-world healthcare AI challenges such as data fusion, model robustness, and interpretability, while showcasing end-to-end model deployment using Gradio for instant clinical simulation.

As a Senior Data Scientist, my goal was to demonstrate the power of cross-modal representation learning for sensitive clinical risk prediction, tackling both tabular and image modalities in a unified inference workflow.

Problem Statement

Excessive alcohol consumption is a known risk factor for cardiovascular diseases, but early detection of heart failure risk remains non-trivial, especially when signals are embedded across ECG patterns and patient lifestyle/comorbidity data. Traditional models often silo modalities, leading to underfitting or clinical oversights.

Why Multimodal?

Combining ECG plots (visual cardiac activity) with structured health metadata (age, diabetes, alcohol intake, etc.) enables more nuanced, data-efficient, and clinically contextualized predictions. This project delivers a fusion-based solution that capitalizes on the complementarity of tabular and visual features, simulating how clinicians make informed diagnoses from multiple inputs.

Model Architecture

This solution is built around a hybrid neural architecture with two synchronized branches:

Tabular Metadata Branch
Input: Normalized patient data (e.g., age, weekly alcohol units, comorbidities)
Layers: Dense → ReLU → Dropout
ECG Image Branch
Input: Resized ECG plot (.png)
Layers: Pre-trained MobileNetV2 (frozen) → Global Avg Pooling → Dense → Dropout
Fusion Head
Concatenated features from both branches
Post-Fusion: Dense layers for joint reasoning
Output: Sigmoid-activated neuron for binary classification (Low Risk / High Risk)

Files & Structure

multimodal_heart_failure_risk_prediction_in_alcoholic_patients.py:
Self-contained app with model definition, loading utilities, preprocessing pipeline, and Gradio UI.
hybrid_model_weights.h5: Trained weights (not in repo).
scaler.pkl, label_encoder.pkl: Fitted preprocessing objects.
patient_cardiovascular_risk_data.csv: Metadata input (user uploads this).
ecg_plots_for_patient_cardiovascular_risk/: Folder with ECG images for each patient.

Inference Workflow (Gradio UI)

Upload ECG Plot (PNG format).
Input Patient Metadata (Age, Gender, Alcohol Consumption, Duration, Diabetes, Hypertension).
Click 'Predict' to get:
Risk Category: Low or High
Probability Score: e.g., "74.2% Risk"
This allows on-the-fly inference and mimics a real-world diagnostic assistant.

Deployment

This project is designed for one-click deployment on Hugging Face Spaces using Gradio.

How to Run Locally

pip install -r requirements.txt
python multimodal_heart_failure_risk_prediction_in_alcoholic_patients.py

Deploy on Hugging Face
Ensure the following are included in your repo:

multimodal_heart_failure_risk_prediction_in_alcoholic_patients.py
requirements.txt
Model weights + scaler + encoder files
Hugging Face will auto-run the Gradio app once deployed.

Model Training & Validation

Optimizer: Adam
Loss: Binary Crossentropy
Metrics: Accuracy, AUC-ROC
Callbacks: EarlyStopping, ReduceLROnPlateau
Typical AUC on validation set: ~0.85+
Training is omitted from this deployment version but available in Jupyter format (.ipynb upon request).

Ethical Considerations

This model was developed for real-world experimental use under the supervision of trained medical professionals. While it demonstrates the potential of multimodal AI in healthcare, it is not intended for clinical decision-making without formal validation. It serves as a research prototype to explore the integration of image and tabular data for diagnostic support and should not be deployed in production environments without appropriate regulatory approvals.

License

MIT License
