# 🫁 Lung Cancer Detection using Computer Vision

### 📌 Project by:
- **Siddharth Linga** – Dept. of Computer Science, Texas A&M University - Corpus Christi  
  📧 slinga1@islander.tamucc.edu  

---

## 📝 Abstract

This project aims to build a **Computer-Aided Diagnosis (CAD) system** for **early and accurate detection of lung cancer** using advanced computer vision and deep learning techniques. By leveraging Convolutional Neural Networks (CNNs), the system classifies and localizes lung tumors in CT scan images, aiding healthcare professionals in making timely, informed decisions.

---

## 🚀 Project Objectives

- Review and evaluate existing lung cancer detection methods.
- Design and develop an AI-powered prototype for early lung cancer detection.
- Train and validate the system using real clinical datasets.
- Assess the model’s performance and its potential impact on healthcare outcomes.

---

## 🧠 Methodology

1. **Data Collection**  
   - Large dataset of annotated CT scans and chest X-rays collected from multiple healthcare sources.
   - Ensures diversity in age, gender, ethnicity, and lung conditions.
   - Annotations by expert radiologists ensure high-quality ground truth.

2. **Model Architecture**  
   - CNN-based model inspired by VGG-like architecture.
   - Input: 224x224 RGB images  
   - Layers:  
     - Convolution (ReLU) → MaxPooling  
     - Flatten → Dense (ReLU) → Dense (Softmax with 3 classes: `lung_aca`, `lung_n`, `lung_scc`)

3. **Training**  
   - Techniques used: Data augmentation, Adam optimizer, categorical cross-entropy.
   - Metrics: Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity.
   - Achieved:  
     - **Accuracy:** 92%  
     - **Sensitivity:** 89%  
     - **Specificity:** 93%

4. **System Development**  
   - User-friendly interface for uploading images and viewing predictions.
   - Integration with Hospital Information Systems (HIS) via HL7/FHIR protocols.

5. **Validation**  
   - Independent test datasets used for final evaluation.
   - Real-world pilot testing and user feedback from clinical professionals.

---

## 🖼️ Features

- ✅ Non-invasive, automated lung cancer detection.
- 📊 High diagnostic accuracy using CNN.
- 💡 Highlights abnormal regions on CT images.
- 🔒 Fully anonymized and privacy-compliant dataset.
- 🧑‍⚕️ Intuitive UI for healthcare professionals.
- 🔗 Integration-ready with EHR and clinical IT systems.

---

## 🧪 Results

- High model performance in real clinical environments.
- Positive user satisfaction and workflow enhancement.
- Demonstrated capability to detect early-stage lesions accurately.

---

## 📈 Future Scope

- Expand dataset with more varied pathology.
- Introduce ensemble models or GANs for better robustness.
- Integrate real-time learning from new data.
- Tailor predictions for specific specialists (e.g., oncologists, radiologists).
- Deploy lightweight versions for under-resourced health facilities.

---

## 📂 Repository Structure

```bash
📁 lung-cancer-detection-cv/
├── data/                     # Sample data (de-identified)
├── models/                   # Trained model weights (if shareable)
├── src/                      # Model architecture, training, preprocessing scripts
├── ui/                       # Frontend interface code
├── results/                  # Output screenshots and prediction examples
├── report/                   # Final project report (PDF)
├── README.md                 # This file
└── requirements.txt          # Python dependencies
