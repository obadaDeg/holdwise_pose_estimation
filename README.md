
# HoldWise Pose Estimation Project

This repository contains the implementation for HoldWise's posture detection and classification using multi-modal data. The primary focus is on recognizing good and bad posture while using mobile devices, integrating pose estimation and sensor data for improved accuracy.

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [Dataset](#dataset)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Scripts](#scripts)
- [Visualization](#visualization)
- [Contributors](#contributors)

---

## Overview
HoldWise is a mobile application aimed at improving user posture. This project leverages machine learning models and pose estimation techniques to classify posture into three categories:
1. Good posture
2. Bad posture
3. Cannot determine

The models are trained on image data and sensor readings from gyroscope and accelerometer inputs to provide accurate posture assessment.

---

## Directory Structure
```
obadadeg-holdwise_pose_estimation/
├── README.md                  # Project documentation
├── Training_Approach_1.csv    # Training logs for approach 1
├── Training_Approach_2.csv    # Training logs for approach 2
├── analysis.py                # Script for data analysis
├── extract.py                 # Script to extract training data
├── generate_plots.py          # Script to generate performance plots
├── index.html                 # Web interface for dataset management
├── multi_modal_posture_model.h5  # Keras model for multi-modal data
├── multi_modal_posture_model.tflite # TensorFlow Lite version of the model
├── pose_classifier_v2.ipynb   # Jupyter Notebook for pose classifier
├── pose_classifier_v2.py      # Python script version of the pose classifier
├── posture_classifier.py      # Script for data preprocessing and model integration
├── tensorflowjs_converter     # TensorFlow.js conversion tool
└── output/                    # Folder containing generated plots and metrics
```

---

## Setup Instructions

### Prerequisites
- Python 3.10
- TensorFlow 2.x
- Keras
- OpenCV
- MediaPipe
- Pandas, Matplotlib, NumPy

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/obadadeg/holdwise_pose_estimation.git
   cd holdwise_pose_estimation
   ```
2. Create a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate   # Linux/Mac
   .\myenv\Scripts\activate   # Windows
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset
The dataset used for training consists of images and corresponding sensor data categorized into:
- `good_pose`
- `bad_pose`
- `cant_determine`

Each category contains images extracted from videos, along with synchronized sensor data.

---

## Model Training and Evaluation
### Training Approaches:
1. **Pose Estimation Only**
2. **Multi-modal Approach (Pose + Sensor Data)**

The models are trained using TensorFlow/Keras with performance metrics logged and saved in `Training_Approach_1.csv` and `Training_Approach_2.csv`.

### Model Files:
- `multi_modal_posture_model.h5`: Keras model for multi-modal data
- `multi_modal_posture_model.tflite`: TensorFlow Lite version for deployment

---

## Scripts
- **`analysis.py`**: Performs data analysis and computes statistics.
- **`extract.py`**: Extracts and processes training data from logs.
- **`generate_plots.py`**: Generates performance plots comparing different training approaches.
- **`pose_classifier_v2.ipynb`**: Jupyter Notebook for training the pose classifier.
- **`posture_classifier.py`**: Data preprocessing, augmentation, and model training script.

---

## Visualization
Generated plots include:
- Training and Validation Accuracy
- Training and Validation Loss
- Accuracy Difference Over Epochs
- Performance Efficiency

Plots are saved in the `output/` directory.

---

## Contributors
- **Obada Daghlas** - Project Lead and Developer
