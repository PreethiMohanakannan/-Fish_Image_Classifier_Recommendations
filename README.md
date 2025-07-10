# 🐟 Fish Image Classifier & Predictor

> A visually engaging deep learning web app that classifies fish species using the **InceptionV3**, , **CNN** model, built with **Streamlit**, and enhanced with a soothing seashore UI.

which includes data preprocessing, augmentation, transfer learning with InceptionV3, and a **Streamlit-powered app** for real-time image predictions.

---
## 🌟 Key Features

- 🔍 **Classifies fish species** from uploaded images
- 🎯 **Top-3 predictions** with confidence scores
- 🧠 Powered by **InceptionV3 (Keras)**, trained on a custom dataset
- 🎨 Beautiful **seashore-themed UI** with responsive design
- ⚡ Fast model loading with `@st.cache_resource`

---

## 🗂 Project Structure
📦 Fish_Image_Classifier_Recommendations
├── fish.py # Streamlit web app
├── fish.ipynb # Model training/testing notebook
├── inception_model.h5 # Trained InceptionV3 model
├── class_indices.json # Mapping of class indices to labels
└── README.md # This file

---

## 📁 Dataset Structure

The dataset is organized as follows:
dataset/
├── train/
│ ├── Goldfish/
│ ├── Salmon/
│ └── ... [other fish species]
├── val/
│ └── [same structure as train]
└── test/
└── [same structure as train]

---

## 🔄 Data Loading

Using **TensorFlow’s `ImageDataGenerator`**:

- Loads batches from folders
- Infers class labels automatically
- Uses `class_mode='categorical'` for multiclass classification

---

## 🧠 CNN Model Architecture

A custom sequential CNN built using Keras:
Input: (292x292x3)
↓ Conv2D (32 filters) + MaxPooling
↓ Conv2D (64 filters) + MaxPooling
↓ Conv2D (128 filters) + MaxPooling
↓ Flatten
↓ Dense (128) + ReLU + Dropout(0.5)
↓ Output layer (Softmax, #classes)

## 📈 Training & Evaluation

- Trained on augmented dataset  
- Evaluated using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - Confusion Matrix  

📊 Training performance was visualized over epochs

---

## 💾 Model Saving

- Saved trained model as `.h5` using `ModelCheckpoint`
- Exported `class_indices.json` for label decoding

---

## 🖥️ Streamlit Deployment

> Interactive web app using **Streamlit** for fish classification

👩‍💻 Author
Preethi
🐍 Python | 🧠 Deep Learning | 🌊 Passionate about Marine Life

