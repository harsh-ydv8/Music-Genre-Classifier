# 🎵 Music Genre Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hrsh-music-genre-classifier.streamlit.app/)

A deep learning web application built with Streamlit that classifies music tracks into one of ten genres. Upload a `.wav` file and see the model predict its genre in real-time.

### ✨ [Live Demo](https://hrsh-music-genre-classifier.streamlit.app/)

---

### 📸 [App Screenshot](https://private-user-images.githubusercontent.com/127317417/495297582-bf438670-83eb-4064-9f89-8f5539a83c30.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTkxNjU2MjYsIm5iZiI6MTc1OTE2NTMyNiwicGF0aCI6Ii8xMjczMTc0MTcvNDk1Mjk3NTgyLWJmNDM4NjcwLTgzZWItNDA2NC05Zjg5LThmNTUzOWE4M2MzMC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwOTI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDkyOVQxNzAyMDZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0wNzU3ZDVmN2QxMmEyOGYzNDhlZWY2OGE2NjNjYTQxM2ZmN2FjYjRmMTJjNTQ1MjA0NDIxOGVkZWVhMzU3ODVlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.IEvIcfyew2jYdPZccQ7EpolT-vGwKkTOMonb-UZPTUE)


## 📚 Project Overview

This project is an end-to-end music genre classification system. It takes raw audio files from the GTZAN dataset, processes them to extract key audio features, and uses these features to train and evaluate multiple machine learning models. The best-performing model, a Convolutional Neural Network (CNN), is then deployed as an interactive web application using Streamlit and containerized with Docker for portability and easy deployment.

### Key Features:

*   **Feature Extraction:** Processes audio files to extract 28 features, including MFCCs, Chroma, Spectral Centroid, and more.
*   **Model Training:** Implements and trains four different models: Logistic Regression, SVM, Random Forest, and a CNN.
*   **Model Evaluation:** Compares models using detailed classification reports and confusion matrices to select the best performer.
*   **Interactive Web App:** A user-friendly interface built with Streamlit that allows users to upload their own `.wav` files for classification.
*   **Containerized Deployment:** The entire application is containerized using Docker, making it portable and easy to deploy.

## 🛠️ Tech Stack

*   **Python:** The core programming language.
*   **Librosa:** For audio processing and feature extraction.
*   **Scikit-learn:** For traditional machine learning models and data preprocessing.
*   **TensorFlow/Keras:** For building and training the Convolutional Neural Network.
*   **Pandas & NumPy:** For data manipulation and numerical operations.
*   **Streamlit:** For building the interactive web application interface.
*   **Docker:** For containerizing the application for deployment.
*   **GitHub:** For version control and hosting the source code.
*   **Streamlit Community Cloud:** For hosting the live application.
### 🗄️[Dataset ➡️ Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data?suggestionBundleId=1320)

## 🚀 Setup and Local Installation

To run this project on your local machine, please follow the steps below.

### 1. Clone the Repository

```bash
git clone https://github.com/[YOUR_GITHUB_USERNAME]/music-genre-classifier.git
cd music-genre-classifier
