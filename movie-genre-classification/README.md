# 🎬 The Enigma Engine: AI Movie Genre Predictor

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.0-F7931E)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75)

This repository contains a full-stack Machine Learning pipeline built for a software engineering internship submission. It uses Natural Language Processing (NLP) to analyze movie plot summaries and predict their genre with high accuracy.

The project features a complete backend training pipeline and a highly customized, cinematic front-end web application built with Streamlit.

## ✨ Features

- **Advanced Text Processing:** Utilizes `TF-IDF` vectorization to convert raw text plots into mathematically significant dense arrays.
- **Robust Machine Learning:** Employs a hyperparameter-tuned `Logistic Regression` algorithm with balanced class weights to handle imbalanced datasets (e.g., thousands of Dramas vs. hundreds of Westerns).
- **Interactive UI:** A custom "Glassmorphism" Streamlit interface featuring a dark mystery aesthetic, custom CSS, and interactive background imagery.
- **Dynamic Probability Analysis:** Uses `predict_proba()` to generate interactive, animated horizontal bar charts via `Plotly`, showcasing the model's exact confidence scores across multiple genres.

## 🚀 Quickstart (Running Locally)

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/movie-genre-classification.git
cd movie-genre-classification
pip install -r requirements.txt
```

### 2. Prepare the Data
Download the [Genre Classification Dataset from Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb).
Extract the zip file and place `train_data.txt` into the `data/` directory.

### 3. Train the Model
Run the pipeline to clean the text, train the Logistic Regression model, and save the binary `.joblib` files to the `models/` directory:
```bash
python main.py --model logistic_regression
```

### 4. Launch the Web App
Start the custom Streamlit interface:
```bash
python -m streamlit run app.py
```
Your browser will automatically open `http://localhost:8501`.

## ☁️ Deployment (Streamlit Cloud)

This project is fully ready to be deployed to the internet for free via Streamlit Community Cloud:
1. Ensure your trained models (`logistic_regression.joblib` and `vectorizer.joblib`) are generated.
2. Push this entire repository to GitHub (The `.gitignore` is pre-configured to safely ignore your massive Kaggle dataset, but it will keep your models!).
3. Go to [share.streamlit.io](https://share.streamlit.io/), link your GitHub account, and select `app.py` as your main file.
4. Click Deploy!

## 📊 Evaluation Metric
Because the dataset is heavily imbalanced, standard Accuracy is a misleading metric. This model is rigorously evaluated using the **Macro F1-Score**, ensuring it performs equally well on rare genres as it does on common ones.
