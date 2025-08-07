# Emotion Sentiment Classifier

A machine learning pipeline for emotion and sentiment classification on text data, leveraging Logistic Regression and TF-IDF vectorization. The project includes dataset analysis, model training, evaluation, and interactive prediction using a trained model.

## Features

- ✨ **Fast TF-IDF + Logistic Regression**: Efficient sentiment classification  
- 🔄 **Data Cleaning**: Filters out sentences repeated more than 3 times to improve robustness  
-  📈 **Evaluation Metrics**: Shows accuracy, classification report, and confusion matrix  
- 💾 **Model Persistence**: Saves and loads the trained model using `joblib`  
-  ⌨️ **Interactive CLI**: Lets users input text and get emotion predictions in real-time  
- 📝 **Training Logs**: Detailed logs saved with timestamps (`Logs/Train.log`)  
-  🪶 **Lightweight Pipeline**: Clean, easy-to-run training and testing flow  

---

## Overview

This project implements an emotion and sentiment classifier using Python’s scikit-learn library. It processes a dataset of sentences labeled by emotions, cleans the data by removing overly repeated entries, trains a Logistic Regression classifier on TF-IDF features, and visualizes the performance via a confusion matrix.

An interactive console tool allows users to input text and get predicted emotion labels in real-time.

## Features

- Dataset loading and exploratory analysis  
- Data cleaning by filtering repeated sentences  
- Text vectorization with TF-IDF  
- Emotion classification with Logistic Regression  
- Model evaluation using accuracy, classification report, and confusion matrix visualization  
- Model persistence with `joblib`  
- Interactive prediction interface for user input  
- Logging of training processes  

## Dataset

The dataset (`Dataset/sentiment.csv`) contains sentences labeled with emotion categories.

- Handles multiple emotion labels  
- Cleans sentences repeated more than 3 times to avoid bias  
- The CSV file should have at least the columns: `sentence` and `emotion`  


## Evaluation

- **Accuracy**: Measures the percentage of correct predictions on the test data  
- **Classification Report**: Provides precision, recall, and F1-score for each emotion class  
- **Confusion Matrix**: A visual heatmap to analyze prediction errors  

![Confusion Matrix](Confusion_Matrix/Confusion_Matrix.png)


## Installation

```bash
git clone https://github.com/AUX-441/emotion-sentiment-classifier.git
cd emotion-sentiment-classifier

python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

pip install -r requirements.txt

# If requirements.txt is not provided:
pip install pandas scikit-learn matplotlib seaborn joblib
