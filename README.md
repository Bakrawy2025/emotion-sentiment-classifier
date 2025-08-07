# Emotion Sentiment Classifier

A machine learning pipeline for emotion and sentiment classification on text data, leveraging Logistic Regression and TF-IDF vectorization. The project includes dataset analysis, model training, evaluation, and interactive prediction using a trained model.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Dataset Analysis](#dataset-analysis)  
  - [Training the Model](#training-the-model)  
  - [Testing the Model](#testing-the-model)  
- [Evaluation](#evaluation)  
- [Logging](#logging)  
- [Saving and Loading Model](#saving-and-loading-model)  
- [Dependencies](#dependencies)  
- [Folder Structure](#folder-structure)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

This project implements an emotion and sentiment classifier using Python's scikit-learn library. It processes a dataset of sentences labeled by emotions, cleans the data by removing overly repeated entries, trains a Logistic Regression classifier on TF-IDF features, and visualizes the performance via a confusion matrix.

An interactive console tool allows users to input text and get predicted emotion labels in real-time.

---

## Features

- Dataset loading and exploratory analysis  
- Data cleaning by filtering repeated sentences  
- Text vectorization with TF-IDF  
- Emotion classification with Logistic Regression  
- Model evaluation using accuracy, classification report, and confusion matrix visualization  
- Model persistence with `joblib`  
- Interactive prediction interface for user input  
- Logging of training processes  

---

## Dataset

The dataset (`Dataset/sentiment.csv`) contains sentences labeled with emotion categories. Key points:

- Handles multiple emotion labels  
- Cleans sentences repeated more than 3 times to avoid bias  
- CSV should have at least columns: `sentence` and `emotion`

---

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
bash
Copy
Edit
pip install -r requirements.txt
bash
Copy
Edit
pip install pandas scikit-learn matplotlib seaborn joblib
python
Copy
Edit
from dataset import dataset_Analyse

analyzer = dataset_Analyse()
raw_data = analyzer.Dataset()
cleaned_data = analyzer.Cleaned_DS()

print(cleaned_data.head())
python
Copy
Edit
from main_model import Main_model

model_trainer = Main_model()
model_trainer.Train()
This will:

Split data into training and testing (70/30 split, stratified)

Train a Logistic Regression classifier with TF-IDF features

Output accuracy and classification report

Save confusion matrix plot as Confusion_matrix.png

Persist the trained model as Trained_model/Trained_model.pkl

python
Copy
Edit
from test_model import Test_on_model

tester = Test_on_model()
tester.Test_on_data()
Type your text input, and the model will output the predicted emotion label. Type exit to quit.

Accuracy: Measures correct predictions on test data.

Classification Report: Precision, Recall, F1-score per emotion class.

Confusion Matrix: Visual heatmap to analyze prediction errors:



Training activities and debug information are saved to:

bash
Copy
Edit
Logs/Train.log
with timestamps and log levels.

Trained model saved as:

Copy
Edit
Trained_model/Trained_model.pkl
Loaded using joblib.load() for inference.

Python 3.x
pandas
scikit-learn
matplotlib
seaborn
joblib

plaintext
Copy
Edit
.
├── Dataset/
│   └── sentiment.csv
├── Fonts/
│   └── Vazir-Regular-FD.ttf
├── Logs/
│   └── Train.log
├── Trained_model/
│   └── Trained_model.pkl
├── dataset.py
├── main_model.py
├── test_model.py
├── Confusion_matrix.png
├── README.md
└── requirements.txt (optional)
Contributions, issues, and feature requests are welcome!
Feel free to fork the repo and submit pull requests.

Specify your license here (e.g., MIT License).

Created by [Your Name or GitHub handle]
Date: 2025-08-08

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AUX-441/emotion-sentiment-classifier.git
   cd emotion-sentiment-classifier
