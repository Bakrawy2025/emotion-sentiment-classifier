Emotion Sentiment Classifier
A machine learning pipeline for emotion and sentiment classification on text data, leveraging Logistic Regression and TF-IDF vectorization. The project includes dataset analysis, model training, evaluation, and interactive prediction using a trained model.

Table of Contents
Overview

Features

Dataset

Installation

Usage

Dataset Analysis

Training the Model

Testing the Model

Evaluation

Logging

Saving and Loading Model

Dependencies

Folder Structure

Contributing

License

Overview
This project implements an emotion and sentiment classifier using Python's scikit-learn library. It processes a dataset of sentences labeled by emotions, cleans the data by removing overly repeated entries, trains a Logistic Regression classifier on TF-IDF features, and visualizes the performance via a confusion matrix.

An interactive console tool allows users to input text and get predicted emotion labels in real-time.

Features
Dataset loading and exploratory analysis

Data cleaning by filtering repeated sentences

Text vectorization with TF-IDF

Emotion classification with Logistic Regression

Model evaluation using accuracy, classification report, and confusion matrix visualization

Model persistence with joblib

Interactive prediction interface for user input

Logging of training processes

Dataset
The dataset (Dataset/sentiment.csv) contains sentences labeled with emotion categories.

Handles multiple emotion labels

Cleans sentences repeated more than 3 times to avoid bias

The CSV file should have at least the columns: sentence and emotion

Installation
Clone the repository:

Bash

git clone https://github.com/AUX-441/emotion-sentiment-classifier.git
cd emotion-sentiment-classifier
Create and activate a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
Install required packages:

Bash

pip install -r requirements.txt
Note: If requirements.txt is not provided, install the following manually:

Bash

pip install pandas scikit-learn matplotlib seaborn joblib
Usage
Dataset Analysis
Run the dataset analyzer to load and clean data:

Python

from dataset import dataset_Analyse

analyzer = dataset_Analyse()
raw_data = analyzer.Dataset()
cleaned_data = analyzer.Cleaned_DS()
print(cleaned_data.head())
Training the Model
Run the training script to train and save the classifier:

Python

from main_model import Main_model

model_trainer = Main_model()
model_trainer.Train()
This script will perform the following actions:

Split the data into training and testing sets (70/30 split, stratified).

Train a Logistic Regression classifier with TF-IDF features.

Output the accuracy and classification report.

Save the confusion matrix plot as Confusion_Matrix.png.

Persist the trained model as Trained_model/Trained_model.pkl.

Testing the Model
Use the interactive script to input sentences and get predicted emotions:

Bash

python test_model.py
Type your text input, and the model will output the predicted emotion label. Type exit to quit.

Evaluation
Accuracy: Measures the percentage of correct predictions on the test data.

Classification Report: Provides precision, recall, and F1-score for each emotion class.

Confusion Matrix: A visual heatmap to analyze prediction errors.

Logging
Training activities and debug information are saved with timestamps and log levels to:

Bash

Logs/Train.log
Saving and Loading Model
The trained model is saved using joblib at the following path:

Bash

Trained_model/Trained_model.pkl
The model can be loaded for inference using joblib.load().

Dependencies
Python 3.x

pandas

scikit-learn

matplotlib

seaborn

joblib

Folder Structure
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
├── Confusion_Matrix.png
├── README.md
└── requirements.txt (optional)
Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit pull requests.
