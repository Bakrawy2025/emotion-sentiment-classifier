# Emotion Sentiment Classifier

A machine learning pipeline for emotion and sentiment classification on text data, leveraging Logistic Regression and TF-IDF vectorization. The project includes dataset analysis, model training, evaluation, and interactive prediction using a trained model.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
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

## Evaluation
Accuracy: Measures correct predictions on test data.

Classification Report: Precision, Recall, F1-score per emotion class.

Confusion Matrix: Visual heatmap to analyze prediction errors:

Confusion_Matrix.png



## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AUX-441/emotion-sentiment-classifier.git
   cd emotion-sentiment-classifier
