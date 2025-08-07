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

## 🌟 Overview

🧠 This project is an **Emotion & Sentiment Classifier** built using Python’s powerful **scikit-learn** library.  
📜 It processes text data, removes overly repeated sentences, trains a **Logistic Regression** model on **TF-IDF** features, and visualizes the performance with a **Confusion Matrix**.  
💬 An interactive console tool lets you input your own text and instantly see the model’s prediction.  

---

## 🚀 Features

- 📂 **Dataset loading & analysis** – Inspect data and explore emotion labels  
- 🧹 **Data cleaning** – Remove sentences repeated more than 3 times to reduce bias  
- ✍️ **TF-IDF vectorization** – Convert text into optimized numerical features  
- 🤖 **Emotion classification with Logistic Regression**  
- 📊 **Model evaluation** – Accuracy score, classification report, and a visual confusion matrix  
- 💾 **Model persistence** – Save and load trained models using `joblib`  
- 🖥 **Interactive predictions** – Real-time predictions from user input  
- 📝 **Logging** – Keep track of training process  

---

## 📊 Evaluation

- **Accuracy** – Percentage of correct predictions on the test data  
- **Classification Report** – Precision, recall, and F1-score for each emotion class  
- **Confusion Matrix** – Heatmap visualization of prediction results  

![Confusion Matrix](Confusion_Matrix/Confusion_matrix.png)

---

## ⚙️ Installation

```bash
git clone https://github.com/AUX-441/emotion-sentiment-classifier.git
cd emotion-sentiment-classifier

python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

pip install -r requirements.txt

# If requirements.txt is not available:
pip install pandas scikit-learn matplotlib seaborn joblib
## 🌟 Overview

🧠 This project is an **Emotion & Sentiment Classifier** built using Python’s powerful **scikit-learn** library.  
📜 It processes text data, removes overly repeated sentences, trains a **Logistic Regression** model on **TF-IDF** features, and visualizes the performance with a **Confusion Matrix**.  
💬 An interactive console tool lets you input your own text and instantly see the model’s prediction.  

---

## 🚀 Features

- 📂 **Dataset loading & analysis** – Inspect data and explore emotion labels  
- 🧹 **Data cleaning** – Remove sentences repeated more than 3 times to reduce bias  
- ✍️ **TF-IDF vectorization** – Convert text into optimized numerical features  
- 🤖 **Emotion classification with Logistic Regression**  
- 📊 **Model evaluation** – Accuracy score, classification report, and a visual confusion matrix  
- 💾 **Model persistence** – Save and load trained models using `joblib`  
- 🖥 **Interactive predictions** – Real-time predictions from user input  
- 📝 **Logging** – Keep track of training process  

---

## 📊 Evaluation

- **Accuracy** – Percentage of correct predictions on the test data  
- **Classification Report** – Precision, recall, and F1-score for each emotion class  
- **Confusion Matrix** – Heatmap visualization of prediction results  

![Confusion Matrix](Confusion_Matrix/Confusion_matrix.png)

---

## ⚙️ Installation

```bash
git clone https://github.com/AUX-441/emotion-sentiment-classifier.git
cd emotion-sentiment-classifier

python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

pip install -r requirements.txt

# If requirements.txt is not available:
pip install pandas scikit-learn matplotlib seaborn joblib
