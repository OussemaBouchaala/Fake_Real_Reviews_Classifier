# Fake Review Detection System

A machine learning-powered web application that detects fake reviews using text analysis and behavioral patterns.

## Features

- **Text Analysis**: TF-IDF vectorization to analyze review content
- **Behavioral Features**: Detects patterns like excessive punctuation, repetition, and capitalization
- **ML Classification**: Linear SVM model for accurate fake/real classification
- **Web Interface**: Interactive Streamlit application for real-time predictions

## Project Structure

```
Summer_Olym_Project/
├── app/
│   ├── streamlit.py       # Main Streamlit application
│   ├── config.py          # Configuration loader
│   └── utils.py           # Helper functions
├── data/                  # Datasets
├── models/                # Trained ML models
├── notebooks/             # Jupyter notebooks
└── config.json            # Centralized configuration
```

## Installation

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

## Usage

```bash
cd app
streamlit run streamlit.py
```

## How It Works

1. **Text Preprocessing**: Removes punctuation, numbers, and stopwords
2. **Feature Extraction**: 
   - TF-IDF features from cleaned text
   - Behavioral features (length, capitals, repetition)
3. **Classification**: Combined features fed to SVM model
4. **Prediction**: Returns REAL or FAKE review classification

## Technologies

- Python 3.x
- Streamlit
- scikit-learn
- NLTK
- NumPy & SciPy
