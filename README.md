# Mini AI Spam Filter

A comparative project demonstrating the evolution of spam detection from a **Naïve Keyword-Based approach** to a **Modern AI Pipeline** using Semantic Embeddings.

## Overview

This project classifies SMS messages as **Ham** (normal) or **Spam** using the Hugging Face `sms_spam` dataset. It implements and compares two distinct methods:

1.  **Naïve Baseline**: A data-driven rule-based approach that identifies high-frequency "spam keywords" (e.g., *free, claim, txt*).
2.  **AI Pipeline**: A machine learning approach using **Sentence Transformers** (`all-MiniLM-L6-v2`) to generate semantic embeddings and a Logistic Regression classifier.

## Prerequisites

To run this notebook, you need Python installed. Please install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Methodology & Results

The project evaluates both models on a test set (20% split) using Accuracy, Precision, Recall, and F1-Score.

### 1. Naïve Baseline
* **Method**: Counts word frequencies to find top spam indicators. Checks if any top keyword exists in the message.
* **Performance**: High precision but lower recall (misses subtle spam).

### 2. AI Pipeline
* **Method**: Converts text into 384-dimensional dense vectors using a pre-trained Transformer model. Classifies vectors using Logistic Regression.
* **Performance**: Superior understanding of context and semantics.

## Project Structure

The repository is organized as follows:

```text
notebooks/
├── MiniAISpamFilter.ipynb  # Main notebook with data processing, training, and evaluation
├── README.md               # Project documentation (this file)
├── report.pdf              # Comprehensive project report and summary
└── requirements.txt        # List of dependencies required to run the project
```
* **`MiniAISpamFilter.ipynb`**: Contains the complete code for:
    * Data loading & preprocessing.
    * Baseline model implementation.
    * Embedding generation & AI Model training.
    * Evaluation metrics & Visualization.
* **`report.pdf`**: A PDF document summarizing the project background, methodology, and final results.
* **`requirements.txt`**: A text file listing all Python libraries needed to run the notebook.


## Visualizations

The notebook generates:
* **Bar Charts**: Comparing Accuracy, Precision, Recall, and F1 scores.
* **Confusion Matrices**: Visualizing true positives vs. false positives for both models.
