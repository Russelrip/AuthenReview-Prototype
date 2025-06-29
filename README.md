# AuthenReview-Prototype

A Flask-based web application and machine learning pipeline for detecting fake reviews using a Random Forest classifier.

## Features

* **Preprocessing**: Cleans review text (HTML tag removal, URL removal, non-letter filtering, stopword removal).
* **Training**: `train.py` script for training and hyperparameter tuning via GridSearchCV.
* **Model Serialization**: Saves the best pipeline as `model_pipeline.pkl`.
* **Web App**: `app.py` serves a responsive interface (Bootstrap 5) to upload CSVs of reviews or paste raw text, runs predictions, and displays:

  * **Model Performance**: Accuracy, precision, recall, F1 (when ground-truth labels exist).
  * **Prediction Breakdown**: Counts and percentages of real vs. fake reviews.
  * **Sample Predictions**: Tabular view with badges—Real (0) vs. Fake (1).
* **Auto-Download Model**: On startup, the app fetches `model_pipeline.pkl` from the GitHub release if not present locally.

## Repository Structure

```
AuthenReview-Prototype/
├── app.py               # Flask application (with text-area & file upload paths)
├── train.py             # Script to train & serialize the model pipeline
├── preprocessing.py     # TextCleaner transformer
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # HTML template (Bootstrap 5; cards, badges, responsive)
├── static/
│   └── custom.css       # Optional custom styles (background, badge sizes)
└── uploads/             # (auto-created) folder for user-uploaded CSVs
```

## Prerequisites

* Python 3.8 or higher
* Git
* (Optional) Virtual environment tool (venv, conda)

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/Russelrip/AuthenReview-Prototype.git
   cd AuthenReview-Prototype
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set your Flask secret key** (optional)

   ```bash
   export FLASK_SECRET_KEY=mysecret   # macOS/Linux
   set FLASK_SECRET_KEY=mysecret      # Windows CMD
   ```

## Usage

### 1. Running the web app

Start the Flask server:

```bash
python app.py
```

Open your browser to `http://localhost:5000`. You can:

* **Upload a CSV** with columns:

  * `review_text` (string)
  * `label` (0 or 1, optional)
* **Or paste raw reviews:** One review per line into the textarea.

The interface will display:

1. **Model Performance** (if `label` column provided)

   * Accuracy, Precision, Recall, F1 Score
2. **Prediction Breakdown**

   * Real vs. Fake counts & percentages
3. **Sample Predictions**

   * Table of review text and predicted label badges (Real 🚩, Fake ❌)

### 2. Training the model

If you want to retrain or update the model:

```bash
python train.py
```

This will:

* Load `uploads/cleaned_formatted_reviews.csv` (must contain `review_text` and `label` columns with labels `0` or `1`).
* Train a `TextCleaner` + `TfidfVectorizer` + `RandomForestClassifier` pipeline.
* Perform 5-fold GridSearchCV (n\_estimators: 100,200; max\_depth: None,20).
* Serialize the best pipeline to `model_pipeline.pkl`.

## Model Auto-Download

On first run, if `model_pipeline.pkl` is missing, `app.py` will automatically download it from the [v1.0 GitHub release](https://github.com/Russelrip/AuthenReview-Prototype/releases/tag/v1.0).
