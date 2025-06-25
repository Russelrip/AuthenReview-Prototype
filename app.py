import os
import pickle
import requests

import pandas as pd
from flask import Flask, flash, redirect, render_template, request

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)

app = Flask(__name__)

app.config.update(
    UPLOAD_FOLDER='uploads',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    SECRET_KEY=os.getenv('FLASK_SECRET_KEY', 'change_me')
)

# URL to your GitHub release asset
MODEL_URL = (
    "https://github.com/Russelrip/AuthenReview-Prototype/"
    "releases/download/v1.0/model_pipeline.pkl"
)
MODEL_PATH = "model_pipeline.pkl"

# Download the model if not present locally
if not os.path.exists(MODEL_PATH):
    resp = requests.get(MODEL_URL, stream=True)
    resp.raise_for_status()
    with open(MODEL_PATH, "wb") as fd:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fd.write(chunk)

# Load trained pipeline
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded = request.files.get('file')
        if not uploaded or not uploaded.filename.lower().endswith('.csv'):
            flash('Please upload a .csv file')
            return redirect(request.url)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded.filename)
        uploaded.save(path)

        df = pd.read_csv(path)
        if 'review_text' not in df or 'label' not in df:
            flash("CSV must contain 'review_text' and 'label' columns")
            return redirect(request.url)

        preds = model.predict(df['review_text'].astype(str))
        df['predicted'] = preds

        y_true = df['label']
        metrics = {
            'accuracy': accuracy_score(y_true, preds),
            'precision': precision_score(y_true, preds, zero_division=0),
            'recall': recall_score(y_true, preds, zero_division=0),
            'f1_score': f1_score(y_true, preds, zero_division=0),
        }

        return render_template(
            'index.html',
            metrics=metrics,
            results=df.to_dict(orient='records')
        )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
