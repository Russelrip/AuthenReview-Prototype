from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk

nltk.download('stopwords')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model_path = 'model.pkl'
tfidf_path = 'tfidf.pkl'

def preprocess_text(text):
    import re
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def train_and_save_model(df):
    X = df['review_text'].apply(preprocess_text)
    y = df['label']
    
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(tfidf_path, 'wb') as f:
        pickle.dump(tfidf, f)
    
    return clf, tfidf, X_test, y_test, df.iloc[y_test.index]['review_text']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        if 'review_text' not in df.columns or 'label' not in df.columns:
            return "Error: CSV must contain 'review_text' and 'label' columns."
        
        if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
            clf, tfidf, X_test, y_test, reviews = train_and_save_model(df)
        else:
            with open(model_path, 'rb') as f:
                clf = pickle.load(f)
            with open(tfidf_path, 'rb') as f:
                tfidf = pickle.load(f)
            
            X = df['review_text'].apply(preprocess_text)
            X_tfidf = tfidf.transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['label'], test_size=0.2, random_state=42)
            reviews = df.iloc[y_test.index]['review_text']
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results = pd.DataFrame({
            'Review': reviews,
            'True_Label': y_test.values,
            'Predicted_Label': y_pred,
            'Accuracy': [1 if pred == true else 0 for pred, true in zip(y_pred, y_test)]
        })
        
        results.to_csv('static/review_accuracy.csv', index=False)
        return render_template('index.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1, results=results.to_dict(orient='records'))
    
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
