import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from preprocessing import TextCleaner

# 1. Load & select only the two columns we need
df = pd.read_csv('uploads/fakereviewsdataset.csv', usecols=['text_', 'label'])
df = df.rename(columns={'text_': 'review_text'})

# 2. Keep only OR/CG and map to 0/1
df = df[df['label'].isin(['OR', 'CG'])]
df['label'] = df['label'].map({'OR': 0, 'CG': 1})

# 3. Split into X and y
X = df['review_text']
y = df['label'].astype(int)


# 4. Build pipeline (unchanged)
template_pipe = Pipeline([
    ('clean', TextCleaner()),
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=5
    )),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# 5. Hyperparameter tuning (unchanged)
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 20]
}
grid = GridSearchCV(
    template_pipe,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid.fit(X, y)

# 6. Cross-validate
scores = cross_validate(
    grid.best_estimator_, X, y,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    cv=5
)
print({
    m: scores[f'test_{m}'].mean()
    for m in ['accuracy', 'precision', 'recall', 'f1']
})

# 7. Serialize pipeline
with open('model_pipeline.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)
print("✅ model_pipeline.pkl saved")
