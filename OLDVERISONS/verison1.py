import numpy as np
import pandas as pd
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import joblib

mainpathdir = Path('/Users/blakeweiss/Desktop/disinformationclassifier')

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  #URLs
    text = re.sub(r'\W', ' ', text)      #special characters
    text = re.sub(r'\s+', ' ', text)     #multiple spaces to a single space
    return text.strip()

def train_model(data_path):
    #load and preprocess the data
    data = pd.read_csv(data_path)
    data['text'] = data['text'].apply(clean_text)
    data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)
    data = data.drop('label', axis=1)
    
    #sentiment 
    data['polarity'] = data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    data['subjectivity'] = data['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    X = data[['text', 'polarity', 'subjectivity']]
    y = data['fake']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2), min_df=3)
    X_train_vectorized = vectorizer.fit_transform(X_train['text'])

    #classifier and hyperparameters 
    param_grid = {'svc__C': [0.1, 1, 10]}
    clf1 = MultinomialNB()
    clf2 = LinearSVC(max_iter=5000)
    ensemble_clf = VotingClassifier(estimators=[
        ('nb', clf1),
        ('svc', clf2)
    ], voting='hard')
    
    grid_search = GridSearchCV(ensemble_clf, param_grid, cv=5)
    grid_search.fit(X_train_vectorized, y_train)
    print("Best Parameters:", grid_search.best_params_)

    #save the model and vectorizer
    joblib.dump(grid_search.best_estimator_, 'ensemble_clf.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    X_test_vectorized = vectorizer.transform(X_test['text'])
    y_pred = grid_search.predict(X_test_vectorized)
    print("Accuracy on training split:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def evaluate_new_data(data_path):
    #load the model and vectorizer that was already trained
    ensemble_clf = joblib.load('ensemble_clf.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    #load new data and preprocess
    new_data = pd.read_csv(data_path)
    new_data['text'] = new_data['text'].apply(clean_text)
    new_data['polarity'] = new_data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    new_data['subjectivity'] = new_data['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    if 'label' in new_data.columns:
        new_data['fake'] = new_data['label'].apply(lambda x: 0 if x == 'REAL' else 1)
        y_true = new_data['fake']
    else:
        y_true = None

    new_data_vectorized = vectorizer.transform(new_data['text'])

    #predict the new data and classify it
    predictions = ensemble_clf.predict(new_data_vectorized)

    if y_true is not None:
        print("Accuracy on new data:", accuracy_score(y_true, predictions))
        print(classification_report(y_true, predictions))

    return predictions




train_model(mainpathdir / 'fake_or_real_news.csv')
new_predictions = evaluate_new_data(mainpathdir / 'fake_or_real_news.csv')
print("Predictions for new data:", new_predictions)
