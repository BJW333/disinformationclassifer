import numpy as np
import pandas as pd
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

mainpathdir = Path('/Users/blakeweiss/Desktop/disinformationclassifier')


#CLEANING TEXT
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  #URLs
    text = re.sub(r'\W', ' ', text)      #special characters
    text = re.sub(r'\s+', ' ', text)     #multiple spaces to a single space
    return text.strip()

#NEWDATAPROCESSING not used for training cleaning too lazy to intergrate rn
def newdataprocessing(data_path):
    def is_text_column(column):
        #algo to detect if 50 or more is text in a collom them thats the text collum
        if isinstance(column.dropna(), pd.Series):
            text_like = column.dropna().apply(lambda x: isinstance(x, str) and len(x) > 50)
            if text_like.mean() > 0.8:
                return True
        return False

    def load_data(file_path):
        ext = Path(file_path).suffix
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        elif ext == '.json':
            return pd.read_json(file_path)
        elif ext == '.txt':
            #assume each line in the text file is seperate piece of data
            return pd.read_csv(file_path, sep='\n', header=None, names=['text'])
        else:
            raise ValueError(f"Unsupported file format: {ext}")


    data = load_data(data_path)

    #detect text columns
    text_columns = [col for col in data.columns if is_text_column(data[col])]
    if not text_columns:
        raise ValueError("No text columns detected in the dataset. Please check the dataset format.")

    #process
    for col in text_columns:
        data['processed_' + col] = data[col].apply(clean_text)
        data['polarity_' + col] = data['processed_' + col].apply(lambda x: TextBlob(x).sentiment.polarity)
        data['subjectivity_' + col] = data['processed_' + col].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    return data

#TRAINING DATA MODEL
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
    joblib.dump(grid_search.best_estimator_, (mainpathdir / 'ensemble_clf.pkl'))
    joblib.dump(vectorizer, (mainpathdir / 'vectorizer.pkl'))

    X_test_vectorized = vectorizer.transform(X_test['text'])
    y_pred = grid_search.predict(X_test_vectorized)
    print("Accuracy on training split:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

#NEWDATA MODEL
def evaluate_new_data(data_path):
    global y_true
    #load the model and vectorizer that was already trained
    ensemble_clf = joblib.load(mainpathdir / 'ensemble_clf.pkl')
    vectorizer = joblib.load(mainpathdir / 'vectorizer.pkl')

    #load new data and preprocess
    new_data = newdataprocessing(data_path)
    
    
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


def plot_roc_curve(y_true, predictions, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(y_true, predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

plt.figure()


train_model(mainpathdir / 'fake_or_real_news.csv')

new_predictions = evaluate_new_data(mainpathdir / 'fake_or_real_news.csv')
plot_roc_curve(y_true, new_predictions, 'ROC Curve')

print("Predictions for new data:", new_predictions)
