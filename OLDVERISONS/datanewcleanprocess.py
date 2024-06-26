import pandas as pd
import numpy as np
from pathlib import Path
import re
from textblob import TextBlob

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

def process_dataset(data_path):
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

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)      # Remove special characters
    text = re.sub(r'\s+', ' ', text)     # Convert multiple spaces to a single space
    return text.strip()

