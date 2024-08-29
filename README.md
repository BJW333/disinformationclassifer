

Disinformation Classifier
---------
This project is a machine learning-based disinformation classifier that identifies whether a piece of news is REAL or FAKE. The classifier uses an ensemble of Naive Bayes and SVM models to predict the authenticity of text data.

This is a program I wrote that is a classifier to identify misinformation it can be trained on more data easily and to determine what's real information and news and what's fake when you give the model a new never before seen dataset. 
The program can be easily improved upon and is expandable.

Directory Structure
---------
	•	/Users/blakeweiss/Desktop/disinformationclassifier/
	•	fake_or_real_news.csv: The dataset containing news articles labeled as REAL or FAKE.
	•	ensemble_clf.pkl: The trained ensemble classifier model.
	•	vectorizer.pkl: The TF-IDF vectorizer used to process text data.
	•	main.py: The main Python script containing the model training, evaluation, and ROC curve plotting functions.

Requirements
---------
To run this project, you need the following Python libraries:

	•	numpy
	•	pandas
	•	re
	•	textblob
	•	scikit-learn
	•	joblib
	•	matplotlib

You can install the required packages using pip:

pip3.10 install numpy pandas textblob scikit-learn joblib matplotlib

Usage
---------
1. Training the Model

The train_model() function loads and preprocesses the training data, then trains an ensemble classifier using Naive Bayes and SVM. It uses grid search to find the best hyperparameters for the SVM model.

The model and vectorizer are saved as ensemble_clf.pkl and vectorizer.pkl in the main directory.

To train the model, run:

train_model(mainpathdir / 'fake_or_real_news.csv')

2. Evaluating New Data

The evaluate_new_data() function processes new text data, predicts whether it is REAL or FAKE, and optionally computes accuracy if labels are available. It loads the saved model and vectorizer for predictions.

To evaluate new data, run:

new_predictions = evaluate_new_data(mainpathdir / 'fake_or_real_news.csv')

3. Plotting the ROC Curve

The plot_roc_curve() function plots the ROC curve to evaluate the performance of the classifier.

To plot the ROC curve, run:

plot_roc_curve(y_true, new_predictions, 'ROC Curve')

4. Cleaning Text

The clean_text() function cleans the text by removing URLs, special characters, and multiple spaces.

5. Processing New Data

The newdataprocessing() function detects text columns in the data, cleans them, and computes polarity and subjectivity scores using the TextBlob library.

Example Run
---------
train_model(mainpathdir / 'fake_or_real_news.csv')
new_predictions = evaluate_new_data(mainpathdir / 'fake_or_real_news.csv')
plot_roc_curve(y_true, new_predictions, 'ROC Curve')
print("Predictions for new data:", new_predictions)

Files
---------
	•	fake_or_real_news.csv: Sample dataset used for training and evaluation.
	•	ensemble_clf.pkl: Trained ensemble model.
	•	vectorizer.pkl: TF-IDF vectorizer used for text data processing.
