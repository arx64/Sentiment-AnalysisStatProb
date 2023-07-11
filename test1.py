import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

# Load the data
data = pd.read_csv("data_traveloka_fix.csv", sep='|')

# Preprocess the data
cv = CountVectorizer()
X = cv.fit_transform(data["tweet"])
y = data["label"]

# Convert the sparse matrix to a dense matrix
X = X.toarray()

# Create the SVM, Naive Bayes, and Logistic Regression models
svm_model = SVC()
nb_model = GaussianNB()
lr_model = LogisticRegression()

# Train the models
svm_model.fit(X, y)
nb_model.fit(X, y)
lr_model.fit(X, y)

# Evaluate the models
svm_predictions = svm_model.predict(X)
nb_predictions = nb_model.predict(X)
lr_predictions = lr_model.predict(X)
y_score = svm_model.predict_proba(X)[:, 1]
y_score = y_score.astype('float32')

# Print the results
print("SVM Precision:", precision_score(y, svm_predictions, pos_label='positive'))
print("SVM Recall:", recall_score(y, svm_predictions, pos_label='positive'))
print("SVM F1 Score:", f1_score(y, svm_predictions, pos_label='positive'))
print("SVM Average Precision Score:", average_precision_score(y, y_score, pos_label='positive'))


print("Naive Bayes Precision:", precision_score(y, nb_predictions))
print("Naive Bayes Recall:", recall_score(y, nb_predictions))
print("Naive Bayes F1 Score:", f1_score(y, nb_predictions))
print("Naive Bayes Average Precision Score:", average_precision_score(y, nb_predictions))

print("Logistic Regression Precision:", precision_score(y, lr_predictions))
print("Logistic Regression Recall:", recall_score(y, lr_predictions))
print("Logistic Regression F1 Score:", f1_score(y, lr_predictions))
print("Logistic Regression Average Precision Score:", average_precision_score(y, lr_predictions))
