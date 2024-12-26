import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Data Collection & Pre-Processing
raw_mail_data = pd.read_csv(r'C:/Users/91918/Downloads/mail_data.csv')  # Use your actual file path
print(raw_mail_data)

# Replace the null values with an empty string
mail_data = raw_mail_data.fillna('')

# Label Encoding
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

X = mail_data['Message']
Y = mail_data['Category']

# Splitting the data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature Extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert Y_train and Y_test values to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Save the model and vectorizer
with open('spam_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(feature_extraction, vectorizer_file)

# Evaluating the trained model
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print(f'Accuracy on training data: {accuracy_on_training_data}')

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print(f'Accuracy on test data: {accuracy_on_test_data}')
