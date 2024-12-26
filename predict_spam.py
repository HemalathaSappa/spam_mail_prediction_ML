import pickle

# Load the trained model and vectorizer
with open('spam_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    feature_extraction = pickle.load(vectorizer_file)

# Making a Prediction
input_mail = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfill my promise. You have been wonderful and a blessing at all times."]
input_data_features = feature_extraction.transform(input_mail)

# Make prediction
prediction = model.predict(input_data_features)

# Output the result
result = 'Ham' if prediction[0] == 1 else 'Spam'
print(f'Prediction: {result} mail')
