import pickle
from flask import Flask, render_template, request

# Load the trained model and vectorizer
with open('spam_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Initialize Flask application
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html', result=None)

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    input_mail = request.form.get('email', '').strip()  # Get email and strip any extra spaces

    # Check if the email input is empty
    if not input_mail:
        return render_template('index.html', result="Error: Please provide the email text.", email="")

    try:
        # Transform the input text into features
        input_features = tfidf_vectorizer.transform([input_mail])

        # Predict whether the email is spam or ham
        prediction = model.predict(input_features)

        # Show the result (Ham or Spam)
        result = 'Ham' if prediction[0] == 1 else 'Spam'

        return render_template('index.html', result=result, email=input_mail)
    
    except Exception as e:
        # Catch any errors that occur during prediction
        return render_template('index.html', result=f"Error: {str(e)}", email=input_mail)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
# 