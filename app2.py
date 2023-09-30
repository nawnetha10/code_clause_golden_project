import os
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
clf = joblib.load('trained_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def preprocess_cv_text(cv_text):
    cleaned_cv_text = cv_text.lower()
    return [cleaned_cv_text]

def predict_personality(cv_text):
    cv_text_cleaned = preprocess_cv_text(cv_text)
    cv_features = vectorizer.transform(cv_text_cleaned)
    predicted_personality = clf.predict(cv_features)
    return predicted_personality[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_personality = None

    if request.method == 'POST':
        cv_file = request.files['cv_file']
        if cv_file and cv_file.filename.endswith('.txt'):
            cv_text = cv_file.read().decode('utf-8')
            predicted_personality = predict_personality(cv_text)

    return render_template('index.html', predicted_personality=predicted_personality)

if __name__ == '__main__':
    app.run(debug=True)
