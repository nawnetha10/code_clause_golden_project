import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv('personpred.csv')  # Replace with your dataset

# Preprocess the text
data['cleaned_cv_text'] = data['cv_text'].apply(lambda x: x.lower())

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['cleaned_cv_text'])
y = data['personality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save the trained model and vectorizer
joblib.dump(clf, 'trained_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Display the column names in the DataFrame


