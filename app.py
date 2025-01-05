    
 # This part of code can be in a separate file for training
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents from JSON
with open("intents.json") as file:
    intents = json.load(file)

# Prepare data for training
patterns = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Classifier
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(X, tags)

# Save the model and vectorizer
joblib.dump(vectorizer, "vectorizer.joblib")
joblib.dump(clf, "clf.joblib")