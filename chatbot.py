import os
import json
import datetime
import csv
import random
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Setting up SSL context for NLTK data download
ssl._create_default_https_context = ssl._create_unverified_context

# Load intents from the JSON file
with open("intents.json") as file:
    intents = json.load(file)

# Load the vectorizer and model
if os.path.exists("vectorizer.joblib") and os.path.exists("clf.joblib"):
    vectorizer = joblib.load("vectorizer.joblib")
    clf = joblib.load("clf.joblib")
else:
    st.error("Model files not found!")
    st.stop()

def chatbot(input_text):
    input_text_vectorized = vectorizer.transform([input_text])
    tag = clf.predict(input_text_vectorized)[0]
    
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Main application logic
def main():
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    # Check if chat_log.csv exists; create if it does not
    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

    user_input = st.text_input("You:")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=120, max_chars=None)
        
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

        # Save the user input and chatbot response to chat_log.csv
        with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([user_input, response, timestamp])

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
    
