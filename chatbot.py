import os
import json
import datetime
import csv
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load intents from the JSON file
with open("intents.json") as file:
    intents = json.load(file)

# Load the vectorizer and model
if os.path.exists("vectorizer.joblib") and os.path.exists("clf.joblib"):
    vectorizer = joblib.load("vectorizer.joblib")
    clf = joblib.load("clf.joblib")
else:
    st.error("Model files not found! Please ensure 'vectorizer.joblib' and 'clf.joblib' are available.")
    st.stop()

# Function for chatbot response based on input text
def chatbot(input_text):
    input_text_vectorized = vectorizer.transform([input_text])
    tag = clf.predict(input_text_vectorized)[0]
    
    # Search for the intent matching the tag
    for intent in intents:
        if intent.get('tag') == tag:  # Using 'get' to avoid KeyError
            response = random.choice(intent['responses']) if intent['responses'] else "Sorry, I didn't understand that."
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

    # User input for chatbot
    user_input = st.text_input("You:")

    if user_input:
        # Get the chatbot's response
        response = chatbot(user_input)
        
        # Display chatbot's response in a text area
        st.text_area("Chatbot:", value=response, height=120)

        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save the user input and chatbot response to chat_log.csv
        try:
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])
        except Exception as e:
            st.error(f"Error saving chat log: {e}")

        # Check for goodbye message
        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
