import joblib
import random
import json
import streamlit as st

# Load intents from the JSON file
with open("intents.json") as file:
    intents = json.load(file)

# Load the trained vectorizer and classifier
vectorizer = joblib.load("vectorizer.joblib")
clf = joblib.load("clf.joblib")

# Function to handle chatbot response
def chatbot(input_text):
    # Transform input_text using the vectorizer and predict the tag using the classifier
    input_text_vectorized = vectorizer.transform([input_text])
    tag = clf.predict(input_text_vectorized)[0]
    
    # Find the response based on the predicted tag
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses']) if intent['responses'] else "Sorry, I didn't understand that."
            return response

# Streamlit interface
def main():
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")
    
    counter = 0  # Initialize the counter for user input and responses
    
    user_input = st.text_input("You:", key=f"user_input_{counter}")
    
    if user_input:
        # Get the chatbot's response
        response = chatbot(user_input)
        
        # Display the chatbot's response
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")
        
        # Increment the counter for new inputs
        counter += 1
        
        # Handle exit response (goodbye or bye)
        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

# Run the Streamlit app
if __name__ == '__main__':
    main()
