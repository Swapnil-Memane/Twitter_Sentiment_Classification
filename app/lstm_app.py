import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.train_lstm import load_lstm_model
from src.train_lstm import predict_sentiment

print("importing cleaned text")
from src.preprocessing import clean_text
print("imported cleaned text")

st.write("App started loading...")

st.title("Twitter Sentiment Analyzer")

try:
    model, tokenizer, MAX_LEN = load_lstm_model()
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")


tweet = st.text_input("Enter a tweet")

if st.button("Predict"):
    
    if tweet == "":
           st.error("Enter a text to check sentiment")
    else:   
        cleaned = clean_text(tweet)
        label, prob = predict_sentiment(cleaned,model,tokenizer,MAX_LEN)
    
        if label == 0:
                st.error("Negative Sentiment")
        else:
                st.success("Positive Sentiment")


    
