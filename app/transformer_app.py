import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.train_transformer import load_transformer_model, predict_sentiment_transformer

print("importing cleaned text")
from src.preprocessing import clean_text
print("imported cleaned text")

st.write("App started loading...")

st.title("Twitter Sentiment Analyzer")


tweet = st.text_input("Enter a tweet")

if st.button("Predict"):
    
    if tweet == "":
           st.error("Enter a text to check sentiment")
    else:   
        cleaned = clean_text(tweet)
        label = predict_sentiment_transformer(cleaned)
    
        if label == 0:
                st.error("Negative Sentiment")
        else:
                st.success("Positive Sentiment")


    
