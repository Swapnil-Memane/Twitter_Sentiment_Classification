import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
print("check train_log_reg import")
from src.train_log_reg import load_model
print("checked train_log_reg import")
print("check clean_text import")
from src.preprocessing import clean_text
print("checked clean_text import")

st.title("Twitter Sentiment Analyzer")

model, vectorizer = load_model()

tweet = st.text_input("Enter a tweet")

if st.button("Predict"):
    
    if tweet == "":
           st.error("Enter a text to check sentiment")
    else:   
        cleaned = clean_text(tweet)
        vectorized = vectorizer.transform([cleaned])
        pred = model.predict(vectorized)[0]
        
        if pred == 0:
                st.error("Negative Sentiment")
        else:
                st.success("Positive Sentiment")
    
    
