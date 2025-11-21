print(" SCRIPT STARTED")      # <--- add this at very top
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Importing train_lstm...")
from src.train_lstm import load_lstm_model, predict_sentiment
print("Import successful")
print("Importing clean_text")
from src.preprocessing import clean_text
print("clean_text import successful")

try:
    print("Trying to load saved model")
    model, tokenizer, MAX_LEN = load_lstm_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model loading failed: {e}")
    exit()

tweet = input("Enter a tweet: ").strip()

if tweet == "":
    print("Enter correct text")
else:
    cleaned = clean_text(tweet)
    label, prob = predict_sentiment(cleaned, model, tokenizer, MAX_LEN)

    if label == 0:
        print(f"Negative Sentiment ({prob:.2f})")
    else:
        print(f"Positive Sentiment ({prob:.2f})")
