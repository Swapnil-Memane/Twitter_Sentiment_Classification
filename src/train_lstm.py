import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
# from src.config import CLEAN_DATA_PATH
from sklearn.model_selection import train_test_split
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import time

def train_lstm_model(
        CLEAN_DATA_PATH,
        model_save_path=r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\models\lstm_model.keras",
        tokenizer_save_path=r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\models\lstm_tokenizer.pkl"
    ):
    
    twitter_data = pd.read_csv(CLEAN_DATA_PATH)
    print("Data is loaded")

    twitter_data = twitter_data.dropna(subset=["cleaned_text","target"])

    X = twitter_data["cleaned_text"].values
    y = twitter_data["target"].values

    X_train_lstm, X_test_lstm, y_train, y_test = train_test_split(X,y,stratify=y,random_state=42,test_size=0.2)

    vocab_size = 5000
    max_len = 40

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

    # Fill NaN values with empty strings before vectorization
    X_train_lstm = np.array([str(x) if pd.notna(x) else '' for x in X_train_lstm])
    X_test_lstm = np.array([str(x) if pd.notna(x) else '' for x in X_test_lstm])

    tokenizer.fit_on_texts(X_train_lstm)

    X_train_seq = tokenizer.texts_to_sequences(X_train_lstm)
    X_test_seq  = tokenizer.texts_to_sequences(X_test_lstm)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad  = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    print("Tokenization and padding done")

    model = Sequential([
        Embedding(vocab_size, 64, input_length=max_len),
        LSTM(32, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])

    print("Model is built")

    model.compile(
        loss="binary_crossentropy",
        optimizer="adamw",
        metrics=["accuracy"],
    )

    # Early stopping to avoid overfitting
    es = EarlyStopping(
        patience=2,
        restore_best_weights=True,
        monitor="val_loss"
    )

    print("Training started...\n")

    model.fit(
        X_train_pad, y_train,
        validation_data=(X_test_pad, y_test),
        epochs=10,
        batch_size=256,   # much faster
        callbacks=[es],
        verbose=1
    )

    print("Model is trained")

    # Model Evaluation
    y_pred_probs = model.predict(X_test_pad)
    y_pred = (y_pred_probs > 0.5).astype(int)   # Convert to class labels
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\data\evaluation\lstm_clf_report.csv")

    # Save Model
    model.save(model_save_path) # Val accuracy achieved 78.06%
    joblib.dump(tokenizer, tokenizer_save_path)

    print(f"\n Model saved at: {model_save_path}")
    print(f" Tokenizer saved at: {tokenizer_save_path}")


def load_lstm_model(
        model_path=r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\models\lstm_model.keras",
        tokenizer_path=r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\models\lstm_tokenizer.pkl",
    ):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print("Loading LSTM model...")
    model = load_model(model_path)

    print("Loading tokenizer...")
    tokenizer = joblib.load(tokenizer_path)

    MAX_LEN = 40 # Same as used for model training

    print("Model, tokenizer config loaded successfully!")

    return model, tokenizer, MAX_LEN 

def predict_sentiment(input_text,model, tokenizer,MAX_LEN=40):  
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_seq,maxlen=MAX_LEN,padding="post",truncating="post")

    prob = float(model.predict(input_padded)[0][0])
    label = 1 if prob> 0.5 else 0 
    
    return label, prob


if __name__ == "__main__": # Prevents auto-running code when imported
    CLEAN_DATA_PATH = r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\data\processed\cleaned_tweets.csv"
    start_time = time.time()
    train_lstm_model(CLEAN_DATA_PATH)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total Training Time: {training_time:.2f} seconds")