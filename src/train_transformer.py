import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

def train_transformer_model(
        CLEAN_DATA_PATH,
        model_save_path=r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\models\transformer_model.keras",
        tokenizer_save_path=r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\models\transformer_tokenizer.pkl"
    ):
    df = pd.read_csv(CLEAN_DATA_PATH)

    # Drop missing values from both columns together
    df = df.dropna(subset=["cleaned_text", "target"])
    # Take 8% sample safely
    subset = df.sample(frac=0.08,random_state=42,axis=0)

    # Extract text and labels together
    X_transformer = subset["cleaned_text"].astype(str).tolist()
    y_transformer = subset["target"].tolist()

    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X_transformer, y_transformer, test_size=0.2, random_state=42, stratify=y_transformer)

    # # Convert all to strings safely and drop weird cases
    # X_train_tf = [str(x) for x in X_train_tf if isinstance(x, (str, int, float))]
    # X_test_tf  = [str(x) for x in X_test_tf if isinstance(x, (str, int, float))]

    # # Remove obvious placeholders
    # X_train_tf = [x for x in X_train_tf if x.strip().lower() not in ["nan", "none", "null", ""]]
    # X_test_tf  = [x for x in X_test_tf if x.strip().lower() not in ["nan", "none", "null", ""]]

    # Tokenize and re-create datasets to reflect the reduced sizes
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(X_train_tf, truncation=True, padding=True, max_length=64)
    test_encodings  = tokenizer(X_test_tf, truncation=True, padding=True, max_length=64)

    # Create datasets again with the reduced and re-tokenized data
    train_encodings= dict(train_encodings)
    test_encodings= dict(test_encodings)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_encodings, y_train_tf)).shuffle(10000).batch(16)
    test_dataset  = tf.data.Dataset.from_tensor_slices((test_encodings, y_test_tf)).batch(16)

    # compile model
    MODEL_NAME = "distilbert-base-uncased"
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, from_pt=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=3e-5, weight_decay=0.01)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train model
    model.fit(train_dataset, validation_data=test_dataset, epochs=3)

    # Model Evaluation
    # Make predictions on the test set
    predictions = model.predict(test_dataset)

    # Extract predicted labels (logits) and convert them to class labels
    y_pred_logits = predictions.logits
    y_pred_transformer = np.argmax(y_pred_logits, axis=1)

    # Collect true labels from the test_dataset by iterating through it
    y_true_transformer = []
    for _, labels in test_dataset:
        y_true_transformer.extend(labels.numpy())
    y_true_transformer = np.array(y_true_transformer)

    # Generate and save the classification report
    report =  classification_report(y_true_transformer, y_pred_transformer)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\data\evaluation\transformer_clf_report.csv")

    # Define the directory to save the transformer model and tokenizer
    save_directory = r"C:/Users/Swapnil Memane/Desktop/Twitter Sentiment/models/transformer/"

    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Save the tokenizer
    tokenizer.save_pretrained(save_directory)
    print(f"Tokenizer saved to {save_directory}")

    # Save the model
    model.save_pretrained(save_directory)
    print(f"Transformer model saved to {save_directory}")


def load_transformer_model(save_directory="C:/Users/Swapnil Memane/Desktop/Twitter Sentiment/models/transformer/"):
    import os
    from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

    # Load the tokenizer
    loaded_tokenizer = DistilBertTokenizerFast.from_pretrained(save_directory)
    print("Tokenizer loaded successfully!")

    # Load the model
    loaded_model_transformer = TFDistilBertForSequenceClassification.from_pretrained(save_directory)
    print("Transformer model loaded successfully!")
    return loaded_tokenizer, loaded_model_transformer

def predict_sentiment_transformer(text):
    #Load saved model
    SAVE_DIR = r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\models\transformer"
    loaded_tokenizer, loaded_model_transformer = load_transformer_model(SAVE_DIR)

    # Tokenize the input text
    inputs = loaded_tokenizer(text, truncation=True, padding=True, return_tensors="tf")

    # Make prediction
    predictions = loaded_model_transformer(inputs)
    logits = predictions.logits

    # Get probabilities and predicted class
    probabilities = tf.nn.softmax(logits, axis=-1)
    predicted_class = tf.argmax(probabilities, axis=1).numpy()[0]

    return predicted_class

if __name__ == "__main__": # Prevents auto-running code when imported
    CLEAN_DATA_PATH = r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\data\processed\cleaned_tweets.csv"
    start_time = time.time()
    train_transformer_model(CLEAN_DATA_PATH)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total Training Time: {training_time:.2f} seconds")
