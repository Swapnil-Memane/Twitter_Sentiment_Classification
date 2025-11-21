import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# from src.config import MODEL_PATH, VECTORIZER_PATH

def train_log_reg_model(
             CLEAN_DATA_PATH,
             model_save_path=r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\models\trained_log_reg.sav",
             vectorizer_save_path=r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\models\vectorizer.pkl"
                      ):

    twitter_data = pd.read_csv(CLEAN_DATA_PATH)
    print("Data is loaded")

    twitter_data = twitter_data.dropna(subset=["cleaned_text","target"])

    X = twitter_data["cleaned_text"].values
    y = twitter_data["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=42,test_size=0.2)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    print("Model training started")
    model.fit(X_train_vec, y_train)
    print("Model training completed")

    #Save Model
    pickle.dump(model, open(model_save_path, "wb"))
    pickle.dump(vectorizer, open(vectorizer_save_path, "wb"))

    # Model Evaluation
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\data\evaluation\log_reg_clf_report.csv")


    return model, vectorizer

def load_model(MODEL_PATH= r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\models\trained_log_reg.sav", 
              VECTORIZER_PATH= r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\models\vectorizer.pkl" ):
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
    return model, vectorizer

if __name__ == "__main__": # Prevents auto-running code when imported
    CLEAN_DATA_PATH = r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\data\processed\cleaned_tweets.csv"
    start_time = time.time()
    train_log_reg_model(CLEAN_DATA_PATH)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total Training Time: {training_time:.2f} seconds")