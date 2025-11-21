import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# importing stopwords if missing
# try:
#     nltk.data.find("corpora/stopwords")
# except LookupError:
#     print("stopwords not found , downloading now")
#     nltk.download("stopwords",quiet=True)
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

ps = PorterStemmer()


def clean_text(text):

    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

def data_preprocessing(input_csv, output_dir):
    # Load Data
    columns = ["target",'ID',"date_time","query","username","text"]
    df = pd.read_csv(input_csv,names=columns,encoding="ISO-8859-1")

    # Change target column values from 4-->1
    #  Negative: 0, Positive: 1
    df.loc[df["target"] == 4, "target"] = 1

    # calling the fucntion to clean the text 
    df['cleaned_text'] = df['text'].apply(clean_text)

    # data cleaning 
    df=df[['cleaned_text','target']]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Save cleaned data to output dir
    df.to_csv(output_dir)
    print("Data preprocessing complete. Files saved in:", output_dir)


input_csv = r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\data\raw\tweets.csv"
output_dir = r"C:\Users\Swapnil Memane\Desktop\Twitter Sentiment\data\processed\cleaned_tweets.csv"


data_preprocessing(input_csv, output_dir)
