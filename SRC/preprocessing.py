import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    data=pd.read_csv("D:/project-2/DATA/e-mail.csv")
    x=data["email_text"]
    y=data["label"]

    vectorizer=CountVectorizer()
    x_vectorized=vectorizer.fit_transform(x)

    x_train,x_test,y_train,y_test=train_test_split(x_vectorized,y,test_size=0.2,random_state=42)

    return x_train,x_test,y_train,y_test
