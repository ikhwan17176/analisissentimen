
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import streamlit as st
from tqdm import trange
hasil = pd.read_excel("hasil_ulasan.xlsx")
import nltk
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
download_nltk_data()
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r'\\u[0-9A-Fa-f]{4}', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text
vectorizer = TfidfVectorizer()
hasil = hasil.dropna(subset=['preprocessed_content'])
X = vectorizer.fit_transform(hasil['preprocessed_content'])
y = hasil['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_smote, y_train_smote)
y_pred_smote = model.predict(X_test)

# Streamlit App
st.title("Sentiment Analysis on Shopee Reviews")

menu = ["EDA", "Sentiment Analysis"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == 'EDA':
    st.subheader("Exploratory Data Analysis")
    jml_label = hasil['label'].value_counts()
    st.bar_chart(jml_label)
    wc = " ".join(hasil['preprocessed_content'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wc)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
elif choice == 'Sentiment Analysis':
    st.subheader("Sentiment Analysis")
    
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred_smote))
    
    cd = confusion_matrix(y_test, y_pred_smote)
    dispay = ConfusionMatrixDisplay(confusion_matrix=cd, display_labels=['Negative', 'Positive'])
    dispay.plot(cmap=plt.cm.Blues)
    st.pyplot(plt)
    
    st.subheader("Predict Sentiment")
    input_text = st.text_area("Enter your review text here")
    if st.button("Predict"):
        input_text_preprocessed = preprocess_text(input_text)
        input_vectorized = vectorizer.transform([input_text_preprocessed])
        prediction = model.predict(input_vectorized)[0]
        st.write(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")

