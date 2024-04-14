import streamlit as st
import joblib
import nltk
from transformers import BertTokenizer, BertModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[:;=x][\-o\*]?[\)\(\[\]dpo\@\>\<\}3]', '', text)
    text = re.sub(r'#[\w-]+', 'hashtag', text)
    text = re.sub(r'@[\w-]+', 'mention', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# Load the trained SVM model
model = joblib.load('svm_model.pkl')

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()

def predict_sentiment(embeddings):
    prediction = model.predict(embeddings)
    sentiments = {0: ('Negative', '😢'), 1: ('Neutral', '😐'), 2: ('Positive', '😊')}
    return sentiments[prediction[0]]

# Streamlit UI setup
st.title("Stock News Headline Sentiment Analysis")
headline = st.text_input("Enter a news headline:")
if st.button("Analyze Sentiment"):
    if headline:
        processed_headline = preprocess_text(headline)
        embeddings = get_bert_embeddings(processed_headline)
        sentiment, emoji = predict_sentiment(embeddings)
        st.write(f"The sentiment of the headline is: {sentiment}")
        st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{emoji}</h1>", unsafe_allow_html=True)
    else:
        st.write("Please enter a headline to analyze.")
