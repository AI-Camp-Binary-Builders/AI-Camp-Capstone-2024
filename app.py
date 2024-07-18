from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import joblib
import os

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)


# Function to load model and vectorizer
def load_model(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


# Function to scrape article content from URL
def scrape_article(url):
    try:
        headers = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        print(response,"%%%%%%%")
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Try multiple methods to extract content
        content = ""

        # Method 1: Look for common article containers
        for container in [
                'article', 'main', 'div[class*="content"]',
                'div[class*="article"]', 'div[id*="content"]',
                'div[id*="article"]'
        ]:
            if content:
                break
            elements = soup.select(container)
            if elements:
                content = elements[0].get_text(strip=True, separator=' ')

        # Method 2: If still empty, get all paragraph text
        if not content:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])

        # Method 3: If still empty, get all div text
        if not content:
            divs = soup.find_all('div')
            content = ' '.join([div.get_text(strip=True) for div in divs])

        # Final check
        if content:
            return clean_text(content)
        else:
            print("Couldn't extract meaningful content from the webpage.")
            return None
    except requests.RequestException as e:
        print(f"An error occurred while fetching the article: {e}")
        return None


# Function to clean text
def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters, keeping basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text.strip()


# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


# Function to predict fake news
def predict_fake_news(text, model, vectorizer):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]


# Load model and vectorizer
model, vectorizer = load_model('model/model.pkl', 'model/vectorizer.pkl')

@app.route('/', methods=["POST","GET"])
def index():
    return render_template('index.html')


# Route for home page
@app.route('/results', methods=["POST","GET"])
def results():
    print(request)
    if request.method == "GET":
        return render_template('index.html')
    else:
        url = request.form.get("articleURL")
        print(url)
        article_content = scrape_article(url)
        print(article_content)
        if article_content:
            # Use your existing predict_fake_news function with model and vectorizer
            prediction = predict_fake_news(article_content, model, vectorizer)
            result = "Real" if prediction == 1 else "Fake"
            return render_template('results.html', url=url, result=result)
        else:
            return render_template('results.html', url=url, result=result)



# Route for results page


if __name__ == '__main__':
    app.run(debug=True)
