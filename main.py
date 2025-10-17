from flask import Flask, render_template, request, jsonify
from scraper import scrape_article
import nltk
nltk.download('punkt_tab')
from spacyanalyzer import full_analysis
from scraper import fetch_data
import spacyanalyzer
import torch
import spacy

app = Flask(__name__)

# 1. Main page
@app.route('/')
def home():
    results = None
    text = ""
    if request.method == "POST":
        text = request.form["user_input"]
        results = full_analysis(text)
    return render_template('index.html', results=results, text=text)

def main():
    results=[]
    for doc in docs:
        processed = analyze_text(doc)
        results.append(processed)

# 2. Analyze - Now returns JSON for same-page updates
@app.route('/analyze', methods=['POST'])
def analyze():
    user_text = request.form.get("text")
    input_type = request.form.get("input_type", "text")
    text = request.form['text']
    analysis = full_analysis(text)
    
    # If URL input, scrape the article first
    if input_type == "url" and user_text:
        try:
            scraped_data = scrape_article(user_text)
            user_text = scraped_data.get("text", user_text)
        except Exception as e:
            print(f"Scraping error: {e}")
    
    
    # Generate analysis results (replace with real ML model)
    result = {
        "words_analyzed": len(user_text.split()) if user_text else 0,
        "domain_data_score": 72,
        "user_computer_data": 28,
        "emotional_words_percentage": 12,
        "source_reliability": "High",
        "positive_sentiment": 90,
        "negative_sentiment": 84,
        "word_repetition": [
            {"word": "Crisis", "count": 22},
            {"word": "Government warned", "count": 18},
            {"word": "Urgent", "count": 16}
        ],
        "framing_perspective": "Recent reading via government sources, can elaborate reading as possible",
        "overall_tone": "Balanced but critical",
        "recommendation": "Cross-check similar sources to confirm facts and reduce potential framing bias.",
        "bias_score": 45,
        "fake_news_risk": 18,
        "overview": "This content demonstrates moderate political bias but maintains factual accuracy.",
        "reliability": "Most sources appear trustworthy, with minor subjective language detected."
    }

    return jsonify(result)

# 3. About Us
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)