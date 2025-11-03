from flask import Flask, render_template, request, jsonify
from scraper import scrape_article
from spacyanalyzer import extract_entities, analyze_sentiment, full_analysis
from ml_analysis import (
    analyze_political_bias,
    analyze_social_bias,
    analyze_fake_news,
    get_dbias_score
)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    input_type = request.form.get('input_type')
    user_input = request.form.get('text')

    if not user_input or not input_type:
        return jsonify({"error": "No input data provided."}), 400

    if input_type == 'text':
        text = user_input
    elif input_type == 'url':
        text = scrape_article(user_input)
        if not text:
            return jsonify({"error": "Failed to scrape text from the provided URL."}), 400
    else:
        return jsonify({"error": "Invalid analysis type."}), 400

    if not text.strip():
        return jsonify({"error": "Empty text provided."}), 400

    # Run analyses
    entities = extract_entities(text)
    sentiment_score, sentiment_label = analyze_sentiment(text)
    results = full_analysis(text)
    political_result = analyze_political_bias(text)
    sbic_result = analyze_social_bias(text)
    bias_score = get_dbias_score(text)
    fake_news_score = analyze_fake_news(text)

    final_result = {
        "words_analyzed": len(text.split()),
        "bias_score": bias_score,
        "fake_news_risk": fake_news_score,
        "domain_data_score": 72,
        "user_computer_data": 28,
        "emotional_words_percentage": 12,
        "source_reliability": "High",
        "framing_perspective": "Neutral",
        "positive_sentiment": int(sentiment_score * 100 if sentiment_label == 'Positive' else 50),
        "negative_sentiment": int(sentiment_score * 100 if sentiment_label == 'Negative' else 50),
        "word_repetition": [{"word": w, "count": 1} for w in text.split()[:5]],
        "overview": results.get("overview", ""),
        "reliability": results.get("reliability", ""),
        "recommendation": results.get("recommendation", ""),
        "overall_tone": results.get("overall_tone", ""),
        "political_analysis": political_result,
        "social_bias_analysis": sbic_result,
    }

    return jsonify(final_result)

if __name__ == "__main__":
    app.run(debug=True)
