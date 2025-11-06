from flask import Flask, render_template, request, jsonify
from scraper import scrape_article
from spacyanalyzer import extract_entities, analyze_sentiment, full_analysis
from ml_analysis import (
    analyze_political_bias,
    analyze_social_bias,
    analyze_fake_news,
    get_dbias_score
)
import os
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv(dotenv_path="key.env")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

gemini_model = genai.GenerativeModel("gemini-2.5-flash")

app = Flask(__name__)

def derive_final_verdict(political, social, fake_news, dbias_score, sentiment_score):
    """
    Combine multiple model outputs into one final verdict.
    Returns: "Left", "Right", or "Center"
    """
    # Initialize weighted votes
    votes = {"left": 0, "center": 0, "right": 0}

    # --- Political model ---
    p_label = political["prediction"]
    p_conf = political["confidence"]
    votes[p_label] += p_conf * 1.0  # weight = 1

    # --- Social bias heuristic ---
    if social["bias_category"] in ["race", "gender", "social", "culture"]:
        votes["left"] += 0.3  # progressive bias often correlates with left stance
    else:
        votes["center"] += 0.2  # neutral social

    # --- Dbias ---
    if dbias_score > 60:  # high bias score
        if dbias_score < 75:
            votes["center"] += 0.5
        else:
            # extremely biased, use label polarity
            if dbias_score > 80:
                votes["right"] += 0.5

    # --- Fake news heuristic ---
    if fake_news > 70:
        # Penalize credibility
        votes["center"] -= 0.2
        votes["left"] += 0.1  # leaning left conservatively if untrustworthy
        votes["right"] += 0.1

    # --- Sentiment ---
    if sentiment_score < 0.2:
        votes["center"] += 0.1  # neutral sentiment boosts center

    # Choose label with highest weighted vote
    final_verdict = max(votes, key=votes.get)
    return final_verdict, votes


def summarize_clearify_results(text: str):
    # Run models
    political = analyze_political_bias(text)
    social = analyze_social_bias(text)
    fake_news = analyze_fake_news(text)
    dbias_score, dbias_label = get_dbias_score(text)
    sentiment_score, sentiment_label = analyze_sentiment(text)

    # Derive final verdict combining all signals
    final_verdict, votes = derive_final_verdict(political, social, fake_news, dbias_score, sentiment_score)

    # Build structured summary
    analysis = {
        "input_text": text,
        "political_bias": political,
        "social_bias": social,
        "fake_news_score": fake_news,
        "dbias": {"score": dbias_score, "label": dbias_label},
        "sentiment": {"score": sentiment_score, "label": sentiment_label},
        "weighted_votes": votes,
        "final_verdict": final_verdict
    }

    # Use Gemini to create concise human-readable summary
    prompt = f"""
    You are an unbiased political content summarizer.
    Here is raw analysis data from Clearify (including weighted votes):

    {analysis}

    Generate a concise summary with:
    - overall_summary
    - political_bias_summary
    - social_bias_summary
    - fake_news_summary
    - final_verdict
    Return in JSON format.
    """

    response = gemini_model.generate_content(prompt)
    return response.text, final_verdict, votes




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
    bias_score, bias_label = get_dbias_score(text)
    fake_news_score = analyze_fake_news(text)
    gemini_summary, final_verdict, votes = summarize_clearify_results(text)

    final_result = {
    "words_analyzed": len(text.split()),
    "bias_score": bias_score,
    "bias_label": bias_label,
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
    "final_verdict": final_verdict,
    "weighted_votes": votes,
    "gemini_summary": gemini_summary
    }

    print(summarize_clearify_results(text))
    return jsonify(final_result)



if __name__ == "__main__":
    app.run(debug=True)
