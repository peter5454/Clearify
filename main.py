import os
import re
import json
import logging
from flask import Flask, render_template, request, jsonify
from scraper import scrape_article
from spacyanalyzer import extract_entities, analyze_sentiment, analyze_word_repetition, analyze_tone
from ml_analysis import (
    analyze_political_bias,
    analyze_social_bias,
    analyze_fake_news,
    get_dbias_score
)
from database import save_feedback
import google.genai as genai

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- Environment ---------------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------- Gemini Client ---------------- #
gemini_api_key = os.getenv("GOOGLE_API_KEY")
genai_client = None

if not gemini_api_key:
    logger.error("GOOGLE_API_KEY not found in environment. Gemini functionality will fail.")
else:
    try:
        genai_client = genai.Client(api_key=gemini_api_key)
        logger.info("Gemini client initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize Gemini Client: {e}")
        genai_client = None

def get_gemini_client():
    if genai_client is None:
        raise RuntimeError("Gemini Client not configured or failed to initialize.")
    return genai_client

# ---------------- Flask App ---------------- #
app = Flask(__name__)

# ---------------- Analysis Functions ---------------- #
def derive_final_verdict(political, social, fake_news, dbias_score):
    votes = {"left": 0, "center": 0, "right": 0}

    p_label = political.get("prediction", "center")
    p_conf = political.get("confidence", 0)
    votes[p_label] += p_conf

    if social.get("bias_category") in ["race", "gender", "social", "culture"]:
        votes["center"] -= 0.2
        votes["left"] += 0.1
        votes["right"] += 0.1
    else:
        votes["center"] += 0.2

    if dbias_score > 60:
        if dbias_score < 75:
            votes["center"] += 0.6
        else:
            votes["center"] -= 0.3
            votes["left"] += 0.2
            votes["right"] += 0.2

    if fake_news > 70:
        votes["center"] -= 0.2
        votes["left"] += 0.1
        votes["right"] += 0.1

    final_verdict = max(votes, key=votes.get)
    return final_verdict, votes

def summarize_clearify_results(text, political, social, fake_news, dbias_score, dbias_label):
    client = get_gemini_client()
    final_verdict, votes = derive_final_verdict(political, social, fake_news, dbias_score)

    analysis = {
        "input_text": text,
        "political_bias": political,
        "social_bias": social,
        "fake_news_score": fake_news,
        "dbias": {"score": dbias_score, "label": dbias_label},
        "weighted_votes": votes,
        "final_verdict": final_verdict
    }

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

    Return ONLY JSON.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )
        gemini_text = getattr(response, "output_text", "").strip()
        logger.info("Gemini API call successful.")
    except Exception as e:
        logger.exception(f"Gemini API call failed: {e}")
        return {
            "overall_summary": f"Error: Gemini API call failed: {e}",
            "political_bias_summary": "N/A",
            "social_bias_summary": "N/A",
            "fake_news_summary": "N/A",
            "final_verdict": final_verdict
        }, final_verdict, votes

    parsed = None
    try:
        parsed = json.loads(gemini_text)
    except Exception:
        try:
            match = re.search(r"(\{[\s\S]*\})", gemini_text)
            if match:
                parsed = json.loads(match.group(1))
        except Exception:
            parsed = None

    if isinstance(parsed, dict):
        gemini_summary = {
            "overall_summary": parsed.get("overall_summary"),
            "political_bias_summary": parsed.get("political_bias_summary"),
            "social_bias_summary": parsed.get("social_bias_summary"),
            "fake_news_summary": parsed.get("fake_news_summary"),
            "final_verdict": parsed.get("final_verdict", final_verdict)
        }
    else:
        gemini_summary = {
            "overall_summary": gemini_text,
            "political_bias_summary": None,
            "social_bias_summary": None,
            "fake_news_summary": None,
            "final_verdict": final_verdict
        }

    return gemini_summary, final_verdict, votes

# ---------------- Routes ---------------- #
@app.route('/')
def home():
    logger.info("Serving home page.")
    return render_template('index.html')

@app.route('/about')
def about():
    logger.info("Serving about page.")
    return render_template('about.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    input_type = request.form.get('input_type')
    user_input = request.form.get('text')

    if not user_input or not input_type:
        logger.warning("No input data provided.")
        return jsonify({"error": "No input data provided."}), 400

    if input_type == 'text':
        text = user_input
    elif input_type == 'url':
        text = scrape_article(user_input)
        if not text:
            logger.warning("Failed to scrape text from URL: %s", user_input)
            return jsonify({"error": "Failed to scrape text from the provided URL."}), 400
    else:
        logger.warning("Invalid analysis type: %s", input_type)
        return jsonify({"error": "Invalid analysis type."}), 400

    if not text.strip():
        logger.warning("Empty text provided.")
        return jsonify({"error": "Empty text provided."}), 400

    try:
        entities = extract_entities(text)
        political_result = analyze_political_bias(text)
        sbic_result = analyze_social_bias(text)
        bias_score, bias_label = get_dbias_score(text)
        fake_news_score = analyze_fake_news(text)
        word_repetition = analyze_word_repetition(text)
        tone_result = analyze_tone(text)
        sentiment_label, sentiment_percentage = analyze_sentiment(text)

        gemini_summary, final_verdict, votes = summarize_clearify_results(
            text,
            political_result,
            sbic_result,
            fake_news_score,
            bias_score,
            bias_label
        )

        final_result = {
            "words_analyzed": len(text.split()),
            "bias_score": bias_score,
            "bias_label": bias_label,
            "fake_news_risk": fake_news_score,
            "emotional_words_percentage": tone_result.get("emotional_words_percentage", 0),
            "positive_sentiment": sentiment_percentage if sentiment_label == "Positive" else 0,
            "negative_sentiment": sentiment_percentage if sentiment_label == "Negative" else 0,
            "word_repetition": word_repetition,
            "overall_tone": tone_result.get("tone", ""),
            "political_analysis": political_result,
            "social_bias_analysis": sbic_result,
            "final_verdict": final_verdict,
            "weighted_votes": votes,
            "gemini_summary": gemini_summary
        }

        logger.info("Analysis completed successfully for input type: %s", input_type)
        return jsonify(final_result)

    except Exception as e:
        logger.exception("Error during analysis: %s", e)
        return jsonify({"error": f"Analysis failed: {e}"}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback_route():
    data = request.get_json()
    rating = data.get('rating')
    feedback_text = data.get('feedback_text', '')
    submitted_text = data.get('submitted_text', '')

    if not rating or not (1 <= int(rating) <= 5):
        logger.warning("Invalid feedback rating: %s", rating)
        return jsonify({"error": "Invalid rating"}), 400

    try:
        save_feedback(int(rating), feedback_text, submitted_text)
        logger.info("Feedback saved successfully.")
        return jsonify({"message": "Feedback saved successfully!"})
    except Exception as e:
        logger.exception("Failed to save feedback: %s", e)
        return jsonify({"error": "Failed to save feedback."}), 500

# ---------------- Run App ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info("Starting Flask app on port %d", port)
    app.run(host="0.0.0.0", port=port)
