from flask import Flask, render_template, request, jsonify
from scraper import scrape_article
from spacyanalyzer import extract_entities, analyze_sentiment, analyze_word_repetition, analyze_tone
from ml_analysis import (
    analyze_political_bias,
    analyze_social_bias,
    analyze_fake_news,
    get_dbias_score
)
import os
import google.generativeai as genai
import json
import re
from database import save_feedback 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


genai_client = None



gemini_api_key = os.getenv("GOOGLE_API_KEY")

if not gemini_api_key:
    print("FATAL ERROR: GOOGLE_API_KEY not found in environment. Gemini functionality will fail.")
else:
    genai.configure(api_key=gemini_api_key)
    try:
        genai_client = genai.Client()
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize Gemini Client with genai.Client(): {e}")
        genai_client = None
# -------------------------------------------



def get_gemini_client(): 
    if genai_client is None:
        raise RuntimeError("Gemini Client is not configured or failed to initialize.")
    return genai_client

app = Flask(__name__)

def derive_final_verdict(political, social, fake_news, dbias_score):
    # [ ... derive_final_verdict function remains unchanged ... ]
    votes = {"left": 0, "center": 0, "right": 0}

    p_label = political["prediction"]
    p_conf = political["confidence"]
    votes[p_label] += p_conf

    if social["bias_category"] in ["race", "gender", "social", "culture"]:
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


# --- MODIFIED: summarize_clearify_results uses the client object ---
def summarize_clearify_results(text: str, political, social, fake_news, dbias_score, dbias_label):
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
    Return in JSON format.
    """

    # Call Gemini
    client = get_gemini_client() # Get the initialized client object
    
    try:
        # FIX: The correct modern call using the client object
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        gemini_summary = {
            "overall_summary": f"Error: Gemini API call failed. Details: {e}",
            "political_bias_summary": "N/A",
            "social_bias_summary": "N/A",
            "fake_news_summary": "N/A",
            "final_verdict": final_verdict
        }
        return gemini_summary, final_verdict, votes
    
    gemini_text = getattr(response, "text", "") or str(response)

    parsed = None
    gemini_json_fallback = {
        "overall_summary": None,
        "political_bias_summary": None,
        "social_bias_summary": None,
        "fake_news_summary": None,
        "final_verdict": final_verdict
    }

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
            "overall_summary": parsed.get("overall_summary") or parsed.get("overallSummary") or parsed.get("summary"),
            "political_bias_summary": parsed.get("political_bias_summary") or parsed.get("politicalBiasSummary"),
            "social_bias_summary": parsed.get("social_bias_summary") or parsed.get("socialBiasSummary"),
            "fake_news_summary": parsed.get("fake_news_summary") or parsed.get("fakeNewsSummary"),
            "final_verdict": parsed.get("final_verdict") or final_verdict
        }
    else:
        gemini_summary = gemini_json_fallback.copy()
        gemini_summary["overall_summary"] = gemini_text.strip()
        gemini_summary["final_verdict"] = final_verdict

    return gemini_summary, final_verdict, votes


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
        bias_label,
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


    return jsonify(final_result)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    rating = data.get('rating')
    feedback_text = data.get('feedback_text', '')
    submitted_text = data.get('submitted_text', '')

    if not rating or not (1 <= int(rating) <= 5):
        return jsonify({"error": "Invalid rating"}), 400

    save_feedback(int(rating), feedback_text, submitted_text)
    return jsonify({"message": "Feedback saved successfully!"})


if __name__ == "__main__": 
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)