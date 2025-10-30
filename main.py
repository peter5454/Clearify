from flask import Flask, render_template, request, jsonify
from scraper import scrape_article
from spacyanalyzer import extract_entities, analyze_sentiment, full_analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# ----------------------------
# MODEL PATHS
# ----------------------------
POLITICAL_MODEL_PATH = "bias_model"
SBIC_MODEL_PATH = "sbic_model"

# ----------------------------
# LOAD MODELS & TOKENIZERS
# ----------------------------
# Political bias model
political_tokenizer = AutoTokenizer.from_pretrained(POLITICAL_MODEL_PATH)
political_model = AutoModelForSequenceClassification.from_pretrained(POLITICAL_MODEL_PATH)

# SBIC bias-category model
sbic_tokenizer = AutoTokenizer.from_pretrained(SBIC_MODEL_PATH)
sbic_model = AutoModelForSequenceClassification.from_pretrained(SBIC_MODEL_PATH)

# ----------------------------
# DEVICE CONFIGURATION
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
political_model.to(device)
sbic_model.to(device)

political_model.eval()
sbic_model.eval()

# ----------------------------
# LABEL MAPS
# ----------------------------
# Political orientation model
political_label_map = {0: "left", 1: "center", 2: "right"}

sbic_label_map = {
    0: "none",
    1: "race",
    2: "gender",
    3: "social",
    4: "body",
    5: "culture",
    6: "disabled",
    7: "victim"
}

# ----------------------------
# MAIN PAGE
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ----------------------------
# ANALYSIS ENDPOINT
# ----------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    input_type = request.form.get('input_type')
    user_input = request.form.get('text')

    if not user_input or not input_type:
        return jsonify({"error": "No input data provided."}), 400

    # Handle Text or URL input
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

    # ----------------------------
    # NLP & SENTIMENT
    # ----------------------------
    entities = extract_entities(text)
    sentiment_score, sentiment_label = analyze_sentiment(text)
    results = full_analysis(text)

    # ----------------------------
    # POLITICAL BIAS MODEL
    # ----------------------------
    pol_inputs = political_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    pol_inputs = {k: v.to(device) for k, v in pol_inputs.items()}

    with torch.no_grad():
        pol_outputs = political_model(**pol_inputs)
        pol_probs = torch.nn.functional.softmax(pol_outputs.logits, dim=-1)
        pol_pred_label = torch.argmax(pol_probs, dim=1).item()

    political_result = {
        "prediction": political_label_map[pol_pred_label],
        "confidence": round(pol_probs[0][pol_pred_label].item(), 3)
    }

    # ----------------------------
    # SBIC SOCIAL BIAS MODEL
    # ----------------------------
    sbic_inputs = sbic_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    sbic_inputs = {k: v.to(device) for k, v in sbic_inputs.items()}

    with torch.no_grad():
        sbic_outputs = sbic_model(**sbic_inputs)
        sbic_probs = torch.nn.functional.softmax(sbic_outputs.logits, dim=-1)
        sbic_pred_label = torch.argmax(sbic_probs, dim=1).item()

    sbic_result = {
        "bias_category": sbic_label_map[sbic_pred_label],
        "confidence": round(sbic_probs[0][sbic_pred_label].item(), 3)
    }

    # ----------------------------
    # FINAL COMBINED OUTPUT
    # ----------------------------
    final_result = {
        "words_analyzed": len(text.split()),
        "bias_score": 50,  # placeholder
        "fake_news_risk": 20,  # placeholder
        "domain_data_score": 72,
        "user_computer_data": 28,
        "emotional_words_percentage": 12,
        "source_reliability": "High",
        "framing_perspective": "Neutral",
        "positive_sentiment": int(sentiment_score * 100 if sentiment_label == 'positive' else 50),
        "negative_sentiment": int(sentiment_score * 100 if sentiment_label == 'negative' else 50),
        "word_repetition": [{"word": w, "count": 1} for w in text.split()[:5]],
        "overview": results.get("overview", ""),
        "reliability": results.get("reliability", ""),
        "recommendation": results.get("recommendation", ""),
        "overall_tone": results.get("overall_tone", ""),
        "political_analysis": political_result,
        "social_bias_analysis": sbic_result,
    }

    return jsonify(final_result)


# ----------------------------
# ABOUT PAGE
# ----------------------------
@app.route('/about')
def about():
    return render_template('about.html')


# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
