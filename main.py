from flask import Flask, render_template, request, jsonify
from scraper import scrape_article
from spacyanalyzer import extract_entities, analyze_sentiment, full_analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Dbias.bias_classification import classifier
import torch
import torch.nn.functional as F

app = Flask(__name__)

# ============================================================
# MODEL CONFIGURATION
# ============================================================
POLITICAL_MODEL_PATH = "bias_model"
SBIC_MODEL_PATH = "sbic_model"
FAKE_NEWS_MODEL_PATH = "fake_news_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD MODELS & TOKENIZERS
# ============================================================
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model

# Load political and SBIC models
political_tokenizer, political_model = load_model_and_tokenizer(POLITICAL_MODEL_PATH)
#fake_news_tokenizer, fake_news_model = load_model_and_tokenizer(FAKE_NEWS_MODEL_PATH)
sbic_tokenizer, sbic_model = load_model_and_tokenizer(SBIC_MODEL_PATH)


# ============================================================
# LABEL MAPS
# ============================================================
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

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_dbias_score(text):
    """
    Returns bias intensity score (0–100) using Dbias classifier.
    """
    result = classifier(text)  # returns list of dicts [{'label': ..., 'score': ...}]
    label = result[0]['label']      # 'Biased' or 'Unbiased'
    confidence = result[0]['score'] # float between 0 and 1

    # Convert confidence to 0–100 scale
    if label.lower() == "biased":
        score = confidence * 100
    else:
        score = (1 - confidence) * 100

    return round(score, 2)



def analyze_political_bias(text):
    inputs = political_tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = political_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()

    return {
        "prediction": political_label_map[pred_label],
        "confidence": round(probs[0][pred_label].item(), 3)
    }


def analyze_social_bias(text):
    inputs = sbic_tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = sbic_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()

    return {
        "bias_category": sbic_label_map[pred_label],
        "confidence": round(probs[0][pred_label].item(), 3)
    }

def analyze_fake_news(text):
    """Return fake news probability as 0-100 scale."""
    inputs = fake_tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = fake_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()
    
    # Assume label 1 = Fake, 0 = Real
    if pred_label == 1:
        score = confidence * 100
    else:
        score = (1 - confidence) * 100

    return round(score, 2)

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    input_type = request.form.get('input_type')
    user_input = request.form.get('text')

    if not user_input or not input_type:
        return jsonify({"error": "No input data provided."}), 400

    # Handle text or URL input
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
    # NLP ANALYSIS
    # ----------------------------
    entities = extract_entities(text)
    sentiment_score, sentiment_label = analyze_sentiment(text)
    results = full_analysis(text)

    # ----------------------------
    # MODEL INFERENCES
    # ----------------------------
    political_result = analyze_political_bias(text)
    sbic_result = analyze_social_bias(text)
    bias_score = get_dbias_score(text)
    #fake_news_score = analyze_fake_news(text)

    # ----------------------------
    # COMBINED OUTPUT
    # ----------------------------
    final_result = {
        "words_analyzed": len(text.split()),
        "bias_score": bias_score,
        "fake_news_risk": 11,  # placeholder
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


@app.route('/about')
def about():
    return render_template('about.html')


# ============================================================
# RUN APP
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
