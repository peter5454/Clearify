from flask import Flask, render_template, request, jsonify
from scraper import scrape_article
from spacyanalyzer import extract_entities, analyze_sentiment, full_analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

MODEL_PATH = "bias_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label mapping (reverse of training)
label_map = {0: "left", 1: "center", 2: "right"}

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

    # Handle Text or URL
    if input_type == 'text':
        text = user_input
    elif input_type == 'url':
        text = scrape_article(user_input)
        if not text:
            return jsonify({"error": "Failed to scrape text from the provided URL."}), 400
    else:
        return jsonify({"error": "Invalid analysis type."}), 400

    # Safety check
    if not text or text.strip() == "":
        return jsonify({"error": "Empty text provided."}), 400

   # ----------------------------
    # NLP + BIAS MODEL PREDICTION
    # ----------------------------
    # Run spaCy-based analysis
    entities = extract_entities(text)
    sentiment_score, sentiment_label = analyze_sentiment(text)
    results = full_analysis(text)

    # Run transformer-based bias prediction
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(predictions, dim=1).item()

    bias_result = {
        "bias_prediction": label_map[pred_label],
        "confidence": round(predictions[0][pred_label].item(), 3)
    }

    # Combine all results
    final_result = {
        "words_analyzed": len(text.split()),
        "bias_score": 50,  # placeholder
        "fake_news_risk": 20,  # placeholder
        "domain_data_score": 72,
        "user_computer_data": 28,
        "emotional_words_percentage": 12,
        "source_reliability": "High",
        "framing_perspective": "Neutral",
        "positive_sentiment": int(sentiment_score*100 if sentiment_label=='positive' else 50),
        "negative_sentiment": int(sentiment_score*100 if sentiment_label=='negative' else 50),
        "word_repetition": [{"word": w, "count": 1} for w in text.split()[:5]],
        "overview": results.get("overview", ""),
        "reliability": results.get("reliability", ""),
        "recommendation": results.get("recommendation", ""),
        "overall_tone": results.get("overall_tone", ""),
        "bias_analysis": bias_result
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
