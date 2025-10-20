from flask import Flask, render_template, request, jsonify
from scraper import scrape_article
from spacyanalyzer import extract_entities, analyze_sentiment, full_analysis


app = Flask(__name__)

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

    # Continue with NLP
    entities = extract_entities(text)
    sentiment_score, sentiment_label = analyze_sentiment(text)
    results = full_analysis(text)

    return jsonify(results)

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
