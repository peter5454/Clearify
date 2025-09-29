from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 1. Main page
@app.route('/')
def home():
    return render_template('index.html')  # contains input + submit form

# 2. Analyze
@app.route('/analyze', methods=['POST'])
def analyze():
    user_text = request.form.get("text")  # if using a form
    # user_text = request.json.get("text")  # if sending JSON via fetch()

    # TODO: replace with real analysis model
    result = {
        "words_analyzed": len(user_text.split()),
        "propaganda_percentage": 18,
        "bias_score": 0.42
    }

    # Option A: render result page
    return render_template('results.html', result=result, text=user_text)

    # Option B: return JSON if frontend handles UI
    # return jsonify(result)

# 3. About Us
@app.route('/about')
def about():
    return render_template('about.html')  # static info
