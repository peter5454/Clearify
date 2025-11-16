import spacy
from collections import Counter
import re

# Lazy load SpaCy (Render-safe)
nlp = None

def get_nlp():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    return nlp


# ----------------------------
# Named Entity Recognition
# ----------------------------
def extract_entities(text: str):
    model = get_nlp()
    doc = model(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# ----------------------------
# Simple Sentiment Analysis
# ----------------------------
def analyze_sentiment(text: str):
    text_lower = text.lower()
    positive_words = ["good", "great", "excellent", "positive", "trust", "reliable", "beneficial"]
    negative_words = ["bad", "terrible", "horrible", "negative", "corrupt", "fake", "biased"]

    pos_count = sum(text_lower.count(w) for w in positive_words)
    neg_count = sum(text_lower.count(w) for w in negative_words)

    if pos_count > neg_count:
        score = round((pos_count - neg_count) / (pos_count + neg_count + 1), 2)
        label = "Positive"
    elif neg_count > pos_count:
        score = round(-(neg_count - pos_count) / (pos_count + neg_count + 1), 2)
        label = "Negative"
    else:
        score = 0
        label = "Neutral"

    return score, label


# ----------------------------
# Word Frequency
# ----------------------------
def analyze_word_repetition(text: str, top_n: int = 5):
    words = re.findall(r'\b\w+\b', text.lower())
    common_words = Counter(words).most_common(top_n)
    return [{"word": w, "count": c} for w, c in common_words]


# ----------------------------
# Tone
# ----------------------------
def analyze_tone(text: str):
    emotional_words = ["love", "hate", "fear", "anger", "joy", "disgust", "trust", "surprise"]
    total_words = len(text.split())
    emotion_count = sum(1 for w in text.lower().split() if w in emotional_words)

    percentage = round((emotion_count / total_words) * 100, 2) if total_words > 0 else 0
    tone = "Emotionally charged" if percentage > 10 else "Neutral and objective"

    return {
        "emotional_words_percentage": percentage,
        "tone": tone
    }


# ----------------------------
# Full Combined Analysis
# ----------------------------
def full_analysis(text: str):
    entities = extract_entities(text)
    sentiment_score, sentiment_label = analyze_sentiment(text)
    tone_result = analyze_tone(text)
    word_repetition = analyze_word_repetition(text)

    bias_score = round(abs(sentiment_score) * 100)
    fake_news_risk = max(5, 20 - (len(entities) * 2))
    domain_data_score = 80 + (len(entities) % 10)
    user_computer_data = 100 - domain_data_score

    return {
        "words_analyzed": len(text.split()),
        "bias_score": bias_score,
        "fake_news_risk": fake_news_risk,
        "domain_data_score": domain_data_score,
        "user_computer_data": user_computer_data,
        "emotional_words_percentage": tone_result["emotional_words_percentage"],
        "source_reliability": "High" if domain_data_score > 70 else "Low",
        "positive_sentiment": max(0, sentiment_score * 100) if sentiment_label == "Positive" else 0,
        "negative_sentiment": abs(sentiment_score * 100) if sentiment_label == "Negative" else 0,
        "word_repetition": word_repetition,
        "overall_tone": tone_result["tone"],
        "sentiment": {"score": sentiment_score, "label": sentiment_label},
        "entities": entities
    }
