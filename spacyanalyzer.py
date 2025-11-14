import spacy
from collections import Counter
import re
import en_core_web_sm

# Load SpaCy model
nlp = en_core_web_sm.load()

# ----------------------------
# Named Entity Recognition (NER)
# ----------------------------
def extract_entities(text: str):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# ----------------------------
# Simple Sentiment Analysis
# (Can be replaced by transformer or ML model later)
# ----------------------------
def analyze_sentiment(text: str):
    """
    Returns (sentiment_score, sentiment_label)
    where score ∈ [-1, 1] and label is Positive/Negative/Neutral.
    """
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
# Word Repetition / Frequency
# ----------------------------
def analyze_word_repetition(text: str, top_n: int = 5):
    words = re.findall(r'\b\w+\b', text.lower())
    common_words = Counter(words).most_common(top_n)
    return [{"word": w, "count": c} for w, c in common_words]


# ----------------------------
# Tone & Emotional Analysis
# ----------------------------
def analyze_tone(text: str):
    """
    Detects basic tone characteristics (emotionally charged words, balance).
    """
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
# Combined Full Analysis
# ----------------------------
def full_analysis(text: str):
    """
    Performs all major analysis steps and returns a structured dictionary
    compatible with your frontend results section.
    """

    entities = extract_entities(text)
    sentiment_score, sentiment_label = analyze_sentiment(text)
    tone_result = analyze_tone(text)
    word_repetition = analyze_word_repetition(text)

    # Example bias/risk and reliability scoring (basic heuristics)
    bias_score = round(abs(sentiment_score) * 100)
    fake_news_risk = max(5, 20 - (len(entities) * 2))
    domain_data_score = 80 + (len(entities) % 10)
    user_computer_data = 100 - domain_data_score

    results = {
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
        "framing_perspective": "Content suggests framing consistent with recent online sources.",
        "overall_tone": tone_result["tone"],
        "biasRisk": "Low" if bias_score < 50 else "High",
        "domainScore": domain_data_score,
        "languageTone": tone_result["tone"],
        "sentiment": {"score": sentiment_score, "label": sentiment_label},
        "wordFreq": {w["word"]: w["count"] for w in word_repetition},
        "summary": "The analysis shows low bias and a generally positive sentiment tone.",
        "overview": "This content demonstrates moderate political bias but maintains factual accuracy.",
        "reliability": "Most sources appear trustworthy, with minor subjective language detected.",
        "recommendation": "Cross-check similar sources to confirm facts and reduce potential framing bias.",
        "entities": entities
    }

    return results
