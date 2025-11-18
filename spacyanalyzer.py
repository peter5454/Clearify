import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from collections import Counter
import re

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict
# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

if "spacytextblob" not in nlp.pipe_names:
    nlp.add_pipe("spacytextblob", last=True)

_EMOTION_PIPELINE = None

_SMALL_EMOTION_LEXICON = {
    "love", "hate", "fear", "anger", "joy", "disgust", "trust", "surprise",
    "happy", "sad", "angry", "excited", "afraid", "terrified", "disgusted",
}


def _get_emotion_pipeline():

    global _EMOTION_PIPELINE
    if _EMOTION_PIPELINE is not None:
        return _EMOTION_PIPELINE

    model_name = "cardiffnlp/twitter-roberta-base-emotion"

    # Create pipeline (this downloads model the first time)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    _EMOTION_PIPELINE = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=-1  # CPU. If you have GPU, set device=0
    )
    return _EMOTION_PIPELINE
# ----------------------------
# Named Entity Recognition
# ----------------------------
def extract_entities(text: str):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# ----------------------------
# TRUE Sentiment Analysis (spaCyTextBlob)
# ----------------------------
def analyze_sentiment(text: str):

    doc = nlp(text)

    polarity = float(doc._.blob.polarity)
    polarity = round(polarity, 4)

    # sentiment % conversion:
    sentiment_percentage = round(((polarity + 1) / 2) * 100, 2)

    # sentiment label
    if polarity > 0.05:
        label = "Positive"
        sentiment_percentage = round(polarity * 100, 2)
    elif polarity < -0.05:
        label = "Negative"
        sentiment_percentage = round(abs(polarity) * 100, 2)
    else:
        label = "Neutral"
        sentiment_percentage = 50.0

    return label, sentiment_percentage

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
def analyze_tone(text: str) -> Dict:

    pipe = _get_emotion_pipeline()

    preds = pipe(text)[0]

    # Convert model outputs to clean dict
    scores = {item["label"].lower(): round(float(item["score"]), 4) for item in preds}

    # Ensure consistent 7-class output
    expected = ["anger", "joy", "optimism", "sadness", "surprise", "disgust", "fear"]
    for e in expected:
        scores.setdefault(e, 0.0)

    primary_emotion = max(scores, key=scores.get)
    emotion_strength = scores[primary_emotion]

    # optional lightweight lexicon match for % emotionally-charged words
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = max(len(words), 1)
    emotion_word_count = sum(1 for w in words if w in _SMALL_EMOTION_LEXICON)
    emotional_words_percentage = round((emotion_word_count / total_words) * 100, 2)

    return {
        "tone": f"Primary emotion: {primary_emotion}",
        "emotion_scores": scores,
        "primary_emotion": primary_emotion,
        "emotion_strength": round(emotion_strength, 4),
        "emotional_words_percentage": emotional_words_percentage
    }
