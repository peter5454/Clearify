import spacy
from textblob import TextBlob
from spacy.tokens import Doc

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Register custom sentiment extensions
if not Doc.has_extension("polarity"):
    Doc.set_extension("polarity", getter=lambda doc: TextBlob(doc.text).sentiment.polarity)
if not Doc.has_extension("subjectivity"):
    Doc.set_extension("subjectivity", getter=lambda doc: TextBlob(doc.text).sentiment.subjectivity)


# --------------------------
# ENTITY ANALYSIS
# --------------------------
def extract_entities(text: str):
    """Extract named entities (like people, organizations, places)."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def analyze_entities(text: str):
    """Alias for extract_entities (kept for backward compatibility)."""
    return extract_entities(text)


# --------------------------
# SENTIMENT ANALYSIS
# --------------------------
def analyze_sentiment(text: str):
    """Analyze text sentiment using TextBlob (integrated into SpaCy)."""
    doc = nlp(text)
    return {
        "polarity": round(doc._.polarity, 3),
        "subjectivity": round(doc._.subjectivity, 3)
    }


def full_sentiment(text: str):
    """Return combined sentiment and entities."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentiment = {
        "polarity": round(doc._.polarity, 3),
        "subjectivity": round(doc._.subjectivity, 3)
    }
    return {
        "entities": entities,
        "sentiment": sentiment
    }


# --------------------------
# FULL ANALYSIS
# --------------------------
def full_analysis(text: str):
    """Run full analysis: entities + sentiment."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentiment = {
        "polarity": round(doc._.polarity, 3),
        "subjectivity": round(doc._.subjectivity, 3)
    }

    return {
        "text": text,
        "entities": entities,
        "sentiment": sentiment
    }
