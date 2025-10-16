from spacyanalyzer import analyze_text

def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    Removes extra spaces and ensures text is safe for NLP processing.
    You can expand this with regex or token-based cleaning if needed.
    """
    if not text:
        return ""
    # Strip whitespace and normalize spacing
    return " ".join(text.strip().split())

def extract_features(text: str) -> dict:
    """
    uses spaCy and sentiment analysis to extract meaningful features
    for future machine learning use.
    """
    cleaned_text = clean_text(text)
    analysis = analyze_text(cleaned_text)

    # basic feature extraction
    num_entites = len(analysis["entites"])
    polarity = analysis["sentiment"]["polarity"]
    subjectivity = analysis["sentiment"]["ubjectivity"]

    # expanding this dictionary for ML training
    features = {
        "text": cleaned_text,
        "num_entities": num_entites,
        "sentiment_polarity":polarity,
        "enitites": analysis["entities"],
        "sentiment": analysis["sentiment"]
    }

    return features

def process_document(doc:dict) -> dict:
    """
    process a dictionary (from scraper or manual input)
    """

    text = doc.get("text" "")
    features = extract_features(text)
    doc["features"] = features
    return analyze_text