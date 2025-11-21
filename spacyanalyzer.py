import os
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from collections import Counter
import re
from typing import Dict


import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from google.cloud import storage 

# --- GCS CONFIGURATION ---
BUCKET_NAME = "clearify" 
PROJECT_ID = "eighth-breaker-478412-h9" 
LOCAL_MODEL_BASE_PATH = "/tmp/huggingface_models"
GCS_EMOTION_PATH = "emotion_model"
LOCAL_EMOTION_MODEL_DIR = os.path.join(LOCAL_MODEL_BASE_PATH, "emotion_model")

def download_directory_from_gcs(gcs_prefix: str, local_path: str):
    """Downloads all files under a GCS prefix to a local directory."""
    print(f"Starting download: gs://{BUCKET_NAME}/{gcs_prefix} -> {local_path}")
    
    os.makedirs(local_path, exist_ok=True)
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    
    download_count = 0
    for blob in blobs:
        if blob.name == gcs_prefix or blob.name.endswith('/'):
            continue
            
        prefix_with_slash = gcs_prefix if gcs_prefix.endswith('/') else gcs_prefix + '/'
        relative_path = blob.name[len(prefix_with_slash):]
        local_file_path = os.path.join(local_path, relative_path)
        
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
        download_count += 1
        
    if download_count == 0:
        print(f"WARNING: No files found under the prefix: {gcs_prefix}.")
        
    print(f"Finished downloading {download_count} files for {gcs_prefix}")
# ------------------------------------------------------------------------


# --- EAGER MODEL DOWNLOAD (RUNS AT IMPORT TIME) ---
# This ensures the model is available before any functions are called.
try:
    print("Starting EMOTION Model GCS Download...")
    download_directory_from_gcs(GCS_EMOTION_PATH, LOCAL_EMOTION_MODEL_DIR)
    print("EMOTION Model GCS Download Complete.")
except Exception as e:
    print(f"FATAL ERROR: Could not download EMOTION model from GCS: {e}")
    # You may want to exit or raise here if the model is critical
# --------------------------------------------------


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

    # The local path now points to the downloaded directory
    model_path = LOCAL_EMOTION_MODEL_DIR

    # Create pipeline, loading from the local directory
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Ensure the model is moved to CPU for a standard Cloud Run container
    device = torch.device("cpu")
    model.to(device)

    _EMOTION_PIPELINE = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        # device=-1 for CPU is now redundant since we moved the model explicitly
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
