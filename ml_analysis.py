import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TFAutoModelForSequenceClassification
from google.cloud import storage 
import tensorflow as tf # NEW: Required for the Dbias model prediction
# from Dbias.bias_classification import classifier # REMOVED: Replaced by explicit loading

# ============================================================
# GCS CONFIGURATION
# ============================================================
BUCKET_NAME = "clearify" 
PROJECT_ID = "eighth-breaker-478412-h9" 
LOCAL_MODEL_BASE_PATH = "/tmp/huggingface_models"

# GCS Paths relative to the bucket root
GCS_POLITICAL_PATH = "fake_news_model" # Assuming your paths were swapped in the old example
GCS_FAKE_NEWS_PATH = "political_model"  # Assuming your paths were swapped in the old example
GCS_SBIC_PATH = "sbic_model"
GCS_DBIAS_PATH = "Dbias_model" # NEW: Path to your uploaded Dbias model

# Local directories where models will be saved
POLITICAL_MODEL_DIR = os.path.join(LOCAL_MODEL_BASE_PATH, "political_model")
SBIC_MODEL_DIR = os.path.join(LOCAL_MODEL_BASE_PATH, "sbic_model")
FAKE_NEWS_MODEL_DIR = os.path.join(LOCAL_MODEL_BASE_PATH, "fake_news_model")
DBIAS_MODEL_DIR = os.path.join(LOCAL_MODEL_BASE_PATH, "dbias_model") # NEW

# ============================================================
# HELPER FUNCTIONS: GCS DOWNLOAD (Unchanged)
# ============================================================

def download_directory_from_gcs(gcs_prefix: str, local_path: str):
    """Downloads all files under a GCS prefix to a local directory."""
    print(f"Starting download: gs://{BUCKET_NAME}/{gcs_prefix} -> {local_path}")
    
    os.makedirs(local_path, exist_ok=True)
    
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # List all blobs (files) that start with the prefix and ensure it ends with '/' for directory behavior
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    
    download_count = 0
    for blob in blobs:
        # Skip the directory entry itself if it exists or if the name ends with /
        if blob.name == gcs_prefix or blob.name.endswith('/'):
            continue
            
        # Calculate the relative path within the local directory
        # We need to strip the prefix completely to get the right relative path
        if blob.name.startswith(gcs_prefix):
             # Ensure there is a trailing slash on the prefix for correct slicing
            prefix_with_slash = gcs_prefix if gcs_prefix.endswith('/') else gcs_prefix + '/'
            relative_path = blob.name[len(prefix_with_slash):]
        else:
            # Fallback for unexpected naming, but should not happen if prefix is correct
            relative_path = os.path.relpath(blob.name, gcs_prefix)
            
        local_file_path = os.path.join(local_path, relative_path)
        
        # Ensure subdirectories are created locally if needed
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # print(f"Downloading {blob.name} to {local_file_path}") # Can be noisy, uncomment if needed
        blob.download_to_filename(local_file_path)
        download_count += 1
        
    if download_count == 0:
        print(f"WARNING: No files found under the prefix: {gcs_prefix}. Check GCS path and prefix slash usage.")
        
    print(f"Finished downloading {download_count} files for {gcs_prefix}")

# ============================================================
# MODEL CONFIGURATION & PRE-LOADING
# ============================================================

# Step 1: Download ALL model directories from GCS
print("Starting GCS Model Download Phase...")
download_directory_from_gcs(GCS_POLITICAL_PATH, POLITICAL_MODEL_DIR)
download_directory_from_gcs(GCS_SBIC_PATH, SBIC_MODEL_DIR)
download_directory_from_gcs(GCS_FAKE_NEWS_PATH, FAKE_NEWS_MODEL_DIR)
download_directory_from_gcs(GCS_DBIAS_PATH, DBIAS_MODEL_DIR) # NEW
print("GCS Model Download Phase Complete.")

# Now, the models are available locally at these paths:
POLITICAL_MODEL = POLITICAL_MODEL_DIR
SBIC_MODEL_PATH = SBIC_MODEL_DIR
FAKE_NEWS_MODEL_PATH = FAKE_NEWS_MODEL_DIR
DBIAS_MODEL_PATH = DBIAS_MODEL_DIR # NEW: Local path for Dbias

# The rest of your existing setup logic can remain the same, 
# as the transformer's AutoConfig/AutoModel will now load from the local disk path.
cfg = AutoConfig.from_pretrained(POLITICAL_MODEL) 
print("config.id2label:", getattr(cfg, "id2label", None))
print("config.label2id:", getattr(cfg, "label2id", None))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL LOADING FUNCTION (For PyTorch Models)
# ============================================================
def load_model_and_tokenizer(model_path):
    # This function now loads models from the local directory (downloaded from GCS)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model

# ============================================================
# EAGER MODEL LOADING (This code executes at import time)
# ============================================================
print("Starting Eager Local Model Loading...")

# 1. PyTorch Models
political_tokenizer, political_model = load_model_and_tokenizer(POLITICAL_MODEL)
sbic_tokenizer, sbic_model = load_model_and_tokenizer(SBIC_MODEL_PATH)
fake_tokenizer, fake_news_model = load_model_and_tokenizer(FAKE_NEWS_MODEL_PATH)

# 2. Dbias (TensorFlow) Model (NEW)
dbias_tokenizer = AutoTokenizer.from_pretrained(DBIAS_MODEL_PATH)
# Use TFAutoModelForSequenceClassification for the TensorFlow model
dbias_model = TFAutoModelForSequenceClassification.from_pretrained(DBIAS_MODEL_PATH)
# TF models do not need .to(device) or .eval() in the same way as PyTorch
dbias_model.compile(metrics=["accuracy"]) # Compile is often required for TF models to be usable

# We need the Dbias label map to get the correct output text.
# The original model outputs 0 for 'not bias' and 1 for 'bias'.
DBIAS_LABEL_MAP = {0: "not bias", 1: "bias"} 

print("Eager Local Model Loading Complete.")
# ============================================================
# LABEL MAPS (Unchanged)
# ============================================================
political_label_map = {0: "right", 1: "center", 2: "left"}

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
# D-BIAS SCORE (REWRITTEN)
# ============================================================
def get_dbias_score(text: str):
    try:
        # Use the eagerly loaded Dbias tokenizer and model
        inputs = dbias_tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="tf" # Must be "tf" for the TF model
        )

        # Run prediction using the loaded TF model
        outputs = dbias_model(inputs)
        logits = outputs.logits
        
        # Calculate probabilities using TensorFlow's softmax
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        
        # Find the predicted label (highest probability index)
        pred_index = tf.argmax(probabilities).numpy()
        
        label = DBIAS_LABEL_MAP.get(pred_index, "unknown")
        confidence = probabilities[pred_index]

        score = confidence * 100
        return round(score, 2), label

    except Exception as e:
        # Catch and report any error during Dbias analysis
        print(f"[Dbias Error] {e}")
        return 0.0, "error"

# ============================================================
# POLITICAL BIAS ANALYSIS (Unchanged)
# ============================================================
def analyze_political_bias(text: str) -> dict:
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

# ============================================================
# SOCIAL BIAS ANALYSIS (Unchanged)
# ============================================================
def analyze_social_bias(text: str) -> dict:
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

# ============================================================
# FAKE NEWS ANALYSIS (Unchanged)
# ============================================================
def analyze_fake_news(text: str) -> float:
    inputs = fake_tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = fake_news_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    score = confidence * 100 if pred_label == 1 else (1 - confidence) * 100
    return round(score, 2)