import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from Dbias.bias_classification import classifier
from google.cloud import storage 

# ============================================================
# GCS CONFIGURATION
# ============================================================
BUCKET_NAME = "clearify" 
PROJECT_ID = "eighth-breaker-478412-h9" 
LOCAL_MODEL_BASE_PATH = "/tmp/huggingface_models"

# GCS Paths relative to the bucket root
GCS_POLITICAL_PATH = "clearify/fake_news_model" 
GCS_FAKE_NEWS_PATH = "clearify/political_model"  
GCS_SBIC_PATH = "clearify/sbic_model"

# Local directories where models will be saved
POLITICAL_MODEL_DIR = os.path.join(LOCAL_MODEL_BASE_PATH, "political_model")
SBIC_MODEL_DIR = os.path.join(LOCAL_MODEL_BASE_PATH, "sbic_model")
FAKE_NEWS_MODEL_DIR = os.path.join(LOCAL_MODEL_BASE_PATH, "fake_news_model")

# ============================================================
# HELPER FUNCTIONS: GCS DOWNLOAD
# ============================================================

def download_directory_from_gcs(gcs_prefix: str, local_path: str):
    """Downloads all files under a GCS prefix to a local directory."""
    print(f"Starting download: gs://{BUCKET_NAME}/{gcs_prefix} -> {local_path}")
    
    # Create the local directory
    os.makedirs(local_path, exist_ok=True)
    
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # List all blobs (files) that start with the prefix
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    
    download_count = 0
    for blob in blobs:
        # Skip the directory entry itself if it exists
        if blob.name == gcs_prefix or blob.name.endswith('/'):
            continue
            
        # Calculate the relative path within the local directory
        relative_path = os.path.relpath(blob.name, gcs_prefix)
        local_file_path = os.path.join(local_path, relative_path)
        
        # Ensure subdirectories are created locally if needed
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        print(f"Downloading {blob.name} to {local_file_path}")
        blob.download_to_filename(local_file_path)
        download_count += 1
        
    if download_count == 0:
        print(f"WARNING: No files found under the prefix: {gcs_prefix}")
        
    print(f"Finished downloading {download_count} files for {gcs_prefix}")

# ============================================================
# MODEL CONFIGURATION & PRE-LOADING
# ============================================================

# Step 1: Download all model directories from GCS
print("Starting GCS Model Download Phase...")
download_directory_from_gcs(GCS_POLITICAL_PATH, POLITICAL_MODEL_DIR)
download_directory_from_gcs(GCS_SBIC_PATH, SBIC_MODEL_DIR)
download_directory_from_gcs(GCS_FAKE_NEWS_PATH, FAKE_NEWS_MODEL_DIR)
print("GCS Model Download Phase Complete.")

# Now, the models are available locally at these paths:
POLITICAL_MODEL = POLITICAL_MODEL_DIR
SBIC_MODEL_PATH = SBIC_MODEL_DIR
FAKE_NEWS_MODEL_PATH = FAKE_NEWS_MODEL_DIR

# The rest of your existing setup logic can remain the same, 
# as the transformer's AutoConfig/AutoModel will now load from the local disk path.
cfg = AutoConfig.from_pretrained(POLITICAL_MODEL) 
print("config.id2label:", getattr(cfg, "id2label", None))
print("config.label2id:", getattr(cfg, "label2id", None))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL LOADING FUNCTION
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

# Eagerly load all three PyTorch models into memory at import time
political_tokenizer, political_model = load_model_and_tokenizer(POLITICAL_MODEL)
sbic_tokenizer, sbic_model = load_model_and_tokenizer(SBIC_MODEL_PATH)
fake_tokenizer, fake_news_model = load_model_and_tokenizer(FAKE_NEWS_MODEL_PATH)

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
# D-BIAS SCORE (Remains unchanged - it's handled by Dbias framework import)
# ============================================================
def get_dbias_score(text: str):
    try:
        from Dbias.bias_classification import tokenizer as dbias_tokenizer

        tokens = dbias_tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="tf"
        )
        safe_text = dbias_tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
        result = classifier(safe_text)

        label = result[0]['label']
        confidence = result[0]['score']

        score = confidence * 100
        return round(score, 2), label

    except Exception as e:
        print(f"[Dbias Error] {}")
        return 0.0, "unknown"

# ============================================================
# POLITICAL BIAS ANALYSIS (Lazy check removed)
# ============================================================
def analyze_political_bias(text: str) -> dict:
    # GLOBAL DECLARATION IS NO LONGER STRICTLY NEEDED but harmless
    # Lazy check 'if political_model is None:' REMOVED. Model is guaranteed loaded.

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
# SOCIAL BIAS ANALYSIS (Lazy check removed)
# ============================================================
def analyze_social_bias(text: str) -> dict:
    # Lazy check 'if sbic_model is None:' REMOVED. Model is guaranteed loaded.

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
# FAKE NEWS ANALYSIS (Lazy check removed)
# ============================================================
def analyze_fake_news(text: str) -> float:
    # Lazy check 'if fake_news_model is None:' REMOVED. Model is guaranteed loaded.

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