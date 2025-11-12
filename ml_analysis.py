import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from Dbias.bias_classification import classifier

# ============================================================
# MODEL CONFIGURATION
# ============================================================
POLITICAL_MODEL_REPO = "pe5tr/political_model"
SBIC_MODEL_PATH = "sbic_model"        # keep local if not on HF
FAKE_NEWS_MODEL_PATH = "fake_news_model"  # keep local if not on HF

cfg = AutoConfig.from_pretrained(POLITICAL_MODEL_REPO)
print("config.id2label:", getattr(cfg, "id2label", None))
print("config.label2id:", getattr(cfg, "label2id", None))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD MODELS & TOKENIZERS
# ============================================================
def load_model_and_tokenizer(model_path_or_repo):
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_repo)
    model = AutoModelForSequenceClassification.from_pretrained(model_path_or_repo)
    model.to(device)
    model.eval()
    return tokenizer, model

# Load models once at import
political_tokenizer, political_model = load_model_and_tokenizer(POLITICAL_MODEL_REPO)
fake_tokenizer, fake_news_model = load_model_and_tokenizer(FAKE_NEWS_MODEL_PATH)
sbic_tokenizer, sbic_model = load_model_and_tokenizer(SBIC_MODEL_PATH)

# ============================================================
# LABEL MAPS
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
# MODEL HELPERS
# ============================================================
def get_dbias_score(text: str) -> float:
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
        print(f"[Dbias Error] {e}")
        return 0.0

def analyze_political_bias(text: str) -> dict:
    inputs = political_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = political_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()
    return {
        "prediction": political_label_map[pred_label],
        "confidence": round(probs[0][pred_label].item(), 3)
    }

def analyze_social_bias(text: str) -> dict:
    inputs = sbic_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = sbic_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()
    return {
        "bias_category": sbic_label_map[pred_label],
        "confidence": round(probs[0][pred_label].item(), 3)
    }

def analyze_fake_news(text: str) -> float:
    inputs = fake_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = fake_news_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    score = confidence * 100 if pred_label == 1 else (1 - confidence) * 100
    return round(score, 2)
