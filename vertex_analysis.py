# vertex_analysis.py
from google.cloud import aiplatform
from Dbias.bias_classification import classifier
from Dbias.bias_classification import tokenizer as dbias_tokenizer
# ============================================================
# CONFIGURATION
# ============================================================

# --- GCP Project and Location ---
# Cloud Run automatically sets environment variables; we use these as defaults.
# These MUST be set in your Cloud Run service environment variables.
PROJECT_ID = "your-gcp-project-id"  # <-- Set this in Cloud Run
LOCATION = "us-central1"           # <-- Set this in Cloud Run

# --- Vertex AI Endpoint IDs ---
# Replace with the actual Endpoint ID strings from your Vertex AI console
POLITICAL_ENDPOINT_ID = "5922821748713062400"  # Example ID
SBIC_ENDPOINT_ID = "6684493035692097536"        # Example ID
FAKE_NEWS_ENDPOINT_ID = "8963314447141568512"   # Example ID
# ----------------------------------------------------------------------



# --- Client Initialization ---
client = aiplatform.PredictionServiceClient(
    client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
)


# ============================================================
# UTILITY FUNCTIONS: Vertex AI Caller
# ============================================================

def _get_endpoint_path(endpoint_id: str) -> str:
    """Utility to format the full endpoint name."""
    return client.endpoint_path(
        project=PROJECT_ID, location=LOCATION, endpoint=endpoint_id
    )

def _call_vertex_endpoint(endpoint_id: str, text: str) -> dict:
    """
    Unified function to call a Vertex AI Endpoint with text input.
    Returns the first prediction object from the response list.
    """
    endpoint = _get_endpoint_path(endpoint_id)
    
    # Format the input for your model (common for text classification)
    instances = [{"text": text}] 

    try:
        response = client.predict(endpoint=endpoint, instances=instances)
        
        # Return the first prediction object
        if response.predictions:
            return response.predictions[0]
        else:
            return {"error": "Prediction returned an empty list."}

    except Exception as e:
        print(f"[Vertex Call Error for ID {endpoint_id}] {e}")
        # Return a safe, default error response
        return {"error": f"API call failed: {e}"}

# ============================================================
# PUBLIC ANALYSIS FUNCTIONS: Vertex AI Based
# ============================================================

def analyze_political_bias(text: str) -> dict:
    """Calls the Political Bias Vertex AI Endpoint."""
    raw_result = _call_vertex_endpoint(POLITICAL_ENDPOINT_ID, text)

    if "error" in raw_result:
        return {"prediction": "center", "confidence": 0.0}

    # IMPORTANT: Adjust these keys ('label', 'score') to match 
    # the actual output dictionary structure of your deployed model
    return {
        "prediction": raw_result.get("label", "center"),
        "confidence": raw_result.get("score", 0.0)
    }

def analyze_social_bias(text: str) -> dict:
    """Calls the Social Bias Vertex AI Endpoint."""
    raw_result = _call_vertex_endpoint(SBIC_ENDPOINT_ID, text)

    if "error" in raw_result:
        return {"bias_category": "none", "confidence": 0.0}

    # IMPORTANT: Adjust these keys ('label', 'score') to match 
    # the actual output dictionary structure of your deployed model
    return {
        "bias_category": raw_result.get("label", "none"),
        "confidence": raw_result.get("score", 0.0)
    }

def analyze_fake_news(text: str) -> float:
    """Calls the Fake News Vertex AI Endpoint and returns the risk score (0-100)."""
    raw_result = _call_vertex_endpoint(FAKE_NEWS_ENDPOINT_ID, text)

    if "error" in raw_result:
        return 0.0

    # IMPORTANT: Adjust these keys ('label', 'score') to match 
    # the actual output dictionary structure of your deployed model
    is_fake_label = raw_result.get("label", 0)  # Assuming 1 is Fake
    confidence = raw_result.get("score", 0.5)

    if is_fake_label == 1:
        score = confidence * 100
    else:
        # Assumes a binary classifier (Fake/Not Fake)
        score = (1 - confidence) * 100 
    
    return round(score, 2)


# ============================================================
# PUBLIC ANALYSIS FUNCTION: Local Dbias Dependency
# ============================================================

def get_dbias_score(text: str):
    """
    Calculates the Dbias score using the locally installed Dbias library.
    """
    try:
        # Replicates your original implementation
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
        return 0.0, "unknown"