# handler_sbic.py - For the sbic_model deployment

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- GLOBAL STATE ---
MODEL = None
TOKENIZER = None

class SBICBiasHandler:
    
    # Your provided label map
    SBIC_LABEL_MAP = {
        0: "none",
        1: "race",
        2: "gender",
        3: "social",
        4: "body",
        5: "culture",
        6: "disabled",
        7: "victim"
    }

    def __init__(self):
        self.initialized = False
        self.device = None
        self.model_dir = None
        self.label_map = None 

    def initialize(self, context):
        self.initialized = True
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        properties = context.system_properties
        self.model_dir = properties.get("model_dir")
        
        global MODEL, TOKENIZER
        
        try:
            TOKENIZER = AutoTokenizer.from_pretrained(self.model_dir)
            MODEL = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            
            # Prefer the config.json map, fallback to the hardcoded map
            self.label_map = MODEL.config.id2label if hasattr(MODEL.config, 'id2label') else self.SBIC_LABEL_MAP
            
            MODEL.to(self.device)
            MODEL.eval() 
            print(f"SBIC Model (Labels: {self.label_map}) loaded successfully.")
            
        except Exception as e:
            print(f"Error loading SBIC model artifacts: {e}")
            raise e

    def preprocess(self, data):
        # Your sbic_tokenizer logic
        input_texts = [row.get("text") for row in data]
        
        tokenized_inputs = TOKENIZER(
            input_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        return {k: v.to(self.device) for k, v in tokenized_inputs.items()}

    def inference(self, preprocessed_data):
        with torch.no_grad():
            outputs = MODEL(**preprocessed_data)
        return outputs.logits

    def postprocess(self, inference_output):
        # Your F.softmax, argmax, and label mapping logic
        probs = F.softmax(inference_output, dim=-1).tolist()
        final_predictions = []

        for sample_probs in probs:
            pred_id = torch.argmax(torch.tensor(sample_probs)).item()
            
            final_predictions.append({
                "bias_category": self.label_map.get(pred_id, f"ID_{pred_id}"),
                "confidence": round(sample_probs[pred_id], 3)
            })
            
        return final_predictions

    def handle(self, data, context):
        if not self.initialized:
            self.initialize(context)
            
        preprocessed_data = self.preprocess(data)
        inference_output = self.inference(preprocessed_data)
        postprocessed_output = self.postprocess(inference_output)
        
        return postprocessed_output

# Instantiate the service
_service = SBICBiasHandler()

# This is the entry point expected by the Vertex AI serving container
def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
        
    return _service.handle(data, context)