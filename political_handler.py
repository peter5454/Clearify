# handler_political.py - For the political_model deployment

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- GLOBAL STATE ---
# These will be loaded once during the container's initialize call
MODEL = None
TOKENIZER = None

class PoliticalBiasHandler:
    
    def __init__(self):
        self.initialized = False
        self.device = None
        self.model_dir = None
        self.label_map = None # Will store the id2label map

    def initialize(self, context):
        self.initialized = True
        
        # 1. Determine Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 2. Get Model Directory
        properties = context.system_properties
        self.model_dir = properties.get("model_dir")
        
        global MODEL, TOKENIZER
        
        # 3. Load Model and Tokenizer (Your load_model_and_tokenizer logic)
        try:
            TOKENIZER = AutoTokenizer.from_pretrained(self.model_dir)
            MODEL = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            
            # Use the label map from the config.json (Recommended)
            self.label_map = MODEL.config.id2label 
            # If config.json doesn't have it, manually use your dict: 
            # self.label_map = {0: "right", 1: "center", 2: "left"}
            
            MODEL.to(self.device)
            MODEL.eval() 
            print(f"Political Model (Labels: {self.label_map}) loaded successfully.")
            
        except Exception as e:
            print(f"Error loading political model artifacts: {e}")
            raise e

    def preprocess(self, data):
        # Your inputs = political_tokenizer(...) logic, adapted for batch input
        input_texts = [row.get("text") for row in data]
        
        # Tokenize the batch of input texts
        tokenized_inputs = TOKENIZER(
            input_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        return {k: v.to(self.device) for k, v in tokenized_inputs.items()}

    def inference(self, preprocessed_data):
        # Your with torch.no_grad(): outputs = political_model(**inputs) logic
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
                "prediction": self.label_map.get(pred_id, f"ID_{pred_id}"),
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
_service = PoliticalBiasHandler()

# This is the entry point expected by the Vertex AI serving container
def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
        
    return _service.handle(data, context)