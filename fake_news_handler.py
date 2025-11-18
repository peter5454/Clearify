# handler_fake_news.py - For the fake_news_model deployment

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- GLOBAL STATE ---
MODEL = None
TOKENIZER = None

class FakeNewsHandler:
    
    # Assuming Binary Classification: 0=Real, 1=Fake (based on original logic)
    REAL_LABEL_ID = 0
    FAKE_LABEL_ID = 1

    def __init__(self):
        self.initialized = False
        self.device = None
        self.model_dir = None

    def initialize(self, context):
        self.initialized = True
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        properties = context.system_properties
        self.model_dir = properties.get("model_dir")
        
        global MODEL, TOKENIZER
        
        try:
            TOKENIZER = AutoTokenizer.from_pretrained(self.model_dir)
            MODEL = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            
            MODEL.to(self.device)
            MODEL.eval() 
            print(f"Fake News Detector Model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading fake news model artifacts: {e}")
            raise e

    def preprocess(self, data):
        # Your fake_tokenizer logic
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
        # Your custom scoring logic: return a single confidence score for "Fake" (ID=1)
        probs = F.softmax(inference_output, dim=-1).tolist()
        final_predictions = []

        for sample_probs in probs:
            # Get the confidence for the 'Fake' label (ID=1)
            fake_confidence = sample_probs[self.FAKE_LABEL_ID]
            
            # The label based on argmax (for complete analysis)
            pred_id = torch.argmax(torch.tensor(sample_probs)).item()
            label = "Fake" if pred_id == self.FAKE_LABEL_ID else "Real"
            
            # Apply your original scoring logic: confidence * 100 
            # if predicted as Fake, otherwise (1-confidence)*100
            if pred_id == self.FAKE_LABEL_ID:
                score = fake_confidence * 100
            else:
                # If Real is predicted, the confidence in Real is high, 
                # meaning confidence in Fake is low (1 - prob[Real])
                real_confidence = sample_probs[self.REAL_LABEL_ID]
                score = (1 - real_confidence) * 100 
                # NOTE: For binary, this is exactly the same as fake_confidence * 100
                
            final_predictions.append({
                "predicted_label": label,
                "fake_news_score": round(score, 2),
                "fake_confidence_raw": round(fake_confidence, 3)
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
_service = FakeNewsHandler()

# This is the entry point expected by the Vertex AI serving container
def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
        
    return _service.handle(data, context)