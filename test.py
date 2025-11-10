from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_PATH = "political_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def test_sentence(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
    print(f"Text: {text}\nPred idx: {pred_idx}, probs: {probs.tolist()}\n")

tests = [
    "The government must increase taxes on corporations to fund universal healthcare.",  # clearly left
    "Individual liberty and small government create opportunity for everyone.",  # clearly right
    "We should find a balanced approach that supports both business and social welfare.",  # centrist
]

for t in tests:
    test_sentence(t)
