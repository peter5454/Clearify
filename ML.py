import torch
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    dtype=torch.float16,
    device=0
)

result = classifier("i")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]