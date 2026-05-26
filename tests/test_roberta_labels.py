"""Test RoBERTa detector label configuration."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = 'openai-community/roberta-large-openai-detector'
print(f"Loading {model_name}...")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check model config for label info
print(f"\nModel config id2label: {model.config.id2label}")
print(f"Num labels: {model.config.num_labels}")

# Test with obvious AI text
ai_text = 'The rapid advancement of artificial intelligence has fundamentally transformed the way we approach complex problem-solving in modern society. These sophisticated algorithms enable unprecedented levels of automation and optimization across diverse sectors.'

human_text = 'So yeah I was just chilling at home watching Netflix when my cat knocked over my coffee all over my laptop. Worst day ever lol.'

def get_probs(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0].tolist()

print()
print(f"AI text probs: {get_probs(ai_text)}")
print(f"Human text probs: {get_probs(human_text)}")
print()
print("Index 0 = ", model.config.id2label.get(0, 'Unknown'))
print("Index 1 = ", model.config.id2label.get(1, 'Unknown'))
