"""Test RoBERTa detector with different text types."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = 'openai-community/roberta-large-openai-detector'
print(f"Loading {model_name}...")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

def get_ai_prob(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    # Index 0 = fake/AI, Index 1 = real/human
    return probs[0, 0].item(), probs[0, 1].item()

# Test texts
tests = [
    ("GPT-2 style formal", "The rapid advancement of artificial intelligence has fundamentally transformed the way we approach complex problem-solving in modern society. These sophisticated algorithms enable unprecedented levels of automation and optimization across diverse sectors, facilitating enhanced productivity and innovative solutions."),
    
    ("Human casual", "So yeah I was just chilling at home watching Netflix when my cat knocked over my coffee all over my laptop. Worst day ever lol. Had to spend like $200 fixing it smh"),
    
    ("MAGE-style ChatGPT", "Google is facing legal action by member of Congress and Democratic presidential candidate, Tulsi Gabbard, due to the suspension of Gabbard's campaign advertising account during the first Democratic presidential debate."),
    
    ("Technical paper abstract", "The amount of data moved over dedicated and non-dedicated network links increases much faster than the increase in the network capacity, but the current solutions fail to guarantee even the promised achievable transfer throughputs."),
    
    ("Very human casual", "Dude idk what ur talking about tbh. Like the movie was fine but people r overreacting lol. Its not THAT good ya know?"),
]

print("\n" + "="*80)
print(f"{'Text Type':<25} | {'AI Prob':<10} | {'Human Prob':<10} | {'Prediction'}")
print("="*80)

for name, text in tests:
    ai_prob, human_prob = get_ai_prob(text)
    pred = "AI" if ai_prob > 0.5 else "HUMAN"
    print(f"{name:<25} | {ai_prob:.4f}     | {human_prob:.4f}     | {pred}")
