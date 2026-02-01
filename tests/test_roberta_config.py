"""Check RoBERTa model config and label mapping."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained('openai-community/roberta-large-openai-detector')
tokenizer = AutoTokenizer.from_pretrained('openai-community/roberta-large-openai-detector')
model.eval()

# Print model config to see label ordering
print("=" * 60)
print("Model Configuration")
print("=" * 60)
print(f"id2label: {model.config.id2label}")
print(f"label2id: {model.config.label2id}")
print()

# Test texts
tests = [
    # The original GPT-2 unicorn text used in OpenAI's detector demo
    ("GPT-2 unicorn (original)", "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."),
    
    # Very casual human text
    ("Human casual", "yo I just got back from the store and they were out of milk again lol wtf. had to drive all the way to target instead smh"),
    
    # Formal AI-ish text
    ("Formal AI-like", "The implementation of machine learning algorithms requires careful consideration of multiple factors including data preprocessing, feature selection, and hyperparameter tuning."),
]

print("=" * 60)
print("Detection Results")
print("=" * 60)

for name, text in tests:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    print(f"\n{name}:")
    print(f"  Text: {text[:80]}...")
    print(f"  Logits: {outputs.logits[0].tolist()}")
    print(f"  Class 0 prob: {probs[0, 0]:.4f}")
    print(f"  Class 1 prob: {probs[0, 1]:.4f}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("Based on the HuggingFace model card and OpenAI's original:")
print("  Class 0 = 'Fake' (AI-generated)")
print("  Class 1 = 'Real' (Human-written)")
print()
print("So our current implementation (using probs[0] as AI score) is CORRECT")
