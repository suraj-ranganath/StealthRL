"""Quick sanity test for RoBERTa detector label orientation."""
from eval.detectors import RoBERTaOpenAIDetector

det = RoBERTaOpenAIDetector(device='cpu')
det.load()

# Clearly AI-ish text (formal, structured)
ai_text = "The implementation of machine learning algorithms requires careful consideration of multiple factors including data preprocessing, feature selection, hyperparameter tuning, and model validation. Furthermore, the deployment of such systems necessitates robust infrastructure capable of handling high-throughput inference requests while maintaining low latency."

# Human-ish text (casual, conversational)  
human_text = "honestly I'm so tired of these meetings lol. had like 5 today and my brain is just fried. gonna grab some coffee and maybe take a walk or something idk"

# GPT-2 generated text (from the original training data style)
gpt2_style = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

ai_score = det.get_scores(ai_text)
human_score = det.get_scores(human_text)
gpt2_score = det.get_scores(gpt2_style)

print("=" * 60)
print("RoBERTa OpenAI Detector Sanity Test")
print("=" * 60)
print(f"\n1. Formal AI-like text:    {ai_score:.4f}")
print(f"2. Casual human-like text: {human_score:.4f}")
print(f"3. GPT-2 style text:       {gpt2_score:.4f}")
print()

# Check label orientation
if ai_score > human_score and gpt2_score > human_score:
    print("✓ LABELS CORRECT: Higher score = more AI-like")
elif ai_score < human_score and gpt2_score < human_score:
    print("✗ LABELS INVERTED: Lower score = more AI-like (need to flip!)")
else:
    print("⚠ MIXED RESULTS: Check individual scores above")
