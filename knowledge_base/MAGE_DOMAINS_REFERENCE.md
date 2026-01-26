# MAGE Dataset Domains Reference

## Overview
- **Total samples**: 60,743 (equal split: 30,265 human + 30,478 AI)
- **Format**: HuggingFace dataset (load from disk)
- **Location**: `data/mage/test`
- **Structure**: columns = [text, label, src]

## Human Text Domains (14 sources)

| Domain | Source | Type | Examples |
|--------|--------|------|----------|
| **Academic** | sci_gen_human, pubmed_human | Scientific writing | Papers, abstracts |
| **Reasoning** | hswag_human | Commonsense reasoning | HellaSWAG dataset |
| **Reading** | squad_human, roct_human | Reading comprehension | SQuAD, ROCT Q&A |
| **News** | xsum_human, cnn_human, tldr_human | News/articles | Summaries, articles |
| **Reviews** | yelp_human, imdb_human | Product/movie reviews | Ratings with text |
| **Social** | cmv_human | Discussion/debate | Change My View posts |
| **Creative** | wp_human | Creative writing | Writing prompts |
| **Dialogue** | dialogsum_human | Dialogue summaries | Multi-turn conversations |
| **Explanations** | eli5_human | Simple explanations | Explain Like I'm 5 |

## AI-Generated Text Variants (200+ sources)

### LLMs Used:
- **OpenAI**: GPT-3.5-turbo, text-davinci-002, text-davinci-003
- **Meta**: Flan-T5 (various sizes), OPT (various sizes)
- **Others**: GLM-130B, BLOOM, GPT-NeoX, GPT-J

### Generation Methods:
- **continuation**: Continue given prompt
- **specified**: Follow specific instructions
- **topical**: Topic-guided generation
- **paraphrase**: Rephrase existing text

### Example sources:
- `sci_gen_davinci_003_continuation` - davinci-003 continuing scientific text
- `yelp_gpt3_specified_v2` - GPT-3.5 following specific instructions for reviews
- `eli5_flan_t5_large_topical` - Flan-T5-large with topic guidance

## Domain Distribution

| Domain | Count | Percentage | Best For |
|--------|-------|-----------|----------|
| sci_gen (human) | 2,538 | 4.18% | Academic detector performance |
| squad (human) | 2,508 | 4.13% | QA-style defense |
| eli5 (human) | 3,156 | 5.20% | Casual explanation defense |
| xsum (human) | 3,283 | 5.40% | News article defense |
| hswag (human) | 3,292 | 5.42% | Commonsense reasoning defense |
| yelp (human) | 2,652 | 4.37% | Review defense |
| wp (human) | 3,099 | 5.10% | Creative writing defense |
| roct (human) | 3,275 | 5.39% | Reading comp defense |
| cmv (human) | 2,403 | 3.96% | Social/debate defense |
| tldr (human) | 2,535 | 4.17% | News TL;DR defense |
| pubmed (human) | 400 | 0.66% | Medical text defense |
| cnn (human) | 400 | 0.66% | CNN news defense |
| imdb (human) | 400 | 0.66% | Movie review defense |
| dialogsum (human) | 324 | 0.53% | Dialogue defense |

## Usage Example

```python
from stealthrl.tinker.dataset import StealthRLDatasetBuilder

# Load MAGE dataset
builder = StealthRLDatasetBuilder(
    data_path="data/mage",        # Auto-detects HuggingFace format
    batch_size=32,
    group_size=4,
    model_name_for_tokenizer="gpt2",
    renderer_name="qwen3",
    reward_config={
        "detector_names": ["roberta_openai", "fast_detectgpt"],
        "detector_weights": {"roberta_openai": 0.6, "fast_detectgpt": 0.4},
    },
)

train_dataset, test_dataset = await builder()

# Each example contains:
# - human_reference: Human-written text to paraphrase
# - domain: "academic", "news", "informal", "reasoning", etc.
# - metadata: {"source": "eli5_human", "original_label": 1}
```

## Domain Mapping

The dataset automatically maps source names to domains:

```python
domain_map = {
    'eli5': 'informal',
    'hswag': 'reasoning',
    'xsum': 'news',
    'roct': 'reading',
    'wp': 'creative',
    'yelp': 'review',
    'sci_gen': 'academic',
    'tldr': 'news',
    'squad': 'reading',
    'cmv': 'social',
    'cnn': 'news',
    'imdb': 'review',
    'pubmed': 'academic',
    'dialogsum': 'dialogue',
}
```

## Key Points

✅ **All human texts** - Only label=1 (human-written) used for training  
✅ **Diverse domains** - 14 different text types to test detector robustness  
✅ **AI baselines** - 200+ AI variants to understand detector biases  
✅ **Large scale** - 30K+ human examples for robust training  
✅ **Auto-detected** - Just point to `data/mage` and loading is automatic  

## Training Recommendation

For best generalization across domains:
- Sample evenly from each domain during training
- Evaluate separately by domain post-training
- Monitor transfer across unseen AI generation methods
