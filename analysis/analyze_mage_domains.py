#!/usr/bin/env python3
"""Extract and organize all domains from MAGE dataset."""
from datasets import load_from_disk
from collections import defaultdict
import re

ds = load_from_disk('data/mage/test')

# Extract base domain from source
domains = defaultdict(lambda: {"human": [], "ai": []})

for item in ds:
    src = item['src']
    label = item['label']  # 1=human, 0=AI
    
    # Extract base domain (part before _human or _machine or _para)
    match = re.match(r'([a-z0-9]+?)(?:_human|_machine|_para|_gpt|_davinci|_flan|_opt|_bloom|_glm|_neox|_continuation|_specified|_topical)?', src)
    if match:
        base_domain = match.group(1)
    else:
        base_domain = src
    
    if label == 1:
        domains[base_domain]["human"].append(src)
    else:
        domains[base_domain]["ai"].append(src)

# Print organized by domain
print("=" * 100)
print("MAGE DATASET: HUMAN vs AI TEXT DOMAINS")
print("=" * 100)

domain_mapping = {
    'hswag': 'Commonsense Reasoning (HellaSWAG)',
    'xsum': 'News Summarization (XSum)',
    'roct': 'Reading Comprehension (ROCT)',
    'eli5': 'Explanations (ELI5 - Explain Like I\'m 5)',
    'wp': 'Creative Writing (Writing Prompts)',
    'yelp': 'Product Reviews (Yelp)',
    'sci_gen': 'Scientific Text Generation',
    'tldr': 'News/Articles with TL;DR',
    'squad': 'Reading Comprehension (SQuAD)',
    'cmv': 'Social Discussion (Change My View)',
    'cnn': 'News Articles (CNN)',
    'imdb': 'Movie Reviews (IMDB)',
    'pubmed': 'Medical/Scientific Abstracts (PubMed)',
    'dialogsum': 'Dialogue Summarization (DialogSum)',
}

print("\n")
for domain in sorted(domains.keys()):
    desc = domain_mapping.get(domain, domain)
    human_srcs = sorted(set(domains[domain]["human"]))
    ai_srcs = sorted(set(domains[domain]["ai"]))
    
    human_count = sum(1 for item in ds if re.match(rf'{domain}(?:_human|_para)?(?:_human)?$', item['src']) and item['label'] == 1)
    ai_count = sum(1 for item in ds if re.match(rf'{domain}(?:_machine|_gpt|_davinci|_flan|_opt|_bloom|_glm|_neox|_para)?', item['src']) and item['label'] == 0 and item['src'].startswith(domain))
    
    print(f"📚 {domain.upper()}: {desc}")
    print(f"   Human sources ({len(human_srcs)}): {', '.join(human_srcs)}")
    if ai_srcs:
        # Show a few representative AI sources
        sample_ai = sorted(set([re.sub(r'_(gpt|davinci|flan|opt|bloom|glm|neox|continuation|specified|topical).*', '', s) for s in ai_srcs]))
        print(f"   AI variants ({len(ai_srcs)}): {sample_ai[0]} with multiple LLMs/methods")
        print(f"              LLMs: GPT-3.5-turbo, text-davinci-002/003, Flan-T5, OPT, GLM130B, BLOOM, GPT-NeoX, GPT-J")
        print(f"              Methods: continuation, specified, topical, paraphrase")
    print(f"   Total: {human_count} human + {ai_count} AI = {human_count + ai_count}")
    print()

print("=" * 100)
print("SUMMARY")
print("=" * 100)
total_domains = len(domains)
print(f"\nTotal distinct domains: {total_domains}")
print(f"Total samples: {len(ds)}")
print(f"Human samples: {sum(1 for x in ds if x['label'] == 1)}")
print(f"AI samples: {sum(1 for x in ds if x['label'] == 0)}")

print("\n✅ Domains available for comparative analysis:")
print("   - Academic: sci_gen, squad, pubmed")
print("   - News: xsum, cnn, tldr")
print("   - Social: cmv, roct")
print("   - Creative: wp (Writing Prompts)")
print("   - Reviews: yelp, imdb")
print("   - Dialogue: dialogsum")
print("   - Commonsense: hswag")
print("   - Explanations: eli5")

print("\n⚠️  ESL data: NOT AVAILABLE in MAGE")
print("   → Use Tinker dataset for ESL fairness evaluation instead")
