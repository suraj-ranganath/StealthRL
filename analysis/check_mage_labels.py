#!/usr/bin/env python3
"""Check MAGE dataset label accuracy."""

from datasets import load_from_disk
from collections import Counter, defaultdict

ds = load_from_disk('data/mage/test')

# Check label distribution
label_counts = Counter(ds['label'])
print('Label distribution:')
for label, count in sorted(label_counts.items()):
    print(f'  Label {label}: {count} samples')

print('\n' + '='*80)
print('Sample with label=1 (should be human):')
print('='*80)
human_sample = [item for item in ds if item['label'] == 1][0]
print(f"Label: {human_sample['label']}")
print(f"Source: {human_sample['src']}")
print(f"Text preview: {human_sample['text'][:200]}...")

print('\n' + '='*80)
print('Sample with label=0 (should be AI):')
print('='*80)
ai_sample = [item for item in ds if item['label'] == 0][0]
print(f"Label: {ai_sample['label']}")
print(f"Source: {ai_sample['src']}")
print(f"Text preview: {ai_sample['text'][:200]}...")

print('\n' + '='*80)
print('Source (src) column distribution:')
print('='*80)
src_counts = Counter(ds['src'])
for src, count in sorted(src_counts.items(), key=lambda x: x[1], reverse=True):
    print(f'  {src}: {count} samples')

print('\n' + '='*80)
print('Cross-tabulation: src vs label')
print('='*80)
cross_tab = defaultdict(lambda: {'label_0': 0, 'label_1': 0})
for item in ds:
    if item['label'] == 0:
        cross_tab[item['src']]['label_0'] += 1
    else:
        cross_tab[item['src']]['label_1'] += 1

for src in sorted(cross_tab.keys()):
    label_0 = cross_tab[src]['label_0']
    label_1 = cross_tab[src]['label_1']
    print(f"{src:30s}: label_0={label_0:5d}, label_1={label_1:5d}")
