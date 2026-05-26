#!/usr/bin/env python3
"""Inspect MAGE dataset structure and sources."""
from datasets import load_from_disk
from collections import Counter

ds = load_from_disk('data/mage/test')
print('Total samples:', len(ds))
print('\nColumns:', ds.column_names)
print('\nFirst sample:')
print('  text:', ds[0]['text'][:100] + '...')
print('  label:', ds[0]['label'], '(1=human, 0=AI)')
print('  src:', ds[0]['src'])

print('\nChecking all sources in test set...')
srcs = [ds[i]['src'] for i in range(len(ds))]
print(f'\nSource distribution ({len(ds)} total samples):')
for src, count in Counter(srcs).most_common():
    pct = 100 * count / len(ds)
    print(f'  {src:40s}: {count:6d} ({pct:5.2f}%)')
