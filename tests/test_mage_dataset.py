#!/usr/bin/env python
"""Quick test of MAGE dataset configuration."""

from datasets import load_from_disk

print("Testing MAGE dataset loader...")

train_ds = load_from_disk('data/mage/train')
val_ds = load_from_disk('data/mage/validation')

print(f'✓ MAGE train split: {len(train_ds)} examples')
print(f'✓ MAGE validation split: {len(val_ds)} examples')

# Show sample
sample = train_ds[0]
print(f'\nSample example:')
print(f'  text: {sample["text"][:80]}...')
print(f'  label: {sample["label"]} (1=human, 0=AI)')
print(f'  src: {sample["src"]}')

# Count human examples
human_count = sum(1 for ex in train_ds if ex['label'] == 1)
ai_count = sum(1 for ex in train_ds if ex['label'] == 0)
print(f'\nTrain split breakdown:')
print(f'  Human examples: {human_count}')
print(f'  AI examples: {ai_count}')
