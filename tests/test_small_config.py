#!/usr/bin/env python
"""Test small training config."""

import yaml

with open('configs/tinker_stealthrl_small.yaml') as f:
    cfg = yaml.safe_load(f)

print('Small Training Config:')
print(f'  Max examples: {cfg["dataset"]["max_examples"]}')
print(f'  Batch size: {cfg["training"]["batch_size"]}')
print(f'  Group size: {cfg["training"]["group_size"]}')
print(f'  Epochs: {cfg["training"]["num_epochs"]}')
print(f'  Dataset: {cfg["dataset"]["path"]}')

# Calculate training details
max_ex = cfg['dataset']['max_examples']
batch_sz = cfg['training']['batch_size']
group_sz = cfg['training']['group_size']
epochs = cfg['training']['num_epochs']

if max_ex:
    num_batches = max_ex // batch_sz
    total_tokens = num_batches * batch_sz * group_sz
    print(f'\nTraining breakdown:')
    print(f'  Total examples: {max_ex}')
    print(f'  Batches per epoch: {num_batches}')
    print(f'  Total batches (all epochs): {num_batches * epochs}')
    print(f'  Total token generations: {total_tokens * epochs}')
    print(f'  Est. training time: ~{num_batches * 2}s per epoch')
