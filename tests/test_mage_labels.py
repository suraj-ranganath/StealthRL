"""Quick test of MAGE label fix."""
from eval.data import MAGEDataset

# Load MAGE
ds = MAGEDataset.download(split='test[:200]')
print(f'Loaded {len(ds)} samples')
print(f'  Human: {len(ds.human_samples)}')
print(f'  AI: {len(ds.ai_samples)}')
print()

# Check samples
print('Sample human text:')
h = ds.human_samples[0]
print(f'  id={h.id}, domain={h.domain}')
print(f'  text: {h.text[:200]}...')
print()

print('Sample AI text:')
a = ds.ai_samples[0]
print(f'  id={a.id}, domain={a.domain}, generator={a.generator}')
print(f'  text: {a.text[:200]}...')
