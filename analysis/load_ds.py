from datasets import load_dataset

ds = load_dataset("yaful/MAGE")

## save to local disk
ds.save_to_disk("data/mage")