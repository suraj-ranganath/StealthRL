# AuthorMist Setup for Ollama

AuthorMist (M4 baseline) uses Ollama for efficient GGUF inference on M4 Mac.

## Files to Place Here

```
models/authormist/
├── Modelfile                    # Ollama model configuration (already created)
├── README.md                    # This file
└── <your-authormist>.gguf       # YOUR GGUF FILE GOES HERE
```

## Setup Steps

### 1. Place your GGUF file
Copy your AuthorMist imatrix GGUF file to this directory:
```bash
cp /path/to/authormist-imatrix-Q4_K_M.gguf models/authormist/
```

### 2. Update the Modelfile
Edit `Modelfile` and update the `FROM` line with your actual filename:
```
FROM ./authormist-imatrix-Q4_K_M.gguf
```

### 3. Create the Ollama model
```bash
cd /Users/suraj/Desktop/StealthRL
ollama create authormist -f models/authormist/Modelfile
```

### 4. Test the model
```bash
ollama run authormist "Please paraphrase the following text to make it more human-like: The advancement of artificial intelligence has led to significant improvements in natural language processing capabilities."
```

### 5. Verify it's available
```bash
ollama list | grep authormist
```

## Troubleshooting

### Model not found
If `ollama list` doesn't show authormist:
```bash
# Re-create with full path
ollama create authormist -f /Users/suraj/Desktop/StealthRL/models/authormist/Modelfile
```

### Memory issues
For M4 Mac with 16GB RAM, Q4_K_M quantization is recommended.
If you have more RAM, Q5_K_M or Q6_K will give better quality.

### Testing in Python
```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "authormist",
        "prompt": "Paraphrase: The quick brown fox jumps over the lazy dog.",
        "stream": False,
    }
)
print(response.json()["response"])
```

## Expected GGUF File Names

Common naming patterns:
- `authormist-originality-Q4_K_M.gguf`
- `authormist-imatrix-Q4_K_M.gguf`
- `authormist-7B-Q4_K_M.gguf`

The exact name depends on where you downloaded it from.
