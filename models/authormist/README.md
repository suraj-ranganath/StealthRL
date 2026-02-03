# AuthorMist (Ollama) Setup

This directory holds an optional AuthorMist baseline for Ollama-based inference.

**Files**
```
models/authormist/
├── Modelfile
├── README.md
└── <your-authormist>.gguf
```

**Setup**
1. Place your GGUF file in this directory.
2. Update the `FROM` line in `Modelfile` to point to that file.
3. Create the Ollama model:
   ```bash
   cd /path/to/StealthRL
   ollama create authormist -f models/authormist/Modelfile
   ```
4. Test the model:
   ```bash
   ollama run authormist "Please paraphrase the following text to make it more human-like."
   ```

**Troubleshooting**
- If the model is not listed by `ollama list`, re-run the create command with an absolute path to `Modelfile`.
