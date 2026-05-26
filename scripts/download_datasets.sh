#!/bin/bash
# Download datasets for StealthRL from original sources

set -e

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "Downloading datasets for StealthRL..."

# DetectRL benchmark
echo ""
echo "=== Downloading DetectRL benchmark ==="
cd "$DATA_DIR"
if [ ! -d "DetectRL" ]; then
    git clone https://github.com/NLP2CT/DetectRL.git
    echo "DetectRL downloaded successfully"
else
    echo "DetectRL already exists, skipping"
fi
cd ../..

# ai-detection-paraphrases (DIPPER benchmark)
echo ""
echo "=== Downloading ai-detection-paraphrases (DIPPER) ==="
cd "$DATA_DIR"
if [ ! -d "ai-detection-paraphrases" ]; then
    git clone https://github.com/martiansideofthemoon/ai-detection-paraphrases.git
    echo "ai-detection-paraphrases downloaded successfully"
else
    echo "ai-detection-paraphrases already exists, skipping"
fi
cd ../..

# ChatGPT-Detector-Bias (ESL vs native data)
echo ""
echo "=== Downloading ChatGPT-Detector-Bias ==="
cd "$DATA_DIR"
if [ ! -d "ChatGPT-Detector-Bias" ]; then
    git clone https://github.com/Weixin-Liang/ChatGPT-Detector-Bias.git
    echo "ChatGPT-Detector-Bias downloaded successfully"
else
    echo "ChatGPT-Detector-Bias already exists, skipping"
fi
cd ../..

# Ghostbuster data
echo ""
echo "=== Downloading Ghostbuster data ==="
cd "$DATA_DIR"
if [ ! -d "ghostbuster" ]; then
    git clone https://github.com/vivek3141/ghostbuster.git
    echo "Ghostbuster downloaded successfully"
else
    echo "Ghostbuster already exists, skipping"
fi
cd ../..

# Human Detectors data
echo ""
echo "=== Downloading Human Detectors data ==="
cd "$DATA_DIR"
if [ ! -d "human_detectors" ]; then
    git clone https://github.com/jenna-russell/human_detectors.git
    echo "Human Detectors downloaded successfully"
else
    echo "Human Detectors already exists, skipping"
fi
cd ../..

echo ""
echo "=== Dataset downloads complete! ==="
echo ""
echo "Downloaded datasets:"
echo "  - DetectRL: $DATA_DIR/DetectRL"
echo "  - ai-detection-paraphrases: $DATA_DIR/ai-detection-paraphrases"
echo "  - ChatGPT-Detector-Bias: $DATA_DIR/ChatGPT-Detector-Bias"
echo "  - Ghostbuster: $DATA_DIR/ghostbuster"
echo "  - Human Detectors: $DATA_DIR/human_detectors"
echo ""
echo "Please check data licenses and citations in each subdirectory."
echo "Refer to the original papers and repositories for proper attribution."
