#!/bin/bash

echo "🔽 Downloading ESL datasets for StealthRL..."

# Create directories
mkdir -p data/raw/ellipse
mkdir -p data/raw/icnale

# Download ELLIPSE via Kaggle
echo "📥 Downloading ELLIPSE from Kaggle..."
if command -v kaggle &> /dev/null; then
    cd data/raw/
    kaggle competitions download -c feedback-prize-english-language-learning
    unzip -q feedback-prize-english-language-learning.zip -d ellipse/
    rm feedback-prize-english-language-learning.zip
    cd ../..
    echo "✅ ELLIPSE downloaded: $(wc -l < data/raw/ellipse/train.csv) training essays"
else
    echo "❌ Kaggle CLI not installed. Install with: pip install kaggle"
    echo "   Then configure API token from https://www.kaggle.com/settings"
    echo ""
    echo "Alternative: Manual download from:"
    echo "   https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data"
fi

# ICNALE requires manual download
echo ""
echo "📋 ICNALE Download Instructions:"
echo "   1. Visit: https://language.sakura.ne.jp/icnale/"
echo "   2. Register for free academic access"
echo "   3. Download 'ICNALE Written Essays' package"
echo "   4. Extract to: data/raw/icnale/"
echo ""
echo "⏸️  Pausing for manual ICNALE download..."
echo "   Press Enter after completing ICNALE download, or Ctrl+C to skip"
read -p ""

# Verify downloads
echo ""
echo "📊 Verification:"
if [ -f "data/raw/ellipse/train.csv" ]; then
    echo "✅ ELLIPSE: $(wc -l < data/raw/ellipse/train.csv) essays"
else
    echo "❌ ELLIPSE: Not found"
fi

if [ -d "data/raw/icnale" ] && [ "$(ls -A data/raw/icnale)" ]; then
    echo "✅ ICNALE: Directory exists with files"
else
    echo "❌ ICNALE: Not downloaded yet"
fi

echo ""
echo "🎉 Download process complete!"
echo "Next steps:"
echo "   1. Run: python scripts/extract_esl_comprehensive.py"
echo "   2. Check: data/esl/ for extracted ESL data"
