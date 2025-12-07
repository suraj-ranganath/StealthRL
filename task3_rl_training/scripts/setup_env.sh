#!/bin/bash
# Setup .env file for TASK 3: RL Training
# This script helps you configure your Tinker API key

set -e

echo "=================================================="
echo "StealthRL TASK 3: Environment Setup"
echo "=================================================="
echo ""

# Check if .env already exists
if [ -f ".env" ]; then
    echo "✓ .env file already exists"
    echo ""
    echo "Current TINKER_API_KEY:"
    grep "TINKER_API_KEY" .env || echo "  (not found)"
    echo ""
    read -p "Do you want to update it? (y/N): " update
    if [[ ! $update =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file"
        exit 0
    fi
fi

# Copy example
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ Created .env from .env.example"
    else
        echo "✗ .env.example not found!"
        exit 1
    fi
fi

echo ""
echo "=================================================="
echo "Tinker API Key Setup"
echo "=================================================="
echo ""
echo "To get your Tinker API key:"
echo "1. Go to: https://tinker.thinkingmachines.ai/"
echo "2. Sign in or create an account"
echo "3. Navigate to Settings → API Keys"
echo "4. Create a new API key or copy existing one"
echo ""
echo "Your API key should start with 'tk-'"
echo ""

# Prompt for API key
read -p "Enter your Tinker API key: " api_key

# Validate format
if [[ ! $api_key =~ ^tk- ]]; then
    echo ""
    echo "⚠️  Warning: API key should start with 'tk-'"
    read -p "Continue anyway? (y/N): " continue
    if [[ ! $continue =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 1
    fi
fi

# Update .env file
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|TINKER_API_KEY=.*|TINKER_API_KEY=$api_key|" .env
else
    # Linux
    sed -i "s|TINKER_API_KEY=.*|TINKER_API_KEY=$api_key|" .env
fi

echo ""
echo "✓ API key saved to .env"
echo ""

# Verify
echo "Verifying setup..."
if grep -q "TINKER_API_KEY=$api_key" .env; then
    echo "✓ API key successfully configured"
else
    echo "✗ Failed to update .env file"
    exit 1
fi

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Run pre-flight check: python task3_rl_training/scripts/preflight_check.py"
echo "  2. Start training: See task3_rl_training/QUICK_START.md"
echo ""
