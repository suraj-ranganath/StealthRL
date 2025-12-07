#!/usr/bin/env python3
"""
Pre-flight check for TASK 3: RL Training.

Verifies all prerequisites are met before starting training:
- Tinker API key configured
- Training data exists
- Detectors working
- Configs valid
- Required packages installed

Usage:
    python task3_rl_training/scripts/preflight_check.py
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple
import subprocess

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print section header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_check(name: str, passed: bool, details: str = ""):
    """Print check result."""
    status = f"{Colors.GREEN}✓{Colors.END}" if passed else f"{Colors.RED}✗{Colors.END}"
    print(f"{status} {name}")
    if details:
        print(f"  {Colors.YELLOW}→{Colors.END} {details}")

def check_env_file() -> Tuple[bool, str]:
    """Check if .env file exists with Tinker API key."""
    env_path = Path(".env")

    if not env_path.exists():
        return False, ".env file not found. Copy .env.example and add your Tinker API key."

    with open(env_path, 'r') as f:
        content = f.read()
        if 'TINKER_API_KEY' not in content:
            return False, "TINKER_API_KEY not found in .env file."

        # Check if key is set (not the example value)
        for line in content.split('\n'):
            if line.startswith('TINKER_API_KEY='):
                key = line.split('=', 1)[1].strip()
                if 'your_tinker_api_key_here' in key or not key:
                    return False, "TINKER_API_KEY is not set (still has placeholder value)."
                return True, f"API key found (starts with '{key[:10]}...')"

    return False, "Could not parse TINKER_API_KEY from .env file."

def check_training_data() -> Tuple[bool, str]:
    """Check if training data exists."""
    train_path = Path("data/tinker/train.jsonl")
    test_path = Path("data/tinker/test.jsonl")

    if not train_path.exists():
        return False, "data/tinker/train.jsonl not found. Run TASK 2 first."

    if not test_path.exists():
        return False, "data/tinker/test.jsonl not found. Run TASK 2 first."

    # Count samples
    with open(train_path, 'r') as f:
        train_count = sum(1 for line in f if line.strip())

    with open(test_path, 'r') as f:
        test_count = sum(1 for line in f if line.strip())

    if train_count < 100:
        return False, f"Only {train_count} training samples (need at least 100)."

    return True, f"Train: {train_count} samples, Test: {test_count} samples"

def check_esl_data() -> Tuple[bool, str]:
    """Check if ESL evaluation data exists."""
    dev_path = Path("data/processed/esl_native_dev.jsonl")
    test_path = Path("data/processed/esl_native_test.jsonl")

    if not dev_path.exists() or not test_path.exists():
        return False, "ESL evaluation data not found. Run TASK 2 step 5."

    return True, "ESL evaluation splits ready"

def check_configs() -> Tuple[bool, str]:
    """Check if training configs exist."""
    configs = [
        "configs/tinker_stealthrl.yaml",
        "configs/tinker_transfer_in_ensemble.yaml",
        "configs/ablations/detector_only.yaml",
    ]

    missing = []
    for config in configs:
        if not Path(config).exists():
            missing.append(config)

    if missing:
        return False, f"Missing configs: {', '.join(missing)}"

    return True, "All training configs present"

def check_python_packages() -> Tuple[bool, str]:
    """Check if required Python packages are installed."""
    required = [
        "torch",
        "transformers",
        "sentence-transformers",
        "tinker",
        "pyyaml",
    ]

    missing = []
    for package in required:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if missing:
        return False, f"Missing packages: {', '.join(missing)}. Run: pip install -r requirements.txt"

    return True, "All required packages installed"

def check_detectors() -> Tuple[bool, str]:
    """Check if detector implementations exist."""
    detector_path = Path("stealthrl/tinker/detectors.py")

    if not detector_path.exists():
        return False, "stealthrl/tinker/detectors.py not found."

    # Check if TASK 1 was completed
    task1_path = Path("task1_detector_implementation")
    if not task1_path.exists():
        return False, "TASK 1 not completed. Set up detectors first."

    return True, "Detector implementations ready (TASK 1 completed)"

def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space."""
    try:
        stat = os.statvfs('.')
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

        if available_gb < 5:
            return False, f"Only {available_gb:.1f}GB available (need at least 5GB for checkpoints)."

        return True, f"{available_gb:.1f}GB available"
    except:
        return True, "Could not check disk space"

def check_outputs_dir() -> Tuple[bool, str]:
    """Check if outputs directory exists."""
    outputs_path = Path("outputs")

    if not outputs_path.exists():
        outputs_path.mkdir(parents=True, exist_ok=True)
        return True, "Created outputs/ directory"

    return True, "outputs/ directory exists"

def main():
    """Run all pre-flight checks."""
    print(f"\n{Colors.BOLD}StealthRL TASK 3: Pre-Flight Check{Colors.END}")
    print(f"{Colors.YELLOW}Verifying all prerequisites for RL training...{Colors.END}")

    checks = []

    # Environment
    print_header("1. Environment Setup")
    passed, details = check_env_file()
    print_check("Tinker API Key (.env)", passed, details)
    checks.append(passed)

    passed, details = check_python_packages()
    print_check("Python Packages", passed, details)
    checks.append(passed)

    passed, details = check_disk_space()
    print_check("Disk Space", passed, details)
    checks.append(passed)

    # Data
    print_header("2. Training Data (TASK 2)")
    passed, details = check_training_data()
    print_check("Training Data (data/tinker/)", passed, details)
    checks.append(passed)

    passed, details = check_esl_data()
    print_check("ESL Evaluation Data", passed, details)
    checks.append(passed)

    # Code
    print_header("3. Code & Configs")
    passed, details = check_detectors()
    print_check("Detectors (TASK 1)", passed, details)
    checks.append(passed)

    passed, details = check_configs()
    print_check("Training Configs", passed, details)
    checks.append(passed)

    passed, details = check_outputs_dir()
    print_check("Outputs Directory", passed, details)
    checks.append(passed)

    # Summary
    print_header("Summary")

    total = len(checks)
    passed_count = sum(checks)

    if all(checks):
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED ({passed_count}/{total}){Colors.END}")
        print(f"\n{Colors.GREEN}You're ready to start TASK 3: RL Training!{Colors.END}")
        print(f"\nNext step: {Colors.BLUE}python -m stealthrl.tinker.train --config configs/tinker_stealthrl.yaml --data-path data/tinker{Colors.END}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ SOME CHECKS FAILED ({passed_count}/{total} passed){Colors.END}")
        print(f"\n{Colors.YELLOW}Please fix the issues above before starting training.{Colors.END}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
