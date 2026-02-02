#!/usr/bin/env python3
"""
Pre-flight check for StealthRL evaluation pipeline.

Run this before starting a research run to verify all components work.

Usage:
    python scripts/preflight_check.py
    python scripts/preflight_check.py --full  # Include slow checks
"""

import argparse
import sys
import json
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def check(name: str, success: bool, message: str = ""):
    """Print check result."""
    if success:
        print(f"  {GREEN}✓{RESET} {name}")
    else:
        print(f"  {RED}✗{RESET} {name}: {message}")
    return success


def section(title: str):
    """Print section header."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


def check_imports():
    """Check required Python packages."""
    section("1. Python Dependencies")
    
    all_ok = True
    
    # Core packages
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("requests", "Requests"),
    ]
    
    for pkg, name in packages:
        try:
            __import__(pkg)
            check(name, True)
        except ImportError:
            check(name, False, f"pip install {pkg}")
            all_ok = False
    
    # Optional but recommended
    optional = [
        ("peft", "PEFT (for LoRA)"),
        ("seaborn", "Seaborn (for plots)"),
        ("tabulate", "Tabulate (for tables)"),
    ]
    
    print(f"\n  {YELLOW}Optional packages:{RESET}")
    for pkg, name in optional:
        try:
            __import__(pkg)
            check(name, True)
        except ImportError:
            check(name, False, f"pip install {pkg}")
    
    return all_ok


def check_torch_device():
    """Check PyTorch device availability."""
    section("2. PyTorch Device")
    
    import torch
    
    print(f"  PyTorch version: {torch.__version__}")
    
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    
    check("MPS (Apple Silicon)", mps_available, "Not available")
    check("CUDA", cuda_available, "Not available")
    
    if mps_available:
        print(f"  {GREEN}→ Will use MPS acceleration{RESET}")
    elif cuda_available:
        print(f"  {GREEN}→ Will use CUDA acceleration{RESET}")
    else:
        print(f"  {YELLOW}→ Will use CPU (slower){RESET}")
    
    return True


def check_ollama():
    """Check Ollama availability and models."""
    section("3. Ollama")
    
    import requests
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        
        check("Ollama running", True)
        print(f"  Available models: {models}")
        
        # Check for required models
        has_qwen = any("qwen" in m.lower() for m in models)
        has_authormist = any("authormist" in m.lower() for m in models)
        
        check("Qwen3 (for M1)", has_qwen, "Run: ollama pull qwen3:4b-instruct")
        check("AuthorMist (for M4)", has_authormist, 
              "Setup: see models/authormist/README.md")
        
        return has_qwen
        
    except requests.exceptions.ConnectionError:
        check("Ollama running", False, "Start Ollama: ollama serve")
        return False
    except Exception as e:
        check("Ollama running", False, str(e))
        return False


def check_checkpoints():
    """Check for Tinker checkpoint files."""
    section("4. StealthRL Checkpoints")
    
    ckpt_dir = Path("checkpoints")
    
    if not ckpt_dir.exists():
        check("Checkpoint directory", False, "mkdir checkpoints")
        return False
    
    check("Checkpoint directory", True)
    
    json_files = list(ckpt_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "README.md"]
    
    if not json_files:
        print(f"  {YELLOW}No checkpoint JSON files found{RESET}")
        print(f"  Place your Tinker checkpoint JSON files in checkpoints/")
        return False
    
    print(f"  Found {len(json_files)} checkpoint file(s):")
    for f in json_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            model_id = data.get("model_id", "unknown")[:40]
            base = data.get("base_model", "unknown")
            print(f"    • {f.name}")
            print(f"      Model ID: {model_id}...")
            print(f"      Base: {base}")
        except Exception as e:
            print(f"    • {f.name} {RED}(invalid JSON: {e}){RESET}")
    
    return True


def check_authormist_gguf():
    """Check for AuthorMist GGUF file."""
    section("5. AuthorMist GGUF")
    
    authormist_dir = Path("models/authormist")
    
    if not authormist_dir.exists():
        check("AuthorMist directory", False, "mkdir -p models/authormist")
        return False
    
    check("AuthorMist directory", True)
    
    gguf_files = list(authormist_dir.glob("*.gguf"))
    
    if not gguf_files:
        print(f"  {YELLOW}No GGUF file found{RESET}")
        print(f"  Place your AuthorMist GGUF in models/authormist/")
        return False
    
    for f in gguf_files:
        size_gb = f.stat().st_size / (1024**3)
        print(f"    • {f.name} ({size_gb:.2f} GB)")
    
    # Check Modelfile
    modelfile = authormist_dir / "Modelfile"
    if modelfile.exists():
        check("Modelfile", True)
        
        # Check if FROM line is updated
        content = modelfile.read_text()
        if "authormist-imatrix.gguf" in content:
            print(f"  {YELLOW}⚠ Update Modelfile FROM line with actual GGUF filename{RESET}")
    else:
        check("Modelfile", False, "Missing Modelfile")
    
    return len(gguf_files) > 0


def check_eval_module():
    """Check eval module imports."""
    section("6. Eval Module")
    
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        from eval import (
            load_eval_dataset,
            get_detector,
            get_method,
            compute_detector_metrics,
        )
        check("eval module imports", True)
        return True
    except Exception as e:
        check("eval module imports", False, str(e))
        return False


def test_detector(name: str, quick: bool = True):
    """Test a detector with sample text."""
    from eval.detectors import get_detector
    
    try:
        detector = get_detector(name)
        detector.load()
        
        test_text = "This is a test sentence for the detector."
        score = detector.get_scores(test_text)
        
        print(f"    Score on test text: {score:.4f}")
        return True
    except Exception as e:
        print(f"    {RED}Error: {e}{RESET}")
        return False


def test_method(name: str):
    """Test a method with sample text."""
    from eval.methods import get_method
    
    try:
        method = get_method(name)
        method.load()
        
        test_text = "Artificial intelligence has transformed modern technology."
        result = method.attack(test_text, n_candidates=1)
        
        print(f"    Output: {result.text[:80]}...")
        return True
    except Exception as e:
        print(f"    {RED}Error: {e}{RESET}")
        return False


def run_full_checks():
    """Run slow detector/method checks."""
    section("7. Detector Tests (slow)")
    
    print("  Testing RoBERTa detector...")
    test_detector("roberta")
    
    print("  Testing Fast-DetectGPT...")
    test_detector("fast_detectgpt")
    
    section("8. Method Tests (slow)")
    
    print("  Testing M0 (no attack)...")
    test_method("m0")
    
    print("  Testing M1 (simple paraphrase)...")
    test_method("m1")
    
    print("  Testing M4 (AuthorMist via Ollama)...")
    test_method("m4")


def main():
    parser = argparse.ArgumentParser(description="Pre-flight check for StealthRL evaluation")
    parser.add_argument("--full", action="store_true", help="Include slow detector/method tests")
    args = parser.parse_args()
    
    print(f"\n{BLUE}StealthRL Evaluation Pre-flight Check{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    results = []
    
    results.append(("Python Dependencies", check_imports()))
    results.append(("PyTorch Device", check_torch_device()))
    results.append(("Ollama", check_ollama()))
    results.append(("Checkpoints", check_checkpoints()))
    results.append(("AuthorMist GGUF", check_authormist_gguf()))
    results.append(("Eval Module", check_eval_module()))
    
    if args.full:
        run_full_checks()
    
    # Summary
    section("Summary")
    
    all_ok = all(r[1] for r in results)
    
    for name, ok in results:
        status = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {status} {name}")
    
    if all_ok:
        print(f"\n{GREEN}All checks passed! Ready for research run.{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}Some checks failed. Fix issues above before running.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
