"""
Entry point for running eval as a module.

Usage:
    python -m eval --datasets mage --methods m0 m1 --detectors roberta fast_detectgpt
"""

from .runner import main

if __name__ == "__main__":
    main()
