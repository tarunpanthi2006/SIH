"""
RSVQA Evaluation Entry Point
================================
Thin wrapper that delegates to run_rsvqa.py.

Usage:
    python -m evaluation.rsvqa.evaluate --data <path> --max-samples 500
"""

from evaluation.rsvqa.run_rsvqa import evaluate_on_rsvqa, main

if __name__ == "__main__":
    main()
