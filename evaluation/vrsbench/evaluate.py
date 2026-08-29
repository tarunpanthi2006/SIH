"""
VRSBench Evaluation Entry Point
==================================
Thin wrapper that delegates to run_vrsbench.py.

Usage:
    python -m evaluation.vrsbench.evaluate --max-samples 500
"""

from evaluation.vrsbench.run_vrsbench import evaluate_on_vrsbench, main

if __name__ == "__main__":
    main()
