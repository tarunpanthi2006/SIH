"""
SatQuery-RS Backend — Entry Point
====================================
Convenience entry point for starting the model server.

Usage:
    python -m backend.main
    python -m backend.main --port 8100 --host 0.0.0.0
"""

from backend.serve import main

if __name__ == "__main__":
    main()
