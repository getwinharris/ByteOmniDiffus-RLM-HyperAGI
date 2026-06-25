"""
ByteOmniDiffus-RLM-HyperAGI package.

This package exposes the current local API/CLI adapter around the byte-native
runtime. Provider-shaped endpoint names are compatibility surfaces only; they
are not the project identity and do not imply a hosted model dependency.

Usage:
    python -m rlcodar_hyperagi.api --port 8000
"""

from rlcodar_hyperagi.api import app, main

__version__ = "1.0.0"
__all__ = ["app", "main"]
