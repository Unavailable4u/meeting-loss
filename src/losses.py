"""Loss functions live in src/model.py (kept together with the model for now).

This module re-exports them so the layout in docs/GUIDE.md, Section 3 stays valid.
If the losses grow, move them here and update METHOD.md and the imports in the tests.
"""
from src.model import compute_losses  # noqa: F401
