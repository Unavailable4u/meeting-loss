"""Small helpers. UNTESTED starter code (see docs/RISKS.md R-10)."""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. Log the seed with every run (docs/GUIDE.md, Section 5)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
