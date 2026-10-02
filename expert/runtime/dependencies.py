"""runtime.dependencies phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~expert.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from expert.runtime.context import RuntimeState, export_runtime_locals


def register_dependencies(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # Runtime imports used by the training, checkpoint, and test components.
    import os
    import sys
    import json
    import random
    import re
    import math
    import logging
    import itertools
    import copy
    import hashlib
    from collections import Counter, defaultdict
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Tuple

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
    from PIL import Image
    from tqdm import tqdm

    try:
        import wandb
    except ImportError:
        wandb = None

    import transformers
    from transformers import (
        AutoProcessor,
        get_cosine_schedule_with_warmup,
    )

    print('Python:', sys.version)
    print('transformers version:', transformers.__version__)
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        print('Qwen3_5ForConditionalGeneration: available')
    except Exception as exc:
        print('Qwen3_5ForConditionalGeneration: unavailable ->', exc)

    return export_runtime_locals(state, locals())
