"""coldrank: reproducing the Li et al. (2026) cold-start reranking protocol.

See ../../plan.md for the implementation plan and ../../design.md for design decisions.
"""

import os

# The system Python on the dev machine has TensorFlow + Keras 3 installed, which
# breaks `transformers` imports. We only use PyTorch.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
