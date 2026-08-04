"""Importing this package registers every task under the "task" registry key.

Add new tasks here so `configs/*.yaml` can name them without any import in the
training script.
"""

# Imported for the registration side effect; `__all__` re-exports them so no
# linter suppression is needed.
from qar.tasks import dev_toy, retriever

__all__ = ["dev_toy", "retriever"]
