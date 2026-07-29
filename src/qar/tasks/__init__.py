"""Importing this package registers every task under the "task" registry key.

Add new tasks here so `configs/*.yaml` can name them without any import in the
training script.
"""

from qar.tasks import dev_toy  # noqa: F401  (import for registration side effect)

__all__ = ["dev_toy"]
