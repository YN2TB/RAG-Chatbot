"""Model architectures.

Importing this package registers every model under the "model" registry key, so
`model.name` in a config resolves without any direct import.
"""

from qar.models import biencoder  # noqa: F401  (import for registration side effect)
from qar.models.biencoder import BiEncoder
from qar.models.encoder import TextEncoder

__all__ = ["BiEncoder", "TextEncoder"]
