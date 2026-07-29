"""Make `src/` importable and pin the working directory.

Tests refer to `configs/*.yaml` by repo-relative path, so they must run from the
project root regardless of where pytest was invoked.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
