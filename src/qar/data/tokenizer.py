"""Byte-level BPE trained on this corpus, not a borrowed vocabulary.

The DL report's retriever is from scratch, and a from-scratch encoder deserves a
from-scratch vocabulary: a generic English BPE shatters model numbers, dimensions
and product names ("24-105mm", "B009B0MZ8U") into character soup, which lengthens
every sequence and dilutes exactly the tokens a product question turns on.

Byte-level means no token is ever unrepresentable, so `[UNK]` never fires and the
review text's stray unicode costs nothing.

**Trained on the train split only.** Fitting the vocabulary on val or test text
would leak their token distribution into the model's input representation --
small, but free to avoid.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD = "[PAD]"
SPECIAL_TOKENS = [PAD, "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def train_tokenizer(texts: Iterable[str], vocab_size: int, path: str | Path) -> Tokenizer:
    """Fit a byte-level BPE and save it as a single self-contained JSON file."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no tokenizer at {path}; run scripts/prepare_data.py before training"
        )
    return Tokenizer.from_file(str(path))


def pad_id(tokenizer: Tokenizer) -> int:
    """Id of the padding token, which the collator needs and the model must mask."""
    token_id = tokenizer.token_to_id(PAD)
    if token_id is None:
        raise ValueError(f"tokenizer has no {PAD} token; it was not built by this project")
    return token_id
