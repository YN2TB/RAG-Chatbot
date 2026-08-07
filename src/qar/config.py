"""Typed, file-driven configuration.

Every run is fully described by one YAML file plus explicit CLI overrides, and the
resolved config is snapshotted into the run directory. That is what makes ~20
ablation runs reproducible and comparable after the fact.

    cfg = load_config("configs/dev.yaml", ["optim.lr=1e-4", "loss.temperature=0.02"])
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cache
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass
class DataConfig:
    train_path: str = "train-qar.jsonl"
    val_path: str = "val-qar.jsonl"
    # The upstream corpus ships its own product-disjoint test file. When it is
    # present the validation file stays whole; set this to null to fall back to
    # carving test out of validation by hashed asin (`prepare.test_fraction`).
    test_path: str | None = "test-qar_all.jsonl"
    processed_dir: str = "data/processed"
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 0  # Windows spawn overhead makes >0 a measured decision, not a default
    max_query_len: int = 64
    max_doc_len: int = 128
    train_subset: int | None = None  # None = all pairs; set for the data-scaling curve
    # Evaluation runs at every `train.eval_every`, so a full 43k-row pass would cost
    # more than the training between passes. A fixed prefix keeps the curve cheap and
    # comparable across steps and runs; the headline number comes from
    # scripts/evaluate_retrieval.py over the whole split, not from here.
    val_subset: int | None = 4096
    # Guards against the false-negative problem: ~54k train rows share a question
    # string with another row, so naive in-batch negatives punish correct matches.
    dedup_questions_in_batch: bool = True


@dataclass
class PrepareConfig:
    """Offline corpus preparation. Read by `scripts/prepare_data.py`, not by training.

    AmazonQA gives no snippet-level relevance label: a row knows its answers but not
    which review snippet supports them. The positive is therefore chosen by distant
    supervision, and `selector` is the knob that decides how -- which makes it an
    ablation axis for the DL report, not an implementation detail.
    """

    selector: str = "answer_overlap"  # answer_overlap | answer_recall | first
    min_positive_score: float = 0.10  # below this the row has no trustworthy positive
    min_snippet_tokens: int = 5  # drops "Great!"-style fragments from the pool
    max_snippets: int = 32  # bounds the pool; the median row has ~9
    # Only consulted when `data.test_path` is null. Carving test out of validation
    # by hashed asin was the fallback before the upstream test file was available.
    test_fraction: float = 0.5  # of the *validation* file, split by asin
    # Deliberately not `seed`: changing a run's seed must never reshuffle the split.
    split_seed: int = 20260731
    tokenizer_sample_docs: int = 400_000  # train-split texts sampled to fit the BPE
    max_rows: int | None = None  # None = whole corpus; set for a smoke run


@dataclass
class RetrievalConfig:
    """Untrained baselines the learned retriever has to beat.

    Ranking is scoped to one product's snippet pool (~9 candidates), so these
    numbers are not comparable with a global-index evaluation and must never share
    a table column with one.
    """

    # bm25_global needs scripts/build_idf.py to have run; it is in the default table
    # because pool-local `bm25` measurably under-performs plain `overlap`, so quoting
    # `bm25` alone would understate what a lexical baseline can do.
    baselines: list[str] = field(
        default_factory=lambda: [
            "random", "first", "overlap", "bm25", "bm25_global", "bm25_noidf",
        ]
    )
    split: str = "val"  # val | test | train
    ks: list[int] = field(default_factory=lambda: [1, 3, 5])
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    idf_file: str = "idf.json"  # inside data.processed_dir; built by scripts/build_idf.py
    idf_min_df: int = 5
    # Pools are shuffled before scoring so that all-zero rows land at chance rather
    # than defaulting to snippet 0 and silently importing the `first` baseline.
    tie_break_seed: int = 12345
    max_rows: int | None = None
    # Required by the `dense` retriever and ignored by every other one. The
    # architecture is rebuilt from the checkpoint's own snapshotted config, so this
    # path alone determines what is scored.
    checkpoint: str | None = None
    # Rows handed to `Retriever.score_batch` at once. Irrelevant to the lexical
    # baselines, which ignore the grouping; it is what makes `dense` practical to
    # run across an ablation grid.
    batch_rows: int = 256
    # Texts per encoder forward pass, after length sorting. Bounds VRAM independently
    # of `batch_rows`: at 256 rows a single unbounded pass allocated 7.7 GiB of 8.1 on
    # the 5060 and thrashed. Purely a performance knob — it cannot change a score.
    encode_batch: int = 256
    # Basename for `runs/_baselines/<name>.{json,md}`. Defaults to the split. A sweep
    # sets it per cell, so twenty checkpoints all scored as "dense" do not overwrite
    # each other in one shared table.
    out_name: str | None = None


@dataclass
class ModelConfig:
    name: str = "dev_toy"
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1536
    dropout: float = 0.1
    vocab_size: int = 32000
    max_len: int = 128
    pooling: str = "mean"  # mean | cls
    # One tower or two. Two lets each side specialise (questions and review prose are
    # different registers); one halves the parameters and gives every gradient twice
    # the data. Which wins from scratch on 700k pairs is an ablation, not a given.
    share_encoder: bool = False
    pretrained: str | None = None  # None = from scratch (the DL-report default)


@dataclass
class LossConfig:
    temperature: float = 0.05
    answerable_weight: float = 0.0  # lambda on the multi-task answerability head
    # Same-product snippets added per row, on top of the in-batch negatives. They need
    # no mining pass -- the product's pool is already in every record -- and they are
    # the negatives that matter: an in-batch negative comes from another product, so
    # rejecting it only requires recognising the topic.
    hard_negatives: int = 0


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    grad_clip: float = 1.0
    scheduler: str = "cosine"  # cosine | linear | constant
    warmup_ratio: float = 0.06


@dataclass
class TrainConfig:
    max_steps: int = 10_000
    grad_accum: int = 1
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1000
    # Wall-clock floor on fault tolerance, independent of throughput. `save_every` is
    # a step count, so the real time between checkpoints swings with steps/s — 2000
    # steps is 7 minutes at 4.5 steps/s and 19 at 1.75. This bounds how much work an
    # interruption can destroy. 0 disables it and leaves only the step cadence.
    save_every_minutes: float = 10.0
    keep_last: int = 2
    monitor: str = "val/loss"
    monitor_mode: str = "min"  # min | max
    early_stop_patience: int = 0  # evaluations without improvement; 0 disables
    amp: str = "bf16"  # bf16 | fp16 | off
    grad_checkpoint: bool = False  # trades compute for the large batches InfoNCE wants
    compile: bool = False


@dataclass
class RunConfig:
    name: str = "dev"
    task: str = "dev_toy"
    seed: int = 42
    deterministic: bool = False
    device: str = "auto"
    out_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    prepare: PrepareConfig = field(default_factory=PrepareConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _is_optional(tp: Any) -> bool:
    return get_origin(tp) is not None and type(None) in get_args(tp)


def _unwrap_optional(tp: Any) -> Any:
    return next((a for a in get_args(tp) if a is not type(None)), str)


def _coerce(value: Any, tp: Any) -> Any:
    """Coerce a YAML/CLI scalar into the annotated field type."""
    if value is None:
        return None
    if _is_optional(tp):
        tp = _unwrap_optional(tp)
    if tp is bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if tp in (int, float, str):
        return tp(value)
    return value


def _build(cls: type, data: dict[str, Any], path: str = "") -> Any:
    """Recursively instantiate a nested dataclass, rejecting unknown keys.

    Unknown keys are an error rather than a warning: a silently ignored typo in an
    ablation config means a run that looks like it tested something and did not.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = path.rstrip(".") or cls.__name__
        raise KeyError(f"unknown config key(s) under '{where}': {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        ftype = _resolve(cls, name)
        value = data[name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build(ftype, value, f"{path}{name}.")
        else:
            kwargs[name] = _coerce(value, ftype)
    return cls(**kwargs)


@cache
def _hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


def _resolve(cls: type, name: str) -> Any:
    """Field type, resolving the string annotations `from __future__` produces."""
    return _hints(cls).get(name, str)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML config, resolving a `_base_:` chain relative to the file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw
    return _deep_merge(_read_yaml((path.parent / base_ref).resolve()), raw)


def _parse_scalar(text: str) -> Any:
    """Parse an override value, defusing YAML 1.1's boolean spellings.

    YAML 1.1 reads `off`, `no` and `y` as booleans, which silently turns
    `train.amp=off` into the string "False" -- a run that reports AMP disabled
    while training in bf16. Only the canonical spellings stay boolean here; the
    rest survive as strings and are coerced by the target field's own type, so
    `train.compile=off` still lands as False on a genuine bool field.
    """
    raw = text.strip()
    value = yaml.safe_load(raw)
    if isinstance(value, bool) and raw.lower() not in {"true", "false"}:
        return raw
    return value


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `dotted.key=value` strings, parsing values as YAML scalars."""
    out = dict(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, _, value = item.partition("=")
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"cannot descend into scalar at '{key}'")
        node[parts[-1]] = _parse_scalar(value)
    return out


def config_from_dict(data: dict[str, Any]) -> RunConfig:
    """Rebuild a `RunConfig` from a plain dict, e.g. a checkpoint's snapshot.

    Same validation as `load_config`: unknown keys raise, scalars are coerced. This
    is how a saved run's own architecture is recovered when scoring its checkpoint,
    so that the config driving the evaluation cannot silently redefine the model.
    """
    return _build(RunConfig, data)


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    data = _read_yaml(Path(path).resolve())
    if overrides:
        data = apply_overrides(data, overrides)
    return _build(RunConfig, data)
