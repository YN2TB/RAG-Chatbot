import pytest
import yaml

from qar.config import RunConfig, apply_overrides, load_config


def test_loads_base_config():
    cfg = load_config("configs/base.yaml")
    assert isinstance(cfg, RunConfig)
    assert cfg.optim.scheduler == "cosine"
    assert cfg.data.train_path.endswith("train-qar.jsonl")


def test_base_inheritance_overrides_only_named_fields():
    cfg = load_config("configs/dev.yaml")
    assert cfg.name == "dev"
    assert cfg.train.max_steps == 300  # from dev.yaml
    assert cfg.optim.lr == pytest.approx(3.0e-4)  # inherited from base.yaml


def test_cli_overrides_are_typed_not_strings():
    cfg = load_config("configs/dev.yaml", ["optim.lr=1e-5", "train.amp=off", "data.batch_size=8"])
    assert cfg.optim.lr == pytest.approx(1e-5)
    assert isinstance(cfg.optim.lr, float)
    assert cfg.train.amp == "off"
    assert cfg.data.batch_size == 8


def test_yaml11_bool_words_do_not_corrupt_string_fields():
    """`off`/`no`/`yes` are YAML 1.1 booleans; they must not leak into str fields."""
    cfg = load_config("configs/base.yaml", ["train.amp=off"])
    assert cfg.train.amp == "off"
    cfg = load_config("configs/base.yaml", ["model.pooling=no"])
    assert cfg.model.pooling == "no"


def test_bool_fields_still_accept_bool_words():
    for text, expected in [("off", False), ("no", False), ("true", True), ("on", True)]:
        cfg = load_config("configs/base.yaml", [f"train.compile={text}"])
        assert cfg.train.compile is expected, text


def test_optional_field_accepts_null():
    cfg = load_config("configs/base.yaml", ["data.train_subset=null"])
    assert cfg.data.train_subset is None
    cfg = load_config("configs/base.yaml", ["data.train_subset=50000"])
    assert cfg.data.train_subset == 50_000


def test_unknown_key_is_rejected():
    """A typo in an ablation config must fail loudly, not silently do nothing."""
    with pytest.raises(KeyError, match="optim"):
        load_config("configs/base.yaml", ["optim.learning_rate=1e-4"])


def test_roundtrip_through_yaml(tmp_path):
    cfg = load_config("configs/dev.yaml", ["loss.temperature=0.02"])
    path = tmp_path / "config.yaml"
    cfg.save(path)
    reloaded = load_config(path)
    assert reloaded.to_dict() == cfg.to_dict()


def test_apply_overrides_builds_nested_dicts():
    data = apply_overrides({}, ["a.b.c=1"])
    assert data == {"a": {"b": {"c": 1}}}
    assert yaml.safe_dump(data)


def test_run_dir_derives_from_name():
    cfg = load_config("configs/base.yaml", ["name=exp1", "out_dir=runs"])
    assert cfg.run_dir.as_posix() == "runs/exp1"
