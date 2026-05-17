from peft import LoraConfig

from style_llm_train.lora import build_lora_config


def test_lora_config_uses_paper_defaults() -> None:
    cfg = build_lora_config()
    assert isinstance(cfg, LoraConfig)
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert set(cfg.target_modules) == expected_targets


def test_lora_config_overrides() -> None:
    cfg = build_lora_config(r=8, alpha=16, dropout=0.1)
    assert cfg.r == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.1
