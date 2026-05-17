from pathlib import Path
from unittest.mock import MagicMock, patch

from style_llm_train.merge import merge_adapter


def test_merge_adapter_invokes_peft_and_saves(tmp_out: Path) -> None:
    adapter_dir = tmp_out / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")
    out_dir = tmp_out / "merged"

    merged_model = MagicMock()
    merged_model.save_pretrained = MagicMock()
    peft_model = MagicMock()
    peft_model.merge_and_unload.return_value = merged_model

    fake_base = MagicMock()
    fake_tokenizer = MagicMock()

    base_patch = "style_llm_train.merge.AutoModelForCausalLM.from_pretrained"
    tok_patch = "style_llm_train.merge.AutoTokenizer.from_pretrained"
    peft_patch = "style_llm_train.merge.PeftModel.from_pretrained"
    with (
        patch(base_patch, return_value=fake_base) as m_base,
        patch(tok_patch, return_value=fake_tokenizer),
        patch(peft_patch, return_value=peft_model) as m_peft,
    ):
        merge_adapter(
            adapter_dir=adapter_dir,
            out_dir=out_dir,
            base_model="Qwen/Qwen2.5-3B-Instruct",
        )

    m_base.assert_called_once()
    m_peft.assert_called_once_with(fake_base, str(adapter_dir))
    peft_model.merge_and_unload.assert_called_once()
    merged_model.save_pretrained.assert_called_once_with(str(out_dir))
    fake_tokenizer.save_pretrained.assert_called_once_with(str(out_dir))
