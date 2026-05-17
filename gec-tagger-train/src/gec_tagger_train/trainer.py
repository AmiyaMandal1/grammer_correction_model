from __future__ import annotations

from typing import Any

from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from gec_tagger_train.config import StageConfig
from gec_tagger_train.model import DebertaTagger
from gec_tagger_train.tag_vocab import TagVocab


def run_training(
    *,
    cfg: StageConfig,
    dataset: Any,
    vocab: TagVocab,
    tokenizer: Any,
) -> dict[str, float]:
    model = DebertaTagger(num_tags=len(vocab), pad_id=vocab.id_of("$PAD"))

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        gradient_checkpointing=cfg.gradient_checkpointing,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        seed=cfg.seed,
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=vocab.id_of("$PAD")
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    out = trainer.train()
    trainer.save_model(str(cfg.output_dir))
    return dict(out.metrics)
