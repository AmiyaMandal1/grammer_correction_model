from __future__ import annotations

from typing import Any

import math

import torch
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class NanAbortCallback(TrainerCallback):
    """Stop training if loss becomes NaN/Inf; always print log dicts."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        print(f"[step {state.global_step}] {logs}", flush=True)
        loss = logs.get("loss")
        if loss is None:
            return
        if math.isnan(loss) or math.isinf(loss):
            print(f"[NanAbortCallback] loss={loss} at step {state.global_step}; aborting", flush=True)
            control.should_training_stop = True


class GradGuardCallback(TrainerCallback):
    """Sanitise gradients before the optimizer step:
      1. zero NaN/Inf entries (a single bad activation should not corrupt
         the whole model, and `clip_grad_norm_` does corrupt it because
         it multiplies every grad by 1/NaN_total_norm = NaN);
      2. clamp finite grads to [-clip, clip] per element so we still get
         the regularisation benefit `max_grad_norm` would have provided.
    """

    def __init__(self, clip: float = 1.0, log_every: int = 100) -> None:
        self.clip = clip
        self.log_every = log_every
        self.zeroed_steps = 0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        had_bad = False
        for p in model.parameters():
            if p.grad is None:
                continue
            mask = torch.isnan(p.grad) | torch.isinf(p.grad)
            if mask.any():
                p.grad[mask] = 0.0
                had_bad = True
            if self.clip > 0:
                p.grad.clamp_(min=-self.clip, max=self.clip)
        if had_bad:
            self.zeroed_steps += 1
            if self.zeroed_steps == 1 or self.zeroed_steps % self.log_every == 0:
                print(
                    f"[GradGuard] step {state.global_step}: zeroed NaN/Inf grads "
                    f"(total bad steps={self.zeroed_steps})",
                    flush=True,
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
        logging_strategy="steps",
        logging_steps=100,
        report_to=[],
        remove_unused_columns=False,
        max_grad_norm=0.0,  # disable HF clip — a single NaN poisons the total norm and
                            # thus every gradient via clip_grad_norm_; GradGuard handles it
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
        callbacks=[NanAbortCallback(), GradGuardCallback()],
    )

    out = trainer.train()
    trainer.save_model(str(cfg.output_dir))
    return dict(out.metrics)
