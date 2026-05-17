from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from gec_tagger_train.tag_vocab import TagVocab
from gec_tagger_train.tokenization import align_tags_to_subwords


class GECTaggerDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        jsonl: Path,
        tokenizer: Any,
        vocab: TagVocab,
        max_length: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length
        self._items: list[dict[str, torch.Tensor]] = []
        self._load(jsonl)

    def _load(self, jsonl: Path) -> None:
        pad_id = self.vocab.id_of("$PAD")
        unk_id = self.vocab.id_of("$UNK")
        with jsonl.open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tokens: list[str] = rec["tokens"]
                tags: list[str] = rec["tags"]
                if len(tokens) != len(tags):
                    raise ValueError(
                        f"length mismatch: {len(tokens)} tokens vs {len(tags)} tags"
                    )
                enc = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    truncation=False,
                    padding=False,
                    return_tensors=None,
                )
                if len(enc["input_ids"]) > self.max_length:
                    continue
                labels = align_tags_to_subwords(
                    enc, tags, pad_id=pad_id, unk_id=unk_id, vocab_lookup=self.vocab.id_of
                )
                self._items.append(
                    {
                        "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(
                            enc["attention_mask"], dtype=torch.long
                        ),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._items[idx]
