from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_SPECIAL_TAGS: tuple[str, ...] = ("$PAD", "$UNK", "$KEEP")


@dataclass
class TagVocab:
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._tag_to_id: dict[str, int] = {t: i for i, t in enumerate(self.tags)}

    @property
    def special_tags(self) -> list[str]:
        return [t for t in self.tags if t in _SPECIAL_TAGS]

    def id_of(self, tag: str) -> int:
        return self._tag_to_id.get(tag, self._tag_to_id["$UNK"])

    def __len__(self) -> int:
        return len(self.tags)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"tags": self.tags}, indent=2))

    @classmethod
    def load(cls, path: Path) -> TagVocab:
        payload = json.loads(path.read_text())
        return cls(tags=list(payload["tags"]))


def build_tag_vocab(*, jsonl: Path, min_count: int, out: Path) -> TagVocab:
    counts: Counter[str] = Counter()
    with jsonl.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for tag in rec["tags"]:
                if tag in _SPECIAL_TAGS:
                    continue
                counts[tag] += 1
    kept = [t for t, c in counts.most_common() if c >= min_count]
    vocab = TagVocab(tags=list(_SPECIAL_TAGS) + kept)
    vocab.save(out)
    return vocab
