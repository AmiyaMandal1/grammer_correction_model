from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class TagKind(StrEnum):
    KEEP = "KEEP"
    DELETE = "DELETE"
    APPEND = "APPEND"
    REPLACE = "REPLACE"
    TRANSFORM_CASE = "TRANSFORM_CASE"
    TRANSFORM_VERB = "TRANSFORM_VERB"


class StyleKind(StrEnum):
    FORMAL = "formal"
    INFORMAL = "informal"
    CONCISE = "concise"
    SIMPLIFY = "simplify"
    DETOXIFY = "detoxify"


@dataclass(frozen=True, slots=True)
class EditTag:
    kind: TagKind
    value: str | None

    def to_str(self) -> str:
        if self.kind is TagKind.KEEP:
            return "$KEEP"
        if self.kind is TagKind.DELETE:
            return "$DELETE"
        if self.value is None:
            raise ValueError(f"tag {self.kind} requires a value")
        return f"${self.kind.value}_{self.value}"


@dataclass
class Pair:
    src: str
    tgt: str
    source: str
    split: Split
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "tgt": self.tgt,
            "source": self.source,
            "split": self.split.value,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Pair:
        return cls(
            src=d["src"],
            tgt=d["tgt"],
            source=d["source"],
            split=Split(d["split"]),
            meta=d.get("meta", {}),
        )
