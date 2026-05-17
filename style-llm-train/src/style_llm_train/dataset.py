from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def load_sft_records(jsonl: Path) -> Iterator[dict[str, Any]]:
    """Yield `{messages, meta}` records from a ChatML SFT JSONL file."""
    with jsonl.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            roles = [m.get("role") for m in msgs]
            if roles[:3] != ["system", "user", "assistant"]:
                continue
            yield rec
