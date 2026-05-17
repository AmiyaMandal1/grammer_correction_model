from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ManifestBuilder:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._sources: dict[str, dict[str, int]] = {}
        self._dedup: dict[str, int] = {}
        self._leakage_passed: bool | None = None  # None = not run

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def add_artifact(self, *, name: str, path: Path, row_count: int) -> None:
        self._artifacts[name] = {
            "path": str(path.relative_to(self.out_dir)),
            "row_count": row_count,
            "sha256": self._sha256(path),
            "bytes": path.stat().st_size,
        }

    def add_source(self, *, name: str, row_count: int, retained: int) -> None:
        self._sources[name] = {"row_count": row_count, "retained": retained}

    def set_dedup(self, *, removed_exact: int) -> None:
        self._dedup = {"removed_exact": removed_exact}

    def set_leakage_passed(self, passed: bool | None) -> None:
        """`None` means leakage check was not run for this build."""
        self._leakage_passed = passed

    def write(self) -> Path:
        out = self.out_dir / "manifest.json"
        payload: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "artifacts": self._artifacts,
            "sources": self._sources,
            "dedup": self._dedup,
            "leakage": {"passed": self._leakage_passed},
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return out
