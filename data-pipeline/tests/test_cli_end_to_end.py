import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from data_pipeline.cli import app


def test_build_gec_only_from_local_m2(tmp_path: Path, fixtures_dir: Path) -> None:
    # Place a tiny BEA-2019 train.m2 into raw dir.
    raw = tmp_path / "data" / "raw" / "bea2019"
    raw.mkdir(parents=True)
    shutil.copy(fixtures_dir / "tiny.m2", raw / "train.m2")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-gec",
            "--root",
            str(tmp_path),
            "--sources",
            "bea2019",
            "--split",
            "train",
        ],
    )
    assert result.exit_code == 0, result.output

    out_jsonl = tmp_path / "data" / "processed" / "gec_tagger.jsonl"
    out_manifest = tmp_path / "data" / "processed" / "manifest_gec.json"
    assert out_jsonl.exists()
    assert out_manifest.exists()

    lines = out_jsonl.read_text().strip().splitlines()
    assert len(lines) == 3
    rec = json.loads(lines[0])
    assert rec["tokens"][0] == "He"
    assert "$REPLACE_goes" in rec["tags"] or rec["tags"][1] == "$REPLACE_goes"

    m = json.loads(out_manifest.read_text())
    assert "gec_tagger" in m["artifacts"]
    assert m["artifacts"]["gec_tagger"]["row_count"] == 3
    assert m["sources"]["bea2019"]["row_count"] == 3
    assert m["sources"]["bea2019"]["retained"] == 3
    # No --eval-sources provided, so leakage check was not run.
    assert m["leakage"]["passed"] is None
