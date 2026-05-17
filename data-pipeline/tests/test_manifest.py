import json
from pathlib import Path

from data_pipeline.manifest import ManifestBuilder


def test_manifest_records_files_and_counts(tmp_path: Path) -> None:
    target = tmp_path / "file.jsonl"
    target.write_text('{"a":1}\n{"a":2}\n')
    mb = ManifestBuilder(out_dir=tmp_path)
    mb.add_artifact(name="gec_tagger", path=target, row_count=2)
    mb.add_source(name="bea2019", row_count=1000, retained=950)
    mb.set_dedup(removed_exact=12)
    mb.set_leakage_passed(True)
    out = mb.write()
    data = json.loads(out.read_text())
    assert data["artifacts"]["gec_tagger"]["row_count"] == 2
    assert "sha256" in data["artifacts"]["gec_tagger"]
    assert data["sources"]["bea2019"] == {"row_count": 1000, "retained": 950}
    assert data["dedup"]["removed_exact"] == 12
    assert data["leakage"]["passed"] is True


def test_manifest_includes_timestamp(tmp_path: Path) -> None:
    mb = ManifestBuilder(out_dir=tmp_path)
    out = mb.write()
    data = json.loads(out.read_text())
    assert "generated_at" in data
