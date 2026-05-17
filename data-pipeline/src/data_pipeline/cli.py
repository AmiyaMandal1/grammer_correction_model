from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import typer

from data_pipeline.config import PipelineConfig
from data_pipeline.dedup import dedupe_pairs
from data_pipeline.gec_builder import build_gec_jsonl
from data_pipeline.manifest import ManifestBuilder
from data_pipeline.normalize import normalize_text
from data_pipeline.sft_builder import build_sft_jsonl
from data_pipeline.sources.bea2019 import BEA2019Source
from data_pipeline.sources.c4_200m import C4200MSource
from data_pipeline.sources.conll2014 import CoNLL2014Source
from data_pipeline.sources.gyafc import GYAFCSource
from data_pipeline.sources.jfleg import JFLEGSource
from data_pipeline.sources.paradetox import ParaDetoxSource
from data_pipeline.sources.wiki_auto import WikiAutoSource
from data_pipeline.types import Pair, Split

app = typer.Typer(no_args_is_help=True)


def _normalize_pair(p: Pair) -> Pair | None:
    src = normalize_text(p.src)
    tgt = normalize_text(p.tgt)
    if not src:
        return None
    return Pair(src=src, tgt=tgt, source=p.source, split=p.split, meta=p.meta)


def _load_gec_sources(
    cfg: PipelineConfig, names: list[str], split: Split
) -> Iterable[Pair]:
    for name in names:
        if name == "bea2019":
            yield from BEA2019Source(config=cfg, split=split).iter_pairs()
        elif name == "jfleg":
            yield from JFLEGSource(config=cfg, split=split).iter_pairs()
        elif name == "conll2014":
            yield from CoNLL2014Source(config=cfg).iter_pairs()
        elif name == "c4_200m":
            yield from C4200MSource(config=cfg).iter_pairs()
        else:
            raise typer.BadParameter(f"unknown GEC source: {name}")


def _load_style_sources(cfg: PipelineConfig, names: list[str]) -> Iterable[Pair]:
    for name in names:
        if name == "gyafc":
            yield from GYAFCSource(
                config=cfg, split=Split.TRAIN, domain="Family_Relationships"
            ).iter_pairs()
        elif name == "paradetox":
            yield from ParaDetoxSource(config=cfg).iter_pairs()
        elif name == "wiki_auto":
            yield from WikiAutoSource(config=cfg).iter_pairs()
        else:
            raise typer.BadParameter(f"unknown style source: {name}")


@app.command("build-gec")
def build_gec(
    root: Path = typer.Option(..., exists=False, help="Project root"),
    sources: str = typer.Option(
        "bea2019,jfleg,conll2014,c4_200m",
        help="Comma-separated GEC source names for the train split.",
    ),
    split: str = typer.Option("train"),
    eval_sources: str = typer.Option(
        "",
        help="Comma-separated GEC source names for the eval split (e.g. 'jfleg,conll2014'). "
        "When set, runs a hard leakage check between train and each eval source.",
    ),
) -> None:
    """Build gec_tagger.jsonl and manifest.json."""
    from data_pipeline.leakage import check_leakage

    split_enum = Split(split)
    cfg = PipelineConfig(root=root).ensure_dirs()
    names = [s.strip() for s in sources.split(",") if s.strip()]

    raw_pairs = list(_load_gec_sources(cfg, names, split_enum))
    normalized = [np for p in raw_pairs if (np := _normalize_pair(p)) is not None]
    dedup, dedup_stats = dedupe_pairs(normalized)

    leakage_passed: bool | None = None
    if eval_sources.strip():
        eval_names = [s.strip() for s in eval_sources.split(",") if s.strip()]
        eval_sets: dict[str, list[Pair]] = {}
        for ev_name in eval_names:
            ev_split = Split.TEST if ev_name == "conll2014" else Split.DEV
            ev_raw = list(_load_gec_sources(cfg, [ev_name], ev_split))
            eval_sets[ev_name] = [
                np for p in ev_raw if (np := _normalize_pair(p)) is not None
            ]
        check_leakage(train=dedup, eval_sets=eval_sets)  # raises on overlap
        leakage_passed = True

    out = cfg.processed_dir / "gec_tagger.jsonl"
    written = build_gec_jsonl(dedup, out)

    mb = ManifestBuilder(out_dir=cfg.processed_dir)
    mb.add_artifact(name="gec_tagger", path=out, row_count=written)
    for name in names:
        per_source = sum(1 for p in raw_pairs if p.source == name)
        retained = sum(1 for p in dedup if p.source == name)
        mb.add_source(name=name, row_count=per_source, retained=retained)
    mb.set_dedup(removed_exact=dedup_stats.removed_exact)
    mb.set_leakage_passed(leakage_passed)
    mb.write("manifest_gec.json")
    typer.echo(f"wrote {written} records to {out}")


@app.command("build-sft")
def build_sft(
    root: Path = typer.Option(..., exists=False, help="Project root"),
    sources: str = typer.Option(
        "gyafc,paradetox,wiki_auto", help="Comma-separated style sources."
    ),
) -> None:
    """Build style_sft.jsonl and manifest.json."""
    cfg = PipelineConfig(root=root).ensure_dirs()
    names = [s.strip() for s in sources.split(",") if s.strip()]

    raw_pairs = list(_load_style_sources(cfg, names))
    normalized = [np for p in raw_pairs if (np := _normalize_pair(p)) is not None]
    dedup, dedup_stats = dedupe_pairs(normalized)

    out = cfg.processed_dir / "style_sft.jsonl"
    written = build_sft_jsonl(dedup, out)

    mb = ManifestBuilder(out_dir=cfg.processed_dir)
    mb.add_artifact(name="style_sft", path=out, row_count=written)
    for name in names:
        per_source = sum(1 for p in raw_pairs if p.source == name)
        retained = sum(1 for p in dedup if p.source == name)
        mb.add_source(name=name, row_count=per_source, retained=retained)
    mb.set_dedup(removed_exact=dedup_stats.removed_exact)
    mb.set_leakage_passed(None)
    mb.write("manifest_sft.json")
    typer.echo(f"wrote {written} records to {out}")


@app.command("build-all")
def build_all(
    root: Path = typer.Option(..., exists=False),
    gec_sources: str = typer.Option("bea2019,c4_200m"),
    eval_sources: str = typer.Option("jfleg,conll2014"),
    style_sources: str = typer.Option("gyafc,paradetox,wiki_auto"),
    split: str = typer.Option("train"),
) -> None:
    """Build all artifacts: GEC tagger and style SFT."""
    build_gec(root=root, sources=gec_sources, split=split, eval_sources=eval_sources)
    build_sft(root=root, sources=style_sources)


if __name__ == "__main__":
    app()
