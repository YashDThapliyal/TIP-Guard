"""Composition tables for a generated dataset.

The dataset card must state the real counts, so they are produced here and
pasted rather than typed by hand: a hand-written table drifts silently the
first time the generator changes.
"""

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from tipguard.benchmark.io import DatasetError
from tipguard.benchmark.schema import BenchmarkCase

#: Read in blocks so the digest does not depend on holding the file in memory.
DIGEST_BLOCK_SIZE = 1 << 20

TOTAL = "total"


def dataset_sha256(path: Path) -> str:
    """Digest of the dataset file exactly as it sits on disk."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(DIGEST_BLOCK_SIZE):
                digest.update(block)
    except OSError as exc:
        raise DatasetError(f"{path}: cannot read file: {exc}") from exc
    return digest.hexdigest()


def _row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: str, columns: Sequence[str], rows: Sequence[tuple[str, Counter[str]]]) -> str:
    """A Markdown table of `rows` by `columns`, with a totals column and row."""
    names = [*columns, TOTAL]
    lines = [_row([header, *names]), _row(["---"] * (len(names) + 1))]
    for label, counts in rows:
        cells = [str(counts[name]) for name in columns]
        lines.append(_row([label, *cells, str(sum(counts.values()))]))
    footer = Counter[str]()
    for _, counts in rows:
        footer.update(counts)
    lines.append(_row([TOTAL, *(str(footer[name]) for name in columns), str(sum(footer.values()))]))
    return "\n".join(lines)


def _crossed(
    cases: Sequence[BenchmarkCase], column: str
) -> tuple[list[str], list[tuple[str, Counter[str]]]]:
    """Case type down the side, `column` across the top."""
    columns = sorted({str(getattr(case, column)) for case in cases})
    rows = []
    for case_type in sorted({case.case_type.value for case in cases}):
        counts = Counter(
            str(getattr(case, column)) for case in cases if case.case_type.value == case_type
        )
        rows.append((case_type, counts))
    return columns, rows


def _cross_table(cases: Sequence[BenchmarkCase], column: str, header: str) -> str:
    columns, rows = _crossed(cases, column)
    return _table(header, columns, rows)


def format_stats(cases: Sequence[BenchmarkCase], digest: str) -> str:
    """The composition of a dataset as Markdown, ready to paste into the card."""
    splits = Counter(case.split.value for case in cases)
    split_rows = [_row(["split", "cases"]), _row(["---", "---"])]
    split_rows += [_row([name, str(splits[name])]) for name in sorted(splits)]
    return "\n\n".join(
        (
            "### Case type by transformation",
            _cross_table(cases, "transformation", "case type"),
            "### Case type by difficulty",
            _cross_table(cases, "difficulty", "case type"),
            "### Split",
            "\n".join(split_rows),
            f"total: {len(cases)}",
            f"sha256: {digest}",
        )
    )
