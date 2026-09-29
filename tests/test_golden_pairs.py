"""Acceptance tests: every example from the project specification."""
from pathlib import Path

import pytest
import yaml

CASES = yaml.safe_load((Path(__file__).parent / "golden" / "pairs.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results(engine):
    analyses = engine.analyze_pairs([(c["v1"], c["v2"]) for c in CASES])
    return {c["id"]: a for c, a in zip(CASES, analyses)}


def _describe(pa) -> str:
    lines = [f"verdict={pa.verdict.value} ({pa.verdict_rule}) text={pa.textual_level.value} "
             f"meaning={pa.semantic_level.value}"]
    lines += [f"  {c.category.value}/{c.subtype} [{c.severity.value}] {c.old!r} -> {c.new!r}" for c in pa.changes]
    if pa.nli:
        lines.append(f"  nli e={pa.nli.entail_12:.2f}/{pa.nli.entail_21:.2f} c={pa.nli.contra_12:.2f}/{pa.nli.contra_21:.2f}")
    lines += [f"  note: {n}" for n in pa.notes]
    return "\n".join(lines)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_golden(case, results):
    pa = results[case["id"]]
    found = [f"{c.category.value}/{c.subtype}" for c in pa.changes]
    info = _describe(pa)
    for exp in case.get("expect", []):
        assert exp in found, f"missing {exp}\n{info}"
    if case.get("exact"):
        assert sorted(found) == sorted(case.get("expect", [])), f"unexpected changes\n{info}"
    if "verdict" in case:
        assert pa.verdict.value == case["verdict"], info
    if "textual_level" in case:
        assert pa.textual_level.value == case["textual_level"], info
    if "semantic_level" in case:
        assert pa.semantic_level.value == case["semantic_level"], info
