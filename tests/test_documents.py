"""Document-level acceptance tests: structure, alignment, moves and cross-clause impact."""
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"


def _rows(report):
    return [(pa.pair.status.value, pa.pair.moved, pa.v1_text, pa.v2_text) for pa in report.pairs]


def test_abcd_alignment(engine):
    v1 = ("The library opens at 9 AM.\n\nStudents must wear identity cards.\n\n"
          "Examination fees must be paid before registration.\n\nHostel residents must sign the register.")
    v2 = ("The library opens at 9 AM.\n\nExamination fees shall be paid prior to registration.\n\n"
          "Hostel residents must sign the register.\n\nParking is not permitted near the main gate.")
    r = engine.compare(v1, v2)
    statuses = [(s, a, b) for s, _, a, b in _rows(r)]
    assert ("UNCHANGED", "The library opens at 9 AM.", "The library opens at 9 AM.") in statuses
    assert ("REMOVED", "Students must wear identity cards.", None) in statuses
    assert any(s == "MODIFIED" and a.startswith("Examination fees") for s, a, _ in statuses)
    assert any(s == "ADDED" and b.startswith("Parking") for s, _, b in statuses)
    assert r.summary["moved"] == 0


def test_pure_reorder_is_move_not_add_delete(engine):
    v1 = "Rule one applies to staff.\n\nRule two applies to students.\n\nRule three applies to visitors."
    v2 = "Rule three applies to visitors.\n\nRule one applies to staff.\n\nRule two applies to students."
    r = engine.compare(v1, v2)
    assert r.summary["added"] == 0 and r.summary["removed"] == 0
    assert r.summary["moved"] == 1                      # minimal explanation: one clause moved


def test_killer_document(engine):
    v1 = ("All students must maintain 75% attendance. "
          "Students with medical exemptions may appear for examinations.")
    v2 = ("All students should maintain 70% attendance. "
          "Students with medical exemptions may not appear for examinations. "
          "Students with pending disciplinary cases require special approval.")
    r = engine.compare(v1, v2)
    found = {f"{c.category.value}/{c.subtype}" for pa in r.pairs for c in pa.changes}
    assert found == {"MODALITY/WEAKENED", "NUMERIC/DECREASED", "NEGATION/INTRODUCED", "STRUCTURE/ADDED"}


@pytest.fixture(scope="module")
def academic(engine):
    return engine.compare(DEMO / "academic_regulations_v1.txt", DEMO / "academic_regulations_v2.txt")


def test_definition_change_propagates(academic):
    affected = {academic.pairs[i.affected_pair_index].v2_text for i in academic.impacts}
    assert "Scholarships are available to all Eligible Students." in affected
    assert "Hostel accommodation is available to all Eligible Students." in affected
    assert "Only Eligible Students may register for end-semester examinations." in affected
    for imp in academic.impacts:
        assert academic.pairs[imp.affected_pair_index].pair.status.value == "UNCHANGED"


def test_academic_demo_expectations(academic):
    by_new = {pa.v2_text: pa for pa in academic.pairs if pa.v2_text}
    pa = by_new["The student portal shall display attendance records once every fortnight."]
    assert pa.verdict.value == "MEANING_PRESERVING"
    pa = by_new['"Examination Committee" means the committee constituted by the Registrar.']
    assert [f"{c.category.value}/{c.subtype}" for c in pa.changes] == ["ENTITY/REPLACED"]
    pa = by_new["Applicants are required to submit their re-evaluation applications within one week "
                "of the declaration of results."]
    assert pa.verdict.value == "MEANING_PRESERVING"
    moved = [pa for pa in academic.pairs if pa.pair.moved]
    assert [pa.v2_text for pa in moved] == ["Students must carry their identity cards on campus."]
