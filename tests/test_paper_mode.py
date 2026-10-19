from semanticdiff.changes.verdict import decide
from semanticdiff.analyze.lexical import token_diff
from semanticdiff.analyze.citations import without_citations
from semanticdiff.analyze.quantities import extract_quantities
from semanticdiff.analyze.temporal import extract_dates
from semanticdiff.ingest.extract import RawLine
from semanticdiff.ingest.structure import build_blocks
from semanticdiff.models import NLIResult, Verdict
from semanticdiff.models import Provision, ProvisionKind, PairStatus
from semanticdiff.align.aligner import align, reconcile_split_merge
from semanticdiff.ingest.segment import segment_blocks
from semanticdiff.ingest.structure import Block


def test_paper_front_matter_is_not_compared_as_prose():
    lines = [
        RawLine("An Improved Method", 1, 0, True),
        RawLine("Alice Researcher", 1, 1, True),
        RawLine("University of Example alice@example.edu", 1, 2, True),
        RawLine("Abstract", 1, 3, True),
        RawLine("We measured the outcome in a controlled experiment.", 1, 4, True),
        RawLine("1", 1, 5, True),
        RawLine("Introduction", 1, 6, True),
        RawLine("The control group received standard care.", 1, 7, True),
        RawLine("2 Results", 2, 8, True),
        RawLine("The treatment group improved by 12%.", 2, 9, True),
    ]
    blocks = build_blocks(lines, paper_mode=True)
    assert [b.text for b in blocks] == [
        "We measured the outcome in a controlled experiment.",
        "The control group received standard care.",
        "The treatment group improved by 12%.",
    ]
    assert [b.section_path for b in blocks] == [["Abstract"], ["1"], ["2"]]


def test_inline_abstract_heading_keeps_abstract_text():
    blocks = build_blocks([RawLine("Alice Researcher", 1, 0, True),
                           RawLine("Abstract: We measured the outcome.", 1, 1, True)], paper_mode=True)
    assert [(b.section_path, b.text) for b in blocks] == [(["Abstract"], "We measured the outcome.")]


def test_formula_and_numbered_table_cell_do_not_change_paper_section():
    lines = [RawLine("Abstract", 1, 0, True),
             RawLine("We describe the model.", 1, 1, True),
             RawLine("1 Introduction", 1, 2, True),
             RawLine("The model uses attention.", 1, 3, True),
             RawLine("A(q, K, V ) = Softmax(qKT )V", 1, 4, True),
             RawLine("25.9 MoE", 1, 5, True)]
    blocks = build_blocks(lines, paper_mode=True)
    assert [b.section_path for b in blocks] == [["Abstract"], ["1"], ["1"], ["1"]]


def test_title_case_table_header_is_not_a_section():
    lines = [RawLine("Abstract", 1, 0, True),
             RawLine("We describe the model.", 1, 1, True),
             RawLine("1 Introduction", 1, 2, True),
             RawLine("Residual Dropout", 1, 3, True),
             RawLine("The model uses attention.", 1, 4, True)]
    blocks = build_blocks(lines, paper_mode=True)
    assert [b.section_path for b in blocks] == [["Abstract"], ["1"], ["1"]]


def test_pdf_paragraph_continues_across_boxes_in_same_block():
    lines = [RawLine("Abstract", 1, 0, True),
             RawLine("The score rose to", 1, 1, True),
             RawLine("80.5% after training.", 1, 1, True)]
    blocks = build_blocks(lines, paper_mode=True)
    assert [(b.section_path, b.text) for b in blocks] == [
        (["Abstract"], "The score rose to 80.5% after training.")]


def test_high_embedding_similarity_does_not_certify_equivalence():
    cfg = {"verdict": {"entail": 0.6, "contradiction": 0.6,
                       "preserve_similarity": 0.92, "unexplained_min_tokens": 1,
                       "similarity_only_preserves": False}}
    nli = NLIResult(0.1, 0.1, 0.8, 0.1, 0.1, 0.8)
    verdict, rule, changes, _, _ = decide([], nli, 0.98, 2, cfg)
    assert verdict == Verdict.UNCLASSIFIED
    assert rule == "V4:similarity-only"
    assert changes[0].category.value == "UNEXPLAINED"


def test_reference_renumbering_is_not_a_numeric_claim(nlp):
    old = nlp("The model improves accuracy [12, 13] by 4%.")
    new = nlp("The model improves accuracy [14, 15] by 5%.")
    assert [(q.raw, q.value) for q in extract_quantities(old)] == [("4%", 4.0)]
    assert extract_dates(old, extract_quantities(old)) == []
    changed_old, changed_new = token_diff(old, new).content_changed(old, new)
    assert {old[i].text for i in changed_old} == {"4"}
    assert {new[i].text for i in changed_new} == {"5"}


def test_table_scores_are_not_calendar_dates(nlp):
    doc = nlp("24.9 65")
    assert extract_dates(doc, extract_quantities(doc)) == []


def test_pdf_footnote_labels_do_not_change_nli_claim():
    assert without_citations("3We used four GPUs [18].") == "We used four GPUs ."


def test_author_year_suffix_is_citation_not_a_date(nlp):
    old = nlp("Unlike Peters et al., 2018, the model uses both contexts.")
    new = nlp("Unlike Peters et al., 2018a, the model uses both contexts.")
    assert extract_dates(old, extract_quantities(old)) == []
    assert extract_dates(new, extract_quantities(new)) == []
    changed_old, changed_new = token_diff(old, new).content_changed(old, new)
    assert not changed_old and not changed_new


def test_reference_renumbering_does_not_become_nli_reversal(engine):
    pair = engine.analyze_pair(
        "The model improves accuracy [12] in all trials.",
        "The model improves accuracy [14] in all trials.")
    assert pair.verdict == Verdict.MEANING_PRESERVING
    assert not pair.changes
    assert any("references changed" in note for note in pair.notes)


def test_paper_non_prose_is_kept_for_review(nlp):
    blocks = [Block("The model improves accuracy by 4%.", ["Results"]),
              Block("Table 1: BLEU scores", ["Results"]),
              Block("41.0 41.2", ["Results"]),
              Block("Smith et al. 2017", ["References"]),
              Block("The variable xi ∈ Rd describes the input sequence.", ["Results"])]
    provisions = segment_blocks(blocks, "v1", nlp, paper_mode=True)
    assert [p.kind for p in provisions] == [ProvisionKind.SENTENCE] + [ProvisionKind.NON_PROSE] * 4


def test_paper_reflow_is_not_called_a_move():
    old = [Provision(f"v1-{i}", "v1", i, t, section_path=["Results"])
           for i, t in enumerate(["Alpha result is stable.", "Beta result is stable.",
                                   "Gamma result is stable."])]
    new = [Provision(f"v2-{i}", "v2", i, t, section_path=["Results"])
           for i, t in enumerate(["Beta result is stable.", "Alpha result is stable.",
                                   "Gamma result is stable."])]
    rows = align(old, new, None, None, paper_mode=True)
    assert all(not row.moved for row in rows)
    assert sum(row.status == PairStatus.UNCHANGED for row in rows) >= 2


def test_paper_short_fragment_never_matches_a_prose_claim():
    old = [Provision("v1-0", "v1", 0, "41.0 41.2", kind=ProvisionKind.NON_PROSE)]
    new = [Provision("v2-0", "v2", 0, "The BLEU result is 41.2.")]
    rows = align(old, new, None, None, paper_mode=True)
    assert {row.status for row in rows} == {PairStatus.ADDED, PairStatus.REMOVED}


def test_nearby_exact_paper_text_survives_heading_renumbering():
    old = [Provision("v1-0", "v1", 0, "The results remain stable across all trials.",
                     section_path=["Conclusion"], page=10)]
    new = [Provision("v2-0", "v2", 0, "The results remain stable across all trials.",
                     section_path=["7"], page=10)]
    rows = align(old, new, None, None, paper_mode=True)
    assert len(rows) == 1 and rows[0].status == PairStatus.UNCHANGED and not rows[0].moved


def test_paper_table_edit_stays_visible_without_a_semantic_claim(engine):
    old = "Abstract\nThe model improves accuracy.\n1 Results\nTable 1: BLEU scores\n41.0 28.4"
    new = "Abstract\nThe model improves accuracy.\n1 Results\nTable 1: BLEU scores\n41.2 28.4"
    report = engine.compare(old, new)
    review = [p for p in report.pairs if p.verdict_rule == "paper:non-prose"
              and p.pair.status != PairStatus.UNCHANGED]
    assert review
    assert report.summary["paper_review"] == len(review)
    assert all(p.verdict == Verdict.UNCLASSIFIED for p in review)
    assert not any(c.category.value == "NUMERIC" for p in review for c in p.changes)
    assert any(any(op != "equal" and "41.0" in old and "41.2" in new
                   for op, old, new in p.token_diff) for p in review)


def test_multiple_paper_metrics_match_their_own_claims(engine):
    old = ("The GLUE score was 80.4% (7.6% improvement), and MultiNLI accuracy was "
           "86.7% (5.6% improvement), outperforming people by 2.0.")
    new = ("The GLUE score was 80.5% (7.7% point improvement), and MultiNLI accuracy was "
           "86.7% (4.6% improvement), while SQuAD Test F1 was 83.1 (5.1 point improvement).")
    pair = engine.analyze_pair(old, new)
    changes = {(c.old, c.new) for c in pair.changes if c.category.value == "NUMERIC"}
    assert ("80.4%", "80.5%") in changes
    assert ("7.6%", "7.7%") in changes
    assert ("5.6%", "4.6%") in changes
    assert not any(a == "2.0" and b == "83.1" for a, b in changes)


def test_paper_sentence_split_and_merge_rejoin():
    whole = "The treatment group improved accuracy by 12 percent after training."
    fragments = ["The treatment group improved accuracy by 12 percent",
                 "after training."]

    def provisions(version, texts):
        return [Provision(f"{version}-{i}", version, i, text,
                          section_path=["Results"], page=2)
                for i, text in enumerate(texts)]

    for old_texts, new_texts in (([whole], fragments), (fragments, [whole])):
        old, new = provisions("v1", old_texts), provisions("v2", new_texts)
        rows = reconcile_split_merge(align(old, new, None, None, paper_mode=True), old, new)
        assert len(rows) == 1
        assert rows[0].status == PairStatus.UNCHANGED
        assert len(rows[0].v1_ids or [rows[0].v1_id]) == len(old)
        assert len(rows[0].v2_ids or [rows[0].v2_id]) == len(new)


def test_paper_real_new_sentence_is_not_swallowed_by_split_repair():
    old = [Provision("v1-0", "v1", 0, "The treatment group improved accuracy by 12 percent.",
                     section_path=["Results"], page=2)]
    new = [Provision("v2-0", "v2", 0, old[0].text, section_path=["Results"], page=2),
           Provision("v2-1", "v2", 1, "The placebo group showed no improvement.",
                     section_path=["Results"], page=2)]
    rows = reconcile_split_merge(align(old, new, None, None, paper_mode=True), old, new)
    assert {row.status for row in rows} == {PairStatus.UNCHANGED, PairStatus.ADDED}
