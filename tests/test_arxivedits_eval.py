"""Check the independent benchmark adapter's gold-link accounting."""

from eval.run_arxivedits_eval import counts, metrics, revisions


def test_arxivedits_adapter_keeps_split_links_and_excludes_unscored_text():
    payload = {"paper-id": {
        "1": {"paper-id-1-0-0": "# Introduction",
              "paper-id-1-1-0": "The model improved after training.",
              "paper-id-1-1-1": "[MATH]"},
        "2": {"paper-id-2-0-0": "# Introduction",
              "paper-id-2-1-0": "The model improved",
              "paper-id-2-1-1": "after training.",
              "paper-id-2-1-2": "[MATH]"},
        "alignment": [["paper-id-1-1-0", "paper-id-2-1-0"],
                      ["paper-id-1-1-0", "paper-id-2-1-1"]],
        "skipped_alignment_indices": ["paper-id-1-1-1", "paper-id-2-1-2"],
    }}
    case, = revisions(payload)
    assert case.old_texts == ["The model improved after training."]
    assert case.new_texts == ["The model improved", "after training."]
    assert case.gold == {(0, 0), (0, 1)}
    result = metrics(counts(case.gold, {(0, 0)}))
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == 0.6667
