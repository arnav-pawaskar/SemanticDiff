"""End-to-end SemanticDiff pipeline.

    engine = SemanticDiff()
    report = engine.compare("v1.pdf", "v2.pdf")        # documents -> DiffReport
    pa = engine.analyze_pair("old sentence", "new sentence")   # one provision pair
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from semanticdiff.align.aligner import align
from semanticdiff.align.similarity import embed
from semanticdiff.analyze.frame import build_frame
from semanticdiff.analyze.lexical import textual_level, token_diff
from semanticdiff.analyze.nli import nli_batch
from semanticdiff.changes.diff import diff_frames
from semanticdiff.changes.severity import assign_severity
from semanticdiff.changes.verdict import decide, semantic_level
from semanticdiff.config import config_hash, load_config
from semanticdiff.impact.graph import build_impact
from semanticdiff.ingest.segment import load_document, normalize_for_match
from semanticdiff.models import (AlignedPair, AtomicChange, ChangeCategory as C, DiffReport, ModalClass,
                                 PairAnalysis, PairStatus, ProvisionFrame, Severity, Verdict)
from semanticdiff.resources import get_encoder, get_nli, get_spacy, resolve_device


def _normative(frame: ProvisionFrame) -> bool:
    return any(p.modal_class != ModalClass.NONE for p in frame.predicates)


class SemanticDiff:
    def __init__(self, cfg: Optional[dict] = None, use_nli: bool = True, use_embeddings: bool = True):
        self.cfg = cfg or load_config()
        m, dev = self.cfg["models"], self.cfg.get("device", "auto")
        self.nlp = get_spacy(m["spacy"])
        self.encoder = get_encoder(m["embedding"], dev) if use_embeddings else None
        self.nli_model, self.nli_labels = get_nli(m["nli"], dev) if use_nli else (None, None)

    # ------------------------------------------------------------------ pair level
    def _analyze_modified(self, doc1, doc2, pa: PairAnalysis) -> tuple[ProvisionFrame, ProvisionFrame, object]:
        f1, f2 = build_frame(doc1), build_frame(doc2)
        lex = token_diff(doc1, doc2)
        fd = diff_frames(doc1, doc2, f1, f2, lex, frozenset(self.cfg.get("ablations", [])))
        pa.textual_change = lex.textual_change
        pa.textual_level = textual_level(lex.textual_change, self.cfg)
        pa.token_diff = [(op.op, op.old, op.new) for op in lex.ops]
        pa.changes = fd.changes
        pa.notes = list(fd.notes)
        pa.frames = {"v1": f1, "v2": f2, "explained_ratio": fd.explained_ratio,
                     "residual": {"old": fd.residual_old, "new": fd.residual_new}}
        return f1, f2, fd

    def _finalise(self, pa: PairAnalysis, fd, f1: ProvisionFrame, f2: ProvisionFrame) -> None:
        identical = pa.pair.status == PairStatus.UNCHANGED
        residual = (fd.residual_old + fd.residual_new) if fd else []
        verdict, rule, extra, notes, conf = decide(
            pa.changes, pa.nli, pa.pair.sim_embed, len(residual), self.cfg, identical=identical,
            residual_desc=" / ".join([" ".join(fd.residual_old), " ".join(fd.residual_new)]) if fd else "")
        pa.changes.extend(extra)
        in_def = bool((f1 and f1.defined_term) or (f2 and f2.defined_term))
        for ch in pa.changes:
            if ch.category != C.STRUCTURE:
                assign_severity(ch, self.cfg, in_definition=in_def)
        pa.verdict, pa.verdict_rule = verdict, rule
        pa.notes.extend(notes)
        pa.semantic_level = semantic_level(verdict, [c for c in pa.changes if c.category != C.STRUCTURE])

    def analyze_pairs(self, pairs: list[tuple[str, str]]) -> list[PairAnalysis]:
        """Analyse sentence pairs directly (used by tests and evaluation)."""
        docs1 = list(self.nlp.pipe([a for a, _ in pairs]))
        docs2 = list(self.nlp.pipe([b for _, b in pairs]))
        sims = [None] * len(pairs)
        if self.encoder is not None:
            e1 = embed([a for a, _ in pairs], self.encoder)
            e2 = embed([b for _, b in pairs], self.encoder)
            sims = [float(x) for x in np.sum(e1 * e2, axis=1)]
        out, work = [], []
        for k, ((a, b), d1, d2) in enumerate(zip(pairs, docs1, docs2)):
            same = normalize_for_match(a) == normalize_for_match(b)
            pa = PairAnalysis(pair=AlignedPair("v1-%03d" % k, "v2-%03d" % k,
                                               PairStatus.UNCHANGED if same else PairStatus.MODIFIED,
                                               sim_embed=sims[k]), v1_text=a, v2_text=b)
            out.append(pa)
            if same:
                work.append((pa, None, None, None))
            else:
                work.append((pa, *self._analyze_modified(d1, d2, pa)))
        self._run_nli([pa for pa, *_ in work if pa.pair.status == PairStatus.MODIFIED])
        for pa, f1, f2, fd in work:
            self._finalise(pa, fd, f1, f2)
        return out

    def analyze_pair(self, v1: str, v2: str) -> PairAnalysis:
        return self.analyze_pairs([(v1, v2)])[0]

    def _run_nli(self, pas: list[PairAnalysis]) -> None:
        if self.nli_model is None or not pas:
            return
        results = nli_batch([(p.v1_text, p.v2_text) for p in pas], self.nli_model, self.nli_labels,
                            self.cfg["nli"]["batch_size"])
        for p, r in zip(pas, results):
            p.nli = r

    # ------------------------------------------------------------------ document level
    def compare(self, src1, src2, name1: str = "Version 1", name2: str = "Version 2",
                filename1: Optional[str] = None, filename2: Optional[str] = None) -> DiffReport:
        t0 = time.perf_counter()
        timings = {}
        d1 = load_document(src1, "v1", self.nlp, self.cfg, name1, filename1)
        d2 = load_document(src2, "v2", self.nlp, self.cfg, name2, filename2)
        timings["ingest"] = time.perf_counter() - t0

        t = time.perf_counter()
        e1 = embed([p.text for p in d1.provisions], self.encoder) if self.encoder else None
        e2 = embed([p.text for p in d2.provisions], self.encoder) if self.encoder else None
        al = self.cfg["align"]
        aligned = align(d1.provisions, d2.provisions, e1, e2, al["alpha"], al["threshold"])
        timings["align"] = time.perf_counter() - t

        t = time.perf_counter()
        prov1 = {p.id: p for p in d1.provisions}
        prov2 = {p.id: p for p in d2.provisions}
        frames: dict[str, ProvisionFrame] = {}
        work = []
        pairs: list[PairAnalysis] = []
        for ap in aligned:
            p1, p2 = prov1.get(ap.v1_id), prov2.get(ap.v2_id)
            pa = PairAnalysis(pair=ap, v1_text=p1.text if p1 else None, v2_text=p2.text if p2 else None,
                              v1_section=p1.section_label if p1 else "", v2_section=p2.section_label if p2 else "")
            pairs.append(pa)
            f1 = f2 = fd = None
            if ap.status == PairStatus.MODIFIED:
                f1, f2, fd = self._analyze_modified(p1.doc, p2.doc, pa)
            else:
                if p1 is not None:
                    f1 = build_frame(p1.doc)
                if p2 is not None:
                    f2 = build_frame(p2.doc)
                pa.frames = {"v1": f1, "v2": f2}
                pa.textual_change = 0.0 if ap.status == PairStatus.UNCHANGED else 1.0
                pa.textual_level = Severity.LOW if ap.status == PairStatus.UNCHANGED else Severity.HIGH
            if f1 is not None:
                frames[p1.id] = f1
            if f2 is not None:
                frames[p2.id] = f2
            work.append((pa, f1, f2, fd))
        timings["analyze"] = time.perf_counter() - t

        t = time.perf_counter()
        self._run_nli([pa for pa, *_ in work if pa.pair.status == PairStatus.MODIFIED])
        timings["nli"] = time.perf_counter() - t

        for pa, f1, f2, fd in work:
            st = pa.pair.status
            if st in (PairStatus.ADDED, PairStatus.REMOVED):
                frame = f2 if st == PairStatus.ADDED else f1
                text = pa.v2_text if st == PairStatus.ADDED else pa.v1_text
                ch = AtomicChange(id="s1", category=C.STRUCTURE, subtype=st.value,
                                  description=f"Clause {st.value.lower()}",
                                  old=pa.v1_text, new=pa.v2_text,
                                  evidence={"normative": _normative(frame) if frame else False})
                assign_severity(ch, self.cfg, normative=bool(frame and _normative(frame)))
                pa.changes = [ch]
                pa.verdict, pa.verdict_rule = Verdict.MATERIAL, f"structure:{st.value.lower()}"
                pa.semantic_level = ch.severity
                if frame and frame.defined_term:
                    pa.notes.append(f"Defines the term “{frame.defined_term}”.")
                continue
            self._finalise(pa, fd, f1, f2)
            if pa.pair.moved:
                mv = AtomicChange(id="s2", category=C.STRUCTURE, subtype="MOVED",
                                  description="Clause moved", old=pa.v1_section, new=pa.v2_section)
                assign_severity(mv, self.cfg)
                pa.changes.append(mv)

        impacts, graph = build_impact(pairs, prov1, prov2, frames, self.cfg["impact"]["max_hops"])

        report = DiffReport(v1_name=d1.name, v2_name=d2.name, pairs=pairs, impacts=impacts, graph=graph)
        report.doc_similarity = self._doc_similarity(d1.provisions, d2.provisions, e1, e2)
        report.summary = summarize(report)
        timings["total"] = time.perf_counter() - t0
        report.meta = {"models": self.cfg["models"], "device": resolve_device(self.cfg.get("device", "auto")),
                       "config_hash": config_hash(self.cfg), "thresholds": {
                           "align": self.cfg["align"], "verdict": self.cfg["verdict"]},
                       "timings_s": {k: round(v, 3) for k, v in timings.items()},
                       "n_provisions": {"v1": len(d1.provisions), "v2": len(d2.provisions)}}
        return report

    @staticmethod
    def _doc_similarity(p1, p2, e1, e2) -> dict[str, float]:
        from difflib import SequenceMatcher

        from sklearn.feature_extraction.text import TfidfVectorizer
        t1 = " ".join(p.text for p in p1)
        t2 = " ".join(p.text for p in p2)
        out = {}
        if t1 and t2:
            X = TfidfVectorizer().fit_transform([t1, t2])
            out["tfidf_cosine"] = round(float((X[0] @ X[1].T).toarray()[0, 0]), 4)
            out["lexical_ratio"] = round(SequenceMatcher(None, t1.split(), t2.split(), autojunk=False).ratio(), 4)
        if e1 is not None and e2 is not None and len(e1) and len(e2):
            m1, m2 = e1.mean(axis=0), e2.mean(axis=0)
            out["embedding_cosine"] = round(float(m1 @ m2 / (np.linalg.norm(m1) * np.linalg.norm(m2))), 4)
        return out


def summarize(report: DiffReport) -> dict:
    s = {"clauses_v1": 0, "clauses_v2": 0, "aligned_rows": len(report.pairs),
         "unchanged": 0, "modified": 0, "added": 0, "removed": 0, "moved": 0,
         "meaning_preserving": 0, "material": 0, "unclassified": 0,
         "severity": {"HIGH": 0, "MEDIUM": 0, "LOW": 0}, "categories": {},
         "affected_by_dependency": len({i.affected_pair_index for i in report.impacts})}
    for pa in report.pairs:
        st = pa.pair.status
        s["clauses_v1"] += pa.pair.v1_id is not None
        s["clauses_v2"] += pa.pair.v2_id is not None
        s[st.value.lower()] += 1
        s["moved"] += pa.pair.moved
        if st == PairStatus.MODIFIED:
            if pa.verdict == Verdict.MEANING_PRESERVING:
                s["meaning_preserving"] += 1
            elif pa.verdict == Verdict.MATERIAL:
                s["material"] += 1
            elif pa.verdict == Verdict.UNCLASSIFIED:
                s["unclassified"] += 1
        for ch in pa.changes:
            if ch.category == C.STRUCTURE and ch.subtype == "MOVED":
                continue
            s["severity"][ch.severity.value] += 1
            s["categories"][ch.category.value] = s["categories"].get(ch.category.value, 0) + 1
    return s
