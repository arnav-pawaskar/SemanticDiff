"""Download and sanity-check all models used by SemanticDiff."""
from semanticdiff.config import load_config
from semanticdiff.resources import get_all, resolve_device

cfg = load_config()
nlp, enc, (nli, idx) = get_all(cfg)
print("device:", resolve_device(cfg["device"]))
print("spaCy:", nlp.meta["name"], nlp.meta["version"])
emb = enc.encode(["Interest rate is 5%.", "Interest rate is 15%."], normalize_embeddings=True)
print("cosine(5% vs 15%):", float(emb[0] @ emb[1]))
print("NLI labels:", idx)
pairs = [("Students may access the laboratory.", "Students may not access the laboratory."),
         ("The server must respond within 500 milliseconds.", "The server shall return a response within half a second.")]
for (p, h), row in zip(pairs, nli.predict(pairs, apply_softmax=True)):
    print({k: round(float(row[i]), 3) for k, i in idx.items()}, "|", p, "->", h)
