"""Fetch two public arXiv versions used as a real-paper regression example.

Run from the repository root with ``python -m scripts.download_paper_demo``.
The PDFs are cached locally and are not committed to the repository.
"""

import ssl
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "results" / "cache" / "papers"
# slug -> (demo label, demo title, arXiv id); each paper is fetched as <slug>_v1.pdf and <slug>_v2.pdf.
PAPER_DEMOS = {
    "attention": ("Research paper · Attention Is All You Need",
                  "Attention Is All You Need, arXiv version 1 to version 2", "1706.03762"),
    "bert": ("Research paper · BERT", "BERT, arXiv version 1 to version 2", "1810.04805"),
    "batchnorm": ("Research paper · Batch Normalization",
                  "Batch Normalization, arXiv version 1 to version 2", "1502.03167"),
    "adam": ("Research paper · Adam", "Adam, arXiv version 1 to version 2", "1412.6980"),
}
PAPERS = {f"{slug}_v{v}.pdf": f"https://arxiv.org/pdf/{arxiv_id}v{v}"
          for slug, (_, _, arxiv_id) in PAPER_DEMOS.items() for v in (1, 2)}


def _ssl_context() -> ssl.SSLContext:
    """Verify against certifi's bundle when present; some Windows setups lack a usable system store."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for filename, url in PAPERS.items():
        path = DEST / filename
        if path.exists() and path.read_bytes().startswith(b"%PDF"):
            print(f"Using {path}")
            continue
        request = Request(url, headers={"User-Agent": "SemanticDiff-paper-regression/1.0"})
        with urlopen(request, timeout=60, context=_ssl_context()) as response:
            data = response.read()
        if not data.startswith(b"%PDF"):
            raise ValueError(f"Expected a PDF from {url}")
        path.write_bytes(data)
        print(f"Downloaded {path} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
