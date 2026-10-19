"""Fetch two public arXiv versions used as a real-paper regression example.

Run from the repository root with ``python -m scripts.download_paper_demo``.
The PDFs are cached locally and are not committed to the repository.
"""

from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "results" / "cache" / "papers"
PAPERS = {
    "attention_v1.pdf": "https://arxiv.org/pdf/1706.03762v1",
    "attention_v2.pdf": "https://arxiv.org/pdf/1706.03762v2",
    "bert_v1.pdf": "https://arxiv.org/pdf/1810.04805v1",
    "bert_v2.pdf": "https://arxiv.org/pdf/1810.04805v2",
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for filename, url in PAPERS.items():
        path = DEST / filename
        if path.exists() and path.read_bytes().startswith(b"%PDF"):
            print(f"Using {path}")
            continue
        request = Request(url, headers={"User-Agent": "SemanticDiff-paper-regression/1.0"})
        with urlopen(request, timeout=60) as response:
            data = response.read()
        if not data.startswith(b"%PDF"):
            raise ValueError(f"Expected a PDF from {url}")
        path.write_bytes(data)
        print(f"Downloaded {path} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
