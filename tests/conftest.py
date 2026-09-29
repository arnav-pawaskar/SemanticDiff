import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def nlp():
    from semanticdiff.resources import get_spacy
    return get_spacy()


@pytest.fixture(scope="session")
def engine():
    from semanticdiff.pipeline import SemanticDiff
    return SemanticDiff()
