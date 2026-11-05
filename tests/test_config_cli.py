import json
from pathlib import Path

import pytest

from rag_service.cli import main
from rag_service.config import IngestionConfig


def test_overlap_validation() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        IngestionConfig(chunk_tokens=40, overlap_tokens=40)


def test_status_works_without_a_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--data-dir", str(tmp_path), "status"]) == 0
    assert json.loads(capsys.readouterr().out) == {"active": None, "history": []}
