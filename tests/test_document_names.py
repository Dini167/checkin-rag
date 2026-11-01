"""בדיקות לשמות הידידותיים של המסמכים (docs/post_build_fixes.md, תיקון 3)."""

from __future__ import annotations

import config
from ingestion.docx_reader import parse_document_id_and_title
from rag.document_names import DOCUMENT_FRIENDLY_NAMES


def test_every_document_has_a_friendly_name() -> None:
    document_ids = {
        parse_document_id_and_title(path)[0] for path in config.DATA_DIR.glob("*.docx")
    }
    assert document_ids == set(DOCUMENT_FRIENDLY_NAMES)


def test_friendly_names_have_no_technical_identifiers() -> None:
    for friendly_name in DOCUMENT_FRIENDLY_NAMES.values():
        assert not any(character.isdigit() for character in friendly_name)
        assert ".docx" not in friendly_name
