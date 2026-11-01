"""בדיקות לשלב הטעינה: קריאת המסמכים, החלוקה לקטעים והתיוג. לא דורשות מפתח API."""

from __future__ import annotations

import pytest

import config
from ingestion.audience_tagging import load_tagging_overrides, tag_chunks
from ingestion.chunk_builder import build_document_chunks
from ingestion.docx_reader import load_word_documents
from ingestion.section_splitter import convert_table_to_text
from rag.knowledge_chunk import Audience, ChunkKind, KnowledgeChunk


@pytest.fixture(scope="module")
def all_chunks() -> list[KnowledgeChunk]:
    documents = load_word_documents(config.DATA_DIR)
    chunks = [chunk for document in documents for chunk in build_document_chunks(document)]
    tag_chunks(chunks, load_tagging_overrides(config.AUDIENCE_OVERRIDES_FILE))
    return chunks


def find_chunks(chunks: list[KnowledgeChunk], doc_id: str, section: str) -> list[KnowledgeChunk]:
    return [chunk for chunk in chunks if chunk.doc_id == doc_id and chunk.section == section]


def test_all_documents_in_folder_are_loaded() -> None:
    assert len(load_word_documents(config.DATA_DIR)) == len(list(config.DATA_DIR.glob("*.docx")))


def test_every_chunk_starts_with_context_header(all_chunks: list[KnowledgeChunk]) -> None:
    for chunk in all_chunks:
        assert chunk.text.startswith(f"מסמך: {chunk.doc_title} | סעיף: {chunk.section}")


def test_chunk_ids_are_unique_and_stable(all_chunks: list[KnowledgeChunk]) -> None:
    chunk_ids = [chunk.chunk_id for chunk in all_chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    documents = load_word_documents(config.DATA_DIR)
    rebuilt_ids = [chunk.chunk_id for document in documents for chunk in build_document_chunks(document)]
    assert rebuilt_ids == chunk_ids


def test_table_is_converted_to_readable_lines() -> None:
    table_text = convert_table_to_text([["תפקיד", "סיסמה"], ["עובד", "Worker-Demo-2026"]])
    assert table_text == "תפקיד: עובד | סיסמה: Worker-Demo-2026"


def test_glossary_terms_are_separate_chunks(all_chunks: list[KnowledgeChunk]) -> None:
    glossary_chunks = [chunk for chunk in all_chunks if chunk.kind == ChunkKind.GLOSSARY_TERM]
    assert len(glossary_chunks) > 10
    assert all("מונח:" in chunk.text for chunk in glossary_chunks)


def test_faq_pairs_contain_question_and_answer(all_chunks: list[KnowledgeChunk]) -> None:
    faq_chunks = [chunk for chunk in all_chunks if chunk.kind == ChunkKind.FAQ_PAIR]
    assert len(faq_chunks) > 30
    for chunk in faq_chunks:
        assert "שאלה:" in chunk.text and "תשובה:" in chunk.text


def test_team_lead_documents_are_tagged_team_lead(all_chunks: list[KnowledgeChunk]) -> None:
    for chunk in all_chunks:
        if chunk.doc_id in {"11", "12", "13"}:
            assert chunk.audience == Audience.TEAM_LEAD


def test_manual_override_marks_role_description_for_everyone(
    all_chunks: list[KnowledgeChunk],
) -> None:
    role_description_chunks = find_chunks(all_chunks, "03", "ראש צוות")
    assert role_description_chunks
    assert all(chunk.audience == Audience.ALL for chunk in role_description_chunks)


def test_red_dots_section_is_team_lead_and_sensitive(all_chunks: list[KnowledgeChunk]) -> None:
    red_dots_chunks = [
        chunk for chunk in all_chunks if chunk.doc_id == "19" and "נקודות אדומות" in chunk.section
    ]
    assert red_dots_chunks
    assert all(chunk.audience == Audience.TEAM_LEAD and chunk.sensitive for chunk in red_dots_chunks)


def test_glossary_sensitive_terms_only(all_chunks: list[KnowledgeChunk]) -> None:
    glossary_chunks = [chunk for chunk in all_chunks if chunk.kind == ChunkKind.GLOSSARY_TERM]
    sensitive_terms = [chunk for chunk in glossary_chunks if chunk.sensitive]
    assert 0 < len(sensitive_terms) < len(glossary_chunks)
