"""בניית קטעי הידע (chunks) מהסעיפים.

שלושה סוגי קטעים (README סעיף 18):
- סעיף רגיל: קטע אחד, וסעיף ארוך מפוצל עם חפיפה.
- מילון מונחים: כל שורה בטבלה היא קטע.
- שאלות נפוצות: כל זוג של שאלה ותשובה הוא קטע.
"""

from __future__ import annotations

import hashlib

from llama_index.core.node_parser import SentenceSplitter

from config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS
from ingestion.docx_reader import BlockKind, WordDocument
from ingestion.section_splitter import (
    DocumentSection,
    convert_blocks_to_text,
    is_glossary_table_section,
    split_by_sections,
    split_into_question_answer_pairs,
)
from rag.knowledge_chunk import ChunkKind, KnowledgeChunk


def build_context_header(doc_title: str, section_title: str) -> str:
    """השורה שמוצמדת לתחילת כל קטע, כדי שההקשר יישמר גם בקטע קצר (README סעיף 3)."""
    return f"מסמך: {doc_title} | סעיף: {section_title}"


def compute_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_chunk_id(doc_id: str, section_title: str, part_index: int) -> str:
    """מזהה קבוע לקטע, כך שהרצה חוזרת של הטעינה לא יוצרת כפילויות (README סעיף 21)."""
    section_hash = compute_text_hash(section_title)[:10]
    return f"doc{doc_id}-{section_hash}-{part_index:02d}"


def create_chunk(
    document: WordDocument,
    section_title: str,
    part_index: int,
    body_text: str,
    kind: ChunkKind,
) -> KnowledgeChunk:
    """יוצרת קטע אחד עם שורת ההקשר בתחילתו. התיוג נעשה בשלב נפרד."""
    header = build_context_header(document.doc_title, section_title)
    full_text = f"{header}\n{body_text.strip()}"
    return KnowledgeChunk(
        chunk_id=build_chunk_id(document.doc_id, section_title, part_index),
        doc_id=document.doc_id,
        doc_title=document.doc_title,
        section=section_title,
        kind=kind,
        text=full_text,
        content_hash=compute_text_hash(full_text),
    )


def create_text_splitter() -> SentenceSplitter:
    """מפצל לסעיפים ארוכים: לפי משפטים, בגודל ובחפיפה שבהגדרות."""
    return SentenceSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        paragraph_separator="\n",
    )


def build_glossary_chunks(
    document: WordDocument, section: DocumentSection
) -> list[KnowledgeChunk]:
    """כל מונח במילון הופך לקטע נפרד בפורמט 'מונח: X | הסבר: Y'."""
    chunks: list[KnowledgeChunk] = []
    for block in section.blocks:
        if block.kind != BlockKind.TABLE:
            continue
        header_cells, *term_rows = block.table_rows
        for term_row in term_rows:
            term_text = " | ".join(
                f"{header}: {value}" for header, value in zip(header_cells, term_row)
            )
            chunks.append(
                create_chunk(
                    document,
                    section.title,
                    part_index=len(chunks),
                    body_text=term_text,
                    kind=ChunkKind.GLOSSARY_TERM,
                )
            )
    return chunks


def build_long_section_chunks(
    document: WordDocument,
    section_title: str,
    section_text: str,
    first_part_index: int,
    text_splitter: SentenceSplitter,
) -> list[KnowledgeChunk]:
    """סעיף רגיל: קטע אחד, או כמה קטעים חופפים אם הסעיף ארוך מגודל הקטע."""
    text_parts = text_splitter.split_text(section_text)
    return [
        create_chunk(
            document,
            section_title,
            part_index=first_part_index + part_number,
            body_text=text_part,
            kind=ChunkKind.SECTION,
        )
        for part_number, text_part in enumerate(text_parts)
    ]


def build_section_chunks(
    document: WordDocument, section: DocumentSection, text_splitter: SentenceSplitter
) -> list[KnowledgeChunk]:
    """בוחרת את שיטת החלוקה המתאימה לסעיף ובונה את הקטעים שלו."""
    if is_glossary_table_section(section):
        return build_glossary_chunks(document, section)

    blocks_before_questions, question_answer_pairs = split_into_question_answer_pairs(
        section.blocks
    )
    chunks: list[KnowledgeChunk] = []
    if blocks_before_questions:
        chunks.extend(
            build_long_section_chunks(
                document,
                section.title,
                convert_blocks_to_text(blocks_before_questions),
                first_part_index=0,
                text_splitter=text_splitter,
            )
        )
    for pair_blocks in question_answer_pairs:
        chunks.append(
            create_chunk(
                document,
                section.title,
                part_index=len(chunks),
                body_text=convert_blocks_to_text(pair_blocks),
                kind=ChunkKind.FAQ_PAIR,
            )
        )
    return chunks


def build_document_chunks(document: WordDocument) -> list[KnowledgeChunk]:
    """בונה את כל הקטעים של מסמך אחד."""
    text_splitter = create_text_splitter()
    chunks: list[KnowledgeChunk] = []
    for section in split_by_sections(document):
        chunks.extend(build_section_chunks(document, section, text_splitter))
    return chunks
