"""חלוקת מסמך לסעיפים לפי הכותרות, והמרת התוכן של כל סעיף לטקסט קריא."""

from __future__ import annotations

from dataclasses import dataclass, field

from ingestion.docx_reader import BlockKind, DocumentBlock, WordDocument

# שם הסעיף לטקסט שמופיע לפני הכותרת הראשונה במסמך
INTRODUCTION_SECTION_TITLE = "מבוא"
QUESTION_PREFIX = "שאלה:"


@dataclass
class DocumentSection:
    title: str
    blocks: list[DocumentBlock] = field(default_factory=list)


def split_by_sections(document: WordDocument) -> list[DocumentSection]:
    """מחלקת את המסמך לסעיפים: כל כותרת סעיף פותחת סעיף חדש."""
    sections: list[DocumentSection] = [DocumentSection(title=INTRODUCTION_SECTION_TITLE)]
    for block in document.blocks:
        if block.kind == BlockKind.DOCUMENT_TITLE:
            continue
        if block.kind == BlockKind.SECTION_HEADING:
            sections.append(DocumentSection(title=block.text))
        else:
            sections[-1].blocks.append(block)
    return [section for section in sections if section.blocks]


def convert_table_to_text(table_rows: list[list[str]]) -> str:
    """ממירה טבלה לשורות קריאות: כל שורה הופכת ל'כותרת: ערך | כותרת: ערך'."""
    header_cells, *data_rows = table_rows
    if not data_rows:
        return " | ".join(header_cells)
    lines = []
    for row in data_rows:
        pairs = [
            f"{header}: {value}"
            for header, value in zip(header_cells, row)
            if value
        ]
        lines.append(" | ".join(pairs))
    return "\n".join(lines)


def convert_block_to_text(block: DocumentBlock) -> str:
    """מחזירה את הטקסט של יחידת תוכן אחת (פסקה או טבלה)."""
    if block.kind == BlockKind.TABLE:
        return convert_table_to_text(block.table_rows)
    return block.text


def convert_blocks_to_text(blocks: list[DocumentBlock]) -> str:
    """מחברת כמה יחידות תוכן לטקסט אחד, יחידה בכל שורה."""
    return "\n".join(convert_block_to_text(block) for block in blocks)


def split_into_question_answer_pairs(
    blocks: list[DocumentBlock],
) -> tuple[list[DocumentBlock], list[list[DocumentBlock]]]:
    """מפרידה סעיף של שאלות נפוצות לזוגות של שאלה ותשובה.

    מחזירה (טקסט שלפני השאלה הראשונה, רשימת זוגות). כל פסקה שמתחילה ב'שאלה:'
    פותחת זוג חדש, וכל מה שאחריה עד השאלה הבאה שייך לאותו זוג.
    """
    blocks_before_first_question: list[DocumentBlock] = []
    question_answer_pairs: list[list[DocumentBlock]] = []
    for block in blocks:
        if block.kind == BlockKind.PARAGRAPH and block.text.startswith(QUESTION_PREFIX):
            question_answer_pairs.append([block])
        elif question_answer_pairs:
            question_answer_pairs[-1].append(block)
        else:
            blocks_before_first_question.append(block)
    return blocks_before_first_question, question_answer_pairs


def is_glossary_table_section(section: DocumentSection) -> bool:
    """סעיף מילון: מכיל טבלה שהעמודה הראשונה שלה היא 'מונח'."""
    return any(
        block.kind == BlockKind.TABLE and block.table_rows[0][0].strip() == "מונח"
        for block in section.blocks
    )
