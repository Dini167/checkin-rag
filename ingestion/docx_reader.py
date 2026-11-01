"""קריאת קבצי Word תוך שמירה על המבנה: כותרות, פסקאות וטבלאות, לפי הסדר במסמך."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

logger = logging.getLogger(__name__)

# שורת כותרת משנה שחוזרת בכל המסמכים ואין בה מידע
REPEATED_SUBTITLE = "מערכת CheckIn — מאגר ידע למשתמשים"
# שם קובץ בפורמט "01 - כותרת"
FILE_NAME_PATTERN = re.compile(r"^(?P<doc_id>\d{2})\s*-\s*(?P<title>.+)$")


class BlockKind(str, Enum):
    DOCUMENT_TITLE = "document_title"
    SECTION_HEADING = "section_heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"


@dataclass
class DocumentBlock:
    """יחידת תוכן אחת מהמסמך. בטבלה, table_rows מכילה את התאים."""

    kind: BlockKind
    text: str = ""
    table_rows: list[list[str]] = field(default_factory=list)


@dataclass
class WordDocument:
    doc_id: str
    doc_title: str
    file_name: str
    blocks: list[DocumentBlock]


def parse_document_id_and_title(file_path: Path) -> tuple[str, str]:
    """מפרקת שם קובץ כמו '07 - תיקון שעות' למזהה '07' ולכותרת."""
    match = FILE_NAME_PATTERN.match(file_path.stem)
    if not match:
        raise ValueError(f"שם הקובץ לא בפורמט 'NN - כותרת': {file_path.name}")
    return match.group("doc_id"), match.group("title").strip()


def read_paragraph_style_id(paragraph: Paragraph) -> str:
    """מחזירה את מזהה הסגנון הגולמי של הפסקה (למשל 'Heading2').

    במסמכים האלה סגנונות הכותרת לא מוגדרים בטבלת הסגנונות של הקובץ, ולכן
    paragraph.style מחזיר None. מזהה הסגנון עצמו כן שמור בפסקה, וקוראים אותו ישירות.
    """
    style_id = paragraph._p.style  # מאפיין של אובייקט ה־XML של python-docx
    return (style_id or "").replace(" ", "")


def classify_paragraph(paragraph: Paragraph) -> BlockKind:
    """קובעת אם פסקה היא כותרת מסמך, כותרת סעיף או פסקה רגילה לפי הסגנון שלה."""
    style_id = read_paragraph_style_id(paragraph)
    if style_id in {"Heading1", "Title"}:
        return BlockKind.DOCUMENT_TITLE
    if style_id.startswith("Heading"):
        return BlockKind.SECTION_HEADING
    return BlockKind.PARAGRAPH


def read_table_rows(table: Table) -> list[list[str]]:
    """מחזירה את תוכן הטבלה כרשימת שורות, ובכל שורה רשימת תאים."""
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        if any(cells):
            rows.append(cells)
    return rows


def read_document_blocks(document_path: Path) -> list[DocumentBlock]:
    """קוראת את כל גוף המסמך לפי הסדר: פסקאות וטבלאות משולבות."""
    document = Document(str(document_path))
    blocks: list[DocumentBlock] = []
    # iter_inner_content מחזירה פסקאות וטבלאות לפי הסדר שבו הן מופיעות
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text or text == REPEATED_SUBTITLE:
                continue
            blocks.append(DocumentBlock(kind=classify_paragraph(item), text=text))
        elif isinstance(item, Table):
            rows = read_table_rows(item)
            if rows:
                blocks.append(DocumentBlock(kind=BlockKind.TABLE, table_rows=rows))
    return blocks


def load_word_document(document_path: Path) -> WordDocument:
    """טוענת מסמך Word אחד עם המזהה, הכותרת והמבנה שלו."""
    doc_id, doc_title = parse_document_id_and_title(document_path)
    blocks = read_document_blocks(document_path)
    logger.info("נטען מסמך %s (%d יחידות תוכן)", doc_id, len(blocks))
    return WordDocument(
        doc_id=doc_id, doc_title=doc_title, file_name=document_path.name, blocks=blocks
    )


def load_word_documents(data_dir: Path) -> list[WordDocument]:
    """טוענת את כל קבצי ה־docx בתיקייה, ממוינים לפי המזהה."""
    document_paths = sorted(data_dir.glob("*.docx"))
    if not document_paths:
        raise FileNotFoundError(f"לא נמצאו קבצי docx בתיקייה {data_dir}")
    return [load_word_document(path) for path in document_paths]
