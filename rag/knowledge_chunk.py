"""מבנה הנתונים של קטע ידע (chunk), שמשותף לשלב הטעינה ולשלב החיפוש."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class Audience(str, Enum):
    ALL = "all"
    TEAM_LEAD = "team_lead"


class ChunkKind(str, Enum):
    SECTION = "section"  # סעיף רגיל או חלק מסעיף ארוך
    GLOSSARY_TERM = "glossary_term"  # שורה אחת במילון המונחים
    FAQ_PAIR = "faq_pair"  # זוג אחד של שאלה ותשובה


@dataclass
class KnowledgeChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    kind: ChunkKind
    # הטקסט כולל בתחילתו את השורה "מסמך: ... | סעיף: ..."
    text: str
    content_hash: str
    audience: Audience = Audience.ALL
    sensitive: bool = False
    # למה התיוג נקבע כך (לדוח הבדיקה)
    tagging_reason: str = ""

    def to_metadata(self) -> dict[str, str | bool]:
        """המטא־דאטה שנשמרת עם הווקטור ב־Chroma."""
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section": self.section,
            "kind": self.kind.value,
            "audience": self.audience.value,
            "sensitive": self.sensitive,
            "content_hash": self.content_hash,
        }


def save_chunks_to_json(chunks: list[KnowledgeChunk], output_path: Path) -> None:
    """שומרת את כל הקטעים לקובץ JSON, כדי ש־BM25 והדוחות יקראו אותם בלי לטעון מחדש."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_chunks = [asdict(chunk) for chunk in chunks]
    output_path.write_text(
        json.dumps(serializable_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_chunks_from_json(input_path: Path) -> list[KnowledgeChunk]:
    """טוענת את הקטעים שנשמרו בשלב הטעינה."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"הקובץ {input_path} לא קיים. הריצו קודם: python -m ingestion.build_index"
        )
    raw_chunks = json.loads(input_path.read_text(encoding="utf-8"))
    return [
        KnowledgeChunk(
            **{
                **raw_chunk,
                "kind": ChunkKind(raw_chunk["kind"]),
                "audience": Audience(raw_chunk["audience"]),
            }
        )
        for raw_chunk in raw_chunks
    ]
