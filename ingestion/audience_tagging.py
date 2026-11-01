"""תיוג אוטומטי של קהל היעד ושל מידע רגיש, ואחריו תיקונים ידניים מקובץ (README סעיף 17)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from rag.knowledge_chunk import Audience, KnowledgeChunk

logger = logging.getLogger(__name__)

# מסמכים שכולם מיועדים לראש צוות
TEAM_LEAD_DOCUMENT_IDS = {"11", "12", "13"}
# מסמכים שחלק מהסעיפים בהם לראש צוות בלבד
MIXED_AUDIENCE_DOCUMENT_IDS = {
    "03", "04", "07", "14", "15", "16", "18", "19", "22", "24", "26",
}
# "ראש צוות", "ראש הצוות", "לראש צוות", "ראשי צוותים"
TEAM_LEAD_MENTION_PATTERN = re.compile(r"ראש(?:י)?\s+(?:ה)?צוות")

# מסמך שכולו מעקב של ראש הצוות אחרי הצוות
SENSITIVE_DOCUMENT_IDS = {"12"}
# מונחים של מעקב ראש הצוות אחרי המצב הנוכחי של הצוות (README סעיף 5)
SENSITIVE_KEYWORDS = ("מי במשמרת", "נקודה אדומה", "נקודות אדומות", "דוח נוכחות")


@dataclass
class TaggingOverride:
    """תיקון ידני לתיוג. text_contains מצמצם את התיקון לקטעים שמכילים את הטקסט."""

    doc_id: str
    section: str
    reason: str
    audience: Audience | None = None
    sensitive: bool | None = None
    text_contains: str | None = None

    def matches(self, chunk: KnowledgeChunk) -> bool:
        if chunk.doc_id != self.doc_id or chunk.section != self.section:
            return False
        return self.text_contains is None or self.text_contains in chunk.text


def section_title_mentions_team_lead(section_title: str) -> bool:
    return TEAM_LEAD_MENTION_PATTERN.search(section_title) is not None


def decide_audience(chunk: KnowledgeChunk) -> tuple[Audience, str]:
    """הכלל האוטומטי לקהל היעד. מחזירה את הקהל ואת הסיבה."""
    if chunk.doc_id in TEAM_LEAD_DOCUMENT_IDS:
        return Audience.TEAM_LEAD, "מסמך שכולו לראש צוות"
    if chunk.doc_id in MIXED_AUDIENCE_DOCUMENT_IDS and section_title_mentions_team_lead(
        chunk.section
    ):
        return Audience.TEAM_LEAD, "כותרת הסעיף מציינת ראש צוות"
    return Audience.ALL, ""


def find_sensitive_keyword(chunk: KnowledgeChunk) -> str | None:
    """מחזירה את המונח הרגיש הראשון שמופיע בקטע, או None."""
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in chunk.text:
            return keyword
    return None


def decide_sensitivity(chunk: KnowledgeChunk) -> tuple[bool, str]:
    """הכלל האוטומטי לסימון רגיש. מחזירה את הסימון ואת הסיבה."""
    if chunk.doc_id in SENSITIVE_DOCUMENT_IDS:
        return True, "מסמך שכולו מעקב אחרי הצוות"
    keyword = find_sensitive_keyword(chunk)
    if keyword:
        return True, f"מכיל את המונח '{keyword}'"
    return False, ""


def apply_automatic_tagging(chunk: KnowledgeChunk) -> None:
    """מתייגת קטע לפי הכללים האוטומטיים."""
    chunk.audience, audience_reason = decide_audience(chunk)
    chunk.sensitive, sensitivity_reason = decide_sensitivity(chunk)
    reasons = [reason for reason in (audience_reason, sensitivity_reason) if reason]
    chunk.tagging_reason = "; ".join(reasons)


def load_tagging_overrides(overrides_path: Path) -> list[TaggingOverride]:
    """טוענת את קובץ התיקונים הידניים. קובץ שלא קיים פירושו שאין תיקונים."""
    if not overrides_path.exists():
        logger.warning("קובץ התיקונים %s לא נמצא, ממשיכים בלי תיקונים", overrides_path)
        return []
    raw_overrides = json.loads(overrides_path.read_text(encoding="utf-8"))["overrides"]
    return [
        TaggingOverride(
            doc_id=raw["doc_id"],
            section=raw["section"],
            reason=raw["reason"],
            audience=Audience(raw["audience"]) if "audience" in raw else None,
            sensitive=raw.get("sensitive"),
            text_contains=raw.get("text_contains"),
        )
        for raw in raw_overrides
    ]


def apply_override(chunk: KnowledgeChunk, override: TaggingOverride) -> None:
    if override.audience is not None:
        chunk.audience = override.audience
    if override.sensitive is not None:
        chunk.sensitive = override.sensitive
    chunk.tagging_reason = f"תיקון ידני: {override.reason}"


def tag_chunks(chunks: list[KnowledgeChunk], overrides: list[TaggingOverride]) -> None:
    """מתייגת את כל הקטעים: קודם אוטומטית, ואחר כך מחילה את התיקונים הידניים."""
    unused_overrides = list(overrides)
    for chunk in chunks:
        apply_automatic_tagging(chunk)
        for override in overrides:
            if override.matches(chunk):
                apply_override(chunk, override)
                if override in unused_overrides:
                    unused_overrides.remove(override)
    for override in unused_overrides:
        logger.warning(
            "תיקון ידני שלא התאים לאף קטע: מסמך %s, סעיף '%s'", override.doc_id, override.section
        )
