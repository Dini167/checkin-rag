"""הכנת טקסט התשובה להצגה (ב־Streamlit ובאתר בעתיד). docs/post_build_fixes.md.

תיקון 1: ב־Markdown רשימה צריכה שורה ריקה לפניה, וכל פריט צריך להתחיל ב־'-' או
במספר בשורה משלו. המודל מעתיק לפעמים את הסימן '•' מהמסמכים.
תיקון 2: המודל מסמן מקורות ([1], [2]) כדי שהקוד יידע מאילו קטעים התשובה הגיעה,
אבל הסימונים לא מוצגים למשתמש.
"""

from __future__ import annotations

import re

LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")
# '•' בתחילת שורה או באמצע שורה (כמה פריטים באותה שורה)
NON_MARKDOWN_BULLET_PATTERN = re.compile(r"(?:^|\s+)[•·▪●]\s*", re.MULTILINE)
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
# שורת מקור שהמודל כתב בעצמו למרות ההוראה. את שורת המקור קובע הקוד (post_build_fixes, תיקון 23)
MODEL_SOURCE_LINE_PATTERN = re.compile(r"\s*\(המידע מתוך[^)]*\)")
# סימון מקור שלם, או סימון שעוד לא הסתיים בסוף טקסט שמוזרם (למשל "[1")
CITATION_MARKER_TO_REMOVE_PATTERN = re.compile(r"\s*\[\d+\]|\s*\[\d*$")


def remove_citation_markers(text: str) -> str:
    return CITATION_MARKER_TO_REMOVE_PATTERN.sub("", text)


def is_list_item(line: str) -> bool:
    return LIST_ITEM_PATTERN.match(line) is not None


def convert_bullets_to_markdown(text: str) -> str:
    """כל '•' הופך לפריט רשימה בשורה משלו שמתחיל ב־'- '."""
    return NON_MARKDOWN_BULLET_PATTERN.sub("\n- ", text).lstrip("\n")


def add_blank_line_before_lists(text: str) -> str:
    """מוסיפה שורה ריקה לפני פריט רשימה שבא ישר אחרי שורת טקסט רגילה."""
    lines: list[str] = []
    for line in text.split("\n"):
        previous_line = lines[-1] if lines else ""
        if is_list_item(line) and previous_line.strip() and not is_list_item(previous_line):
            lines.append("")
        lines.append(line)
    return "\n".join(lines)


def normalize_markdown_lists(text: str) -> str:
    return add_blank_line_before_lists(convert_bullets_to_markdown(text))


def format_answer_for_display(text: str) -> str:
    """כל ההכנות לפני הצגה: הסרת סימוני המקורות ושורות מקור של המודל, ותיקון הרשימות."""
    text_without_sources = MODEL_SOURCE_LINE_PATTERN.sub("", remove_citation_markers(text))
    return normalize_markdown_lists(text_without_sources).strip()
