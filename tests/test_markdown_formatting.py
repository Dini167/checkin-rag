"""בדיקות לתיקון הרשימות בתשובה (docs/post_build_fixes.md, תיקונים 1–2)."""

from __future__ import annotations

from rag.markdown_formatting import format_answer_for_display, normalize_markdown_lists


def test_blank_line_added_before_numbered_list() -> None:
    text = "כך מתקנים:\n1. נכנסים\n2. לוחצים"
    assert normalize_markdown_lists(text) == "כך מתקנים:\n\n1. נכנסים\n2. לוחצים"


def test_bullet_symbol_becomes_markdown_item() -> None:
    text = "היכולות:\n• דיווח כניסה\n• הפסקה"
    assert normalize_markdown_lists(text) == "היכולות:\n\n- דיווח כניסה\n- הפסקה"


def test_several_bullets_in_one_line_are_split() -> None:
    text = "אפשר: • PDF • תמונה"
    assert normalize_markdown_lists(text) == "אפשר:\n\n- PDF\n- תמונה"


def test_correct_markdown_is_unchanged() -> None:
    text = "כך מתקנים:\n\n1. נכנסים\n2. לוחצים\n\nזהו."
    assert normalize_markdown_lists(text) == text


def test_citation_markers_are_removed() -> None:
    text = "נכנסים ל\"הנוכחות שלי\" [1].\n2. לוחצים \"תיקון\" [1][2]."
    assert format_answer_for_display(text) == "נכנסים ל\"הנוכחות שלי\".\n\n2. לוחצים \"תיקון\"."


def test_unfinished_citation_marker_at_end_of_stream_is_hidden() -> None:
    assert format_answer_for_display("נכנס מיד [1") == "נכנס מיד"
