"""כתיבת דוחות בדיקה ידנית: תוצאות התיוג והחלטות הורדת הקידומות."""

from __future__ import annotations

from pathlib import Path

from rag.hebrew_tokenizer import HebrewPrefixTokenizer
from rag.knowledge_chunk import Audience, KnowledgeChunk


def shorten_text(text: str, max_length: int = 70) -> str:
    """מחזירה את תחילת גוף הקטע (בלי שורת ההקשר) בשורה אחת."""
    body = text.split("\n", 1)[-1].replace("\n", " ").replace("|", "/")
    return body if len(body) <= max_length else body[:max_length] + "…"


def format_chunk_row(chunk: KnowledgeChunk) -> str:
    audience_label = "ראש צוות" if chunk.audience == Audience.TEAM_LEAD else "כולם"
    sensitive_label = "כן" if chunk.sensitive else ""
    return (
        f"| {chunk.doc_id} | {chunk.section} | {chunk.kind.value} | {audience_label} "
        f"| {sensitive_label} | {chunk.tagging_reason} | {shorten_text(chunk.text)} |"
    )


def build_tagging_report(chunks: list[KnowledgeChunk]) -> str:
    """דוח Markdown: סיכום, רשימת הקטעים הרגישים, וכל הקטעים לפי מסמך."""
    table_header = (
        "| מסמך | סעיף | סוג | קהל | רגיש | סיבה | תחילת הקטע |\n"
        "|---|---|---|---|---|---|---|"
    )
    sensitive_chunks = [chunk for chunk in chunks if chunk.sensitive]
    team_lead_chunks = [chunk for chunk in chunks if chunk.audience == Audience.TEAM_LEAD]
    lines = [
        "# דוח תיוג קטעים",
        "",
        "נוצר אוטומטית בהרצת `python -m ingestion.build_index`. לתיקון תיוג: "
        "`ingestion/audience_overrides.json`.",
        "",
        f"- סך הכול קטעים: {len(chunks)}",
        f"- לראש צוות בלבד: {len(team_lead_chunks)}",
        f"- רגישים: {len(sensitive_chunks)}",
        "",
        "## קטעים שסומנו רגישים",
        "",
        table_header,
        *[format_chunk_row(chunk) for chunk in sensitive_chunks],
        "",
        "## קטעים שסומנו לראש צוות בלבד",
        "",
        table_header,
        *[format_chunk_row(chunk) for chunk in team_lead_chunks],
        "",
        "## כל הקטעים",
        "",
        table_header,
        *[format_chunk_row(chunk) for chunk in chunks],
    ]
    return "\n".join(lines) + "\n"


def build_prefix_report(tokenizer: HebrewPrefixTokenizer) -> str:
    """דוח Markdown של כל המילים שמתחילות באות שימוש, ומה הוחלט לגבי כל אחת."""
    decisions = tokenizer.list_all_prefix_decisions()
    accepted_decisions = [decision for decision in decisions if decision.accepted]
    rejected_decisions = [decision for decision in decisions if not decision.accepted]
    lines = [
        "# דוח הורדת קידומות",
        "",
        "בדקו את הרשימה הראשונה: כל שורה שבה המילה בלי האות הראשונה היא מילה אחרת, "
        "שלא קשורה למקור, מוסיפים את המילה המקורית ל־`ingestion/prefix_exceptions.json`.",
        "",
        f"## קידומת תוסר ({len(accepted_decisions)} מילים)",
        "",
        "| מילה במסמך | אחרי הורדת הקידומת |",
        "|---|---|",
        *[
            f"| {decision.original_word} | {decision.word_without_prefix} |"
            for decision in accepted_decisions
        ],
        "",
        f"## המילה נשארת כמו שהיא ({len(rejected_decisions)} מילים)",
        "",
        "| מילה במסמך | בלי האות הראשונה | סיבה |",
        "|---|---|---|",
        *[
            f"| {decision.original_word} | {decision.word_without_prefix} | {decision.reason} |"
            for decision in rejected_decisions
        ],
    ]
    return "\n".join(lines) + "\n"


def write_report(report_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
