"""הגדרות מרכזיות של המערכת.

כל ערך שאפשר לשנות בלי לגעת בקוד נקרא מקובץ .env (ראו .env.example).
מפתחות API לעולם לא נכתבים כאן.
"""

from __future__ import annotations

import os
from pathlib import Path

import truststore
from dotenv import load_dotenv

# אימות תעודות SSL דרך מאגר התעודות של Windows ולא דרך המאגר של Python.
# בלי זה, ברשת עם סינון (כמו NetFree) Python 3.13 דוחה את תעודת הסינון והחיבור
# ל־Gemini ול־Groq נכשל (ראו docs/issues_log.md).
truststore.inject_into_ssl()

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def read_text_setting(name: str, default: str) -> str:
    """מחזירה הגדרת טקסט מ־.env, או ערך ברירת מחדל אם היא לא הוגדרה."""
    value = os.getenv(name, "").strip()
    return value or default


def read_int_setting(name: str, default: int) -> int:
    """מחזירה הגדרה מספרית שלמה מ־.env."""
    return int(read_text_setting(name, str(default)))


def read_float_setting(name: str, default: float) -> float:
    """מחזירה הגדרה מספרית עשרונית מ־.env."""
    return float(read_text_setting(name, str(default)))


def read_bool_setting(name: str, default: bool) -> bool:
    """מחזירה הגדרה בוליאנית מ־.env (true/false)."""
    return read_text_setting(name, str(default)).lower() in {"true", "1", "yes"}


# --- מפתחות API ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- מודלים ---
# המודל שכותב את התשובה. flash-lite עונה תוך כשנייה ויש לו מכסה יומית גדולה;
# ל־gemini-3.6-flash יש רק 20 בקשות ביום והוא איטי (docs/post_build_fixes.md, תיקונים 7–8).
ANSWER_MODEL_NAME = read_text_setting("ANSWER_MODEL_NAME", "gemini-3.5-flash-lite")
# מודלי גיבוי לפי הסדר. המכסה של Gemini נפרדת לכל מודל, ולכן כל גיבוי מוסיף מכסה.
BACKUP_ANSWER_MODEL_NAMES = read_text_setting(
    "BACKUP_ANSWER_MODEL_NAMES", "gemini-3.1-flash-lite,gemini-3.6-flash"
).split(",")
# כשכל המודלים בחריגה ממכסה: ממתינים ומנסים סבב נוסף (מגבלות רבות הן לדקה)
QUOTA_RETRY_WAIT_SECONDS = 5
# מודל שופט בהערכה: שונה ממודל התשובה (Groq חסום ב־NetFree)
JUDGE_MODEL_NAME = read_text_setting("JUDGE_MODEL_NAME", "gemini-3.1-flash-lite")
# מודל קל ומהיר לניסוח מחדש ולסיווג השאלה, כדי לא להאט את התשובה.
QUESTION_ANALYSIS_MODEL_NAME = read_text_setting(
    "QUESTION_ANALYSIS_MODEL_NAME", "gemini-3.5-flash-lite"
)
# מודל גיבוי ב־Groq, כשנגמרת המכסה של Gemini. משמש גם כשופט בהערכה.
FALLBACK_MODEL_NAME = read_text_setting("FALLBACK_MODEL_NAME", "openai/gpt-oss-120b")

# ספק ה־embedding: "gemini" או "bge-m3" (מקומי, דורש התקנה נוספת)
EMBEDDING_PROVIDER = read_text_setting("EMBEDDING_PROVIDER", "gemini")
GEMINI_EMBEDDING_MODEL_NAME = read_text_setting(
    "GEMINI_EMBEDDING_MODEL_NAME", "gemini-embedding-001"
)
LOCAL_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# --- נתיבים ---
DATA_DIR = PROJECT_ROOT / read_text_setting("DATA_DIR", "data")
STORAGE_DIR = PROJECT_ROOT / read_text_setting("STORAGE_DIR", "storage")
CHROMA_DIR = STORAGE_DIR / "chroma"
CHUNKS_FILE = STORAGE_DIR / "chunks.json"
DOCS_DIR = PROJECT_ROOT / "docs"
AUDIENCE_OVERRIDES_FILE = PROJECT_ROOT / "ingestion" / "audience_overrides.json"
PREFIX_EXCEPTIONS_FILE = PROJECT_ROOT / "ingestion" / "prefix_exceptions.json"

# --- חלוקה ל־chunks ---
CHUNK_SIZE_TOKENS = read_int_setting("CHUNK_SIZE_TOKENS", 500)
CHUNK_OVERLAP_TOKENS = read_int_setting("CHUNK_OVERLAP_TOKENS", 75)  # 15%

# --- Retrieval ---
TOP_K = read_int_setting("TOP_K", 5)
# כמה מועמדים כל שיטת חיפוש מחזירה לפני האיחוד
CANDIDATES_PER_RETRIEVER = read_int_setting("CANDIDATES_PER_RETRIEVER", 10)
# סף דמיון קוסינוס בחיפוש הווקטורי. מתחתיו קטע לא נחשב רלוונטי.
# כויל לפי ההערכה: שאלות עם תשובה ≥ 0.785, שאלות בלי תשובה ≤ 0.695 (docs/issues_log.md)
RELEVANCE_THRESHOLD = read_float_setting("RELEVANCE_THRESHOLD", 0.72)
# הורדת קידומות עבריות ב־BM25 (ראו README סעיף 13). כבוי כברירת מחדל עד המדידה.
BM25_STRIP_HEBREW_PREFIXES = read_bool_setting("BM25_STRIP_HEBREW_PREFIXES", False)
# ניסוח מחדש של השאלה (README סעיפים 10, 16)
QUERY_REWRITING_ENABLED = read_bool_setting("QUERY_REWRITING_ENABLED", True)

# --- מדרג הפניות למייל (README סעיפים 14, 15) ---
SUPPORT_EMAIL = read_text_setting("SUPPORT_EMAIL", "d533137135@gmail.com")
FAILURES_BEFORE_EMAIL = 3
LONG_CONVERSATION_MINUTES = 30
RECENT_FAILURES_IN_LONG_CONVERSATION = 2
CONVERSATION_HISTORY_TURNS = 3
