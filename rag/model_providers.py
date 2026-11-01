"""יצירת המודלים: embedding, מודל התשובה, מודל ניתוח השאלה ומודל הגיבוי."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM, ChatMessage, ChatResponse, LLMMetadata

import config

logger = logging.getLogger(__name__)

GEMINI_MAX_ANSWER_TOKENS = 2048
GEMINI_CONTEXT_WINDOW_TOKENS = 1_000_000


class MissingApiKeyError(RuntimeError):
    """חסר מפתח API בקובץ .env."""


class QuotaExceededError(RuntimeError):
    """המכסה החינמית נגמרה בכל הספקים הזמינים."""


def is_quota_exceeded_error(error: Exception) -> bool:
    """מזהה שגיאת חריגה ממכסה (HTTP 429) של Gemini או של Groq."""
    status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if status_code == 429:
        return True
    error_text = str(error)
    return "429" in error_text or "RESOURCE_EXHAUSTED" in error_text


def require_api_key(key_value: str, key_name: str) -> str:
    if not key_value:
        raise MissingApiKeyError(
            f"חסר {key_name} בקובץ .env. העתיקו את .env.example ל־.env ומלאו את המפתח."
        )
    return key_value


def get_embedding_model_name() -> str:
    """שם מודל ה־embedding הפעיל. משמש גם לשם האוסף ב־Chroma (README סעיף 21)."""
    if config.EMBEDDING_PROVIDER == "bge-m3":
        return config.LOCAL_EMBEDDING_MODEL_NAME
    return config.GEMINI_EMBEDDING_MODEL_NAME


def create_gemini_embedding_model() -> BaseEmbedding:
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    return GoogleGenAIEmbedding(
        model_name=config.GEMINI_EMBEDDING_MODEL_NAME,
        api_key=require_api_key(config.GOOGLE_API_KEY, "GOOGLE_API_KEY"),
        embed_batch_size=20,
    )


def create_local_embedding_model() -> BaseEmbedding:
    """bge-m3 רץ מקומית. דורש התקנה נוספת, ולכן הייבוא נעשה רק כשבוחרים בו."""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as import_error:
        raise RuntimeError(
            "כדי להשתמש ב־bge-m3 צריך להתקין: "
            "pip install llama-index-embeddings-huggingface"
        ) from import_error
    return HuggingFaceEmbedding(model_name=config.LOCAL_EMBEDDING_MODEL_NAME)


def create_embedding_model() -> BaseEmbedding:
    """יוצרת את מודל ה־embedding לפי EMBEDDING_PROVIDER שבהגדרות."""
    if config.EMBEDDING_PROVIDER == "bge-m3":
        return create_local_embedding_model()
    if config.EMBEDDING_PROVIDER == "gemini":
        return create_gemini_embedding_model()
    raise ValueError(f"ספק embedding לא מוכר: {config.EMBEDDING_PROVIDER}")


def create_gemini_llm(model_name: str, temperature: float) -> LLM:
    from llama_index.llms.google_genai import GoogleGenAI

    return GoogleGenAI(
        model=model_name,
        api_key=require_api_key(config.GOOGLE_API_KEY, "GOOGLE_API_KEY"),
        temperature=temperature,
        max_retries=0,  # המעבר בין מודלים והניסיון החוזר נעשים במנוע, בלי המתנות מיותרות
        # כשהערכים האלה ידועים מראש, הספרייה לא פונה ל־Google בזמן הטעינה כדי לברר
        # אותם, וכך חוסכים כשנייה וחצי לכל מודל (post_build_fixes, תיקון 17)
        max_tokens=GEMINI_MAX_ANSWER_TOKENS,
        context_window=GEMINI_CONTEXT_WINDOW_TOKENS,
    )


class LazyLoadedLlm:
    """מודל גיבוי שנטען רק בפעם הראשונה שצריך אותו.

    ספריית Groq לבדה לוקחת כמה שניות לייבוא, וברוב השיחות לא מגיעים אליה.
    """

    def __init__(self, create_llm: Callable[[], LLM]) -> None:
        self.create_llm = create_llm
        self.loaded_llm: LLM | None = None

    def get_llm(self) -> LLM:
        if self.loaded_llm is None:
            self.loaded_llm = self.create_llm()
        return self.loaded_llm

    def stream_chat(self, messages: list[ChatMessage]) -> Iterator[ChatResponse]:
        return self.get_llm().stream_chat(messages)

    @property
    def metadata(self) -> LLMMetadata:
        return self.get_llm().metadata


def create_answer_llms() -> list[LLM | LazyLoadedLlm]:
    """המודלים שכותבים את התשובה, לפי סדר הניסיון: הראשי, גיבויי Gemini ו־Groq.

    טמפרטורה נמוכה כדי להיצמד למקורות.
    """
    model_names = [name.strip() for name in (config.ANSWER_MODEL_NAME, *config.BACKUP_ANSWER_MODEL_NAMES)]
    # כל חיבור ל־Google לוקח כשנייה (אימות SSL), ולכן יוצרים את המודלים במקביל
    with ThreadPoolExecutor() as executor:
        answer_llms: list[LLM | LazyLoadedLlm] = list(
            executor.map(lambda name: create_gemini_llm(name, temperature=0.1), model_names)
        )
    if config.GROQ_API_KEY:
        answer_llms.append(LazyLoadedLlm(create_groq_llm))
    return answer_llms


def create_question_analysis_llm() -> LLM:
    """מודל מהיר לניסוח מחדש ולסיווג השאלה."""
    return create_gemini_llm(config.QUESTION_ANALYSIS_MODEL_NAME, temperature=0.0)


def create_fallback_llm() -> LLM | None:
    """מודל Groq (משמש גם כשופט בהערכה). מחזירה None אם אין מפתח Groq."""
    if not config.GROQ_API_KEY:
        logger.warning("אין GROQ_API_KEY, ולכן אין מודל Groq")
        return None
    return create_groq_llm()


def create_groq_llm() -> LLM:
    from llama_index.llms.groq import Groq

    return Groq(
        model=config.FALLBACK_MODEL_NAME,
        api_key=config.GROQ_API_KEY,
        temperature=0.1,
        max_retries=0,
    )
