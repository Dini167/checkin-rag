"""ניתוח השאלה בקריאה אחת למודל מהיר: ניסוח עצמאי, ניסוחים חלופיים וסיווג."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from llama_index.core.llms import LLM

from rag.prompts import build_question_analysis_prompt

logger = logging.getLogger(__name__)

MAX_ALTERNATIVE_PHRASINGS = 2
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class QuestionType(str, Enum):
    SYSTEM_QUESTION = "system_question"
    CONTACT_REQUEST = "contact_request"
    BUSINESS_INQUIRY = "business_inquiry"
    MANIPULATION_ATTEMPT = "manipulation_attempt"
    SMALL_TALK = "small_talk"
    FRUSTRATION = "frustration"
    CHEATING_ATTEMPT = "cheating_attempt"
    OFF_TOPIC = "off_topic"


@dataclass
class QuestionAnalysis:
    standalone_question: str
    alternative_phrasings: list[str] = field(default_factory=list)
    question_type: QuestionType = QuestionType.SYSTEM_QUESTION

    def all_search_queries(self) -> list[str]:
        """השאלה העצמאית והניסוחים החלופיים, בלי כפילויות."""
        queries = [self.standalone_question, *self.alternative_phrasings]
        return list(dict.fromkeys(query for query in queries if query.strip()))


def build_default_analysis(question: str) -> QuestionAnalysis:
    """ניתוח ברירת מחדל כשהמודל לא זמין או החזיר תשובה לא תקינה: מחפשים את השאלה כמו שהיא."""
    return QuestionAnalysis(standalone_question=question)


def parse_question_type(raw_type: str) -> QuestionType:
    try:
        return QuestionType(raw_type)
    except ValueError:
        return QuestionType.SYSTEM_QUESTION


def parse_analysis_response(response_text: str, original_question: str) -> QuestionAnalysis:
    """מפרקת את ה־JSON שהמודל החזיר. כל שדה חסר מקבל ערך ברירת מחדל בטוח."""
    match = JSON_OBJECT_PATTERN.search(response_text)
    if not match:
        raise ValueError("לא נמצא JSON בתשובת המודל")
    parsed = json.loads(match.group(0))
    alternatives = [
        str(phrasing)
        for phrasing in parsed.get("alternative_phrasings", [])
        if str(phrasing).strip()
    ][:MAX_ALTERNATIVE_PHRASINGS]
    return QuestionAnalysis(
        standalone_question=str(parsed.get("standalone_question") or original_question),
        alternative_phrasings=alternatives,
        question_type=parse_question_type(str(parsed.get("question_type", ""))),
    )


def analyze_question(
    question: str, conversation_history: str, analysis_llm: LLM
) -> QuestionAnalysis:
    """מנתחת את השאלה. אם המודל נכשל, ממשיכים עם השאלה המקורית במקום להיתקע."""
    prompt = build_question_analysis_prompt(question, conversation_history)
    try:
        response = analysis_llm.complete(prompt)
        analysis = parse_analysis_response(response.text, question)
    except Exception as error:  # כל כישלון כאן לא אמור לעצור את התשובה
        logger.warning("ניתוח השאלה נכשל, ממשיכים עם השאלה המקורית: %s", error)
        return build_default_analysis(question)
    logger.info(
        "ניתוח שאלה: סוג=%s, ניסוח עצמאי='%s', חלופות=%s",
        analysis.question_type.value,
        analysis.standalone_question,
        analysis.alternative_phrasings,
    )
    return analysis
