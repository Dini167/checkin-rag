"""המנוע שעונה על שאלה: ניתוח ← חיפוש ← (סף) ← תשובה מוזרמת עם מקורות.

המנוע לא תלוי בממשק. תפקיד השואל ומצב השיחה מתקבלים כפרמטרים, כדי שבעתיד
אפשר יהיה לקבל אותם מההתחברות לאתר בלי לשנות את הלוגיקה (README סעיף 4).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from llama_index.core.llms import LLM, ChatMessage, MessageRole

import config
from rag.conversation_tracker import ConversationTracker
from rag.escalation_policy import (
    EscalationAction,
    decide_action_after_unanswered_question,
    decide_action_for_business_inquiry,
    decide_action_for_contact_request,
    decide_action_for_frustration,
)
from rag.document_names import get_document_friendly_name
from rag.knowledge_chunk import KnowledgeChunk, load_chunks_from_json
from rag.markdown_formatting import CITATION_PATTERN, format_answer_for_display
from rag.model_providers import (
    LazyLoadedLlm,
    create_answer_llms,
    create_embedding_model,
    create_question_analysis_llm,
    is_quota_exceeded_error,
    get_embedding_model_name,
)
from rag.prompts import (
    ANSWER_SYSTEM_PROMPT,
    BUSINESS_CONTACT_MESSAGE,
    CONTACT_REQUEST_DEFAULT_MESSAGES,
    EMAIL_ALREADY_SHOWN_MESSAGE,
    EMAIL_ESCALATION_MESSAGE,
    CHEATING_ATTEMPT_MESSAGE,
    DOUBT_AFTER_ANSWER_FALLBACK_MESSAGE,
    DOUBT_AFTER_ANSWER_PROMPT,
    EMPTY_ANSWER_FALLBACK_MESSAGE,
    FRUSTRATION_MESSAGES,
    FRUSTRATION_MESSAGES_WITH_QUESTION,
    GUEST_DEMO_USERS_OFFER,
    MANIPULATION_ATTEMPT_MESSAGES,
    NO_ANSWER_MARKER,
    NOT_FOUND_MESSAGES,
    OFF_TOPIC_MESSAGES,
    SERVICE_UNAVAILABLE_MESSAGE,
    SMALL_TALK_MESSAGES,
    UserRole,
    build_answer_prompt,
    pick_varied_message,
)
from rag.question_analyzer import (
    QuestionAnalysis,
    QuestionType,
    analyze_question,
    build_default_analysis,
)
from rag.retriever import HybridRetriever, create_vector_retriever

logger = logging.getLogger(__name__)

MAX_HISTORY_ANSWER_LENGTH = 300
MAX_QUOTED_QUESTION_LENGTH = 60
MAX_SOURCES_IN_LINE = 2
ROUNDS_OF_MODEL_ATTEMPTS = 2


class ServiceUnavailableError(RuntimeError):
    """כל המודלים נכשלו (למשל, המכסה החינמית נגמרה גם בגיבוי)."""


class AnswerOutcome(str, Enum):
    ANSWERED = "answered"
    NOT_FOUND = "not_found"
    OFF_TOPIC = "off_topic"
    MANIPULATION_ATTEMPT = "manipulation_attempt"
    SMALL_TALK = "small_talk"
    FRUSTRATION = "frustration"
    CONTACT_REPLY = "contact_reply"
    SERVICE_UNAVAILABLE = "service_unavailable"


@dataclass
class SourceReference:
    citation_number: int
    doc_id: str
    # השם הידידותי של המסמך (rag/document_names.py)
    source_name: str
    section: str


@dataclass
class ChatTurn:
    question: str
    answer: str
    # האם הבוט ענה על השאלה בהצלחה
    was_answered: bool = True
    # שאלה על המערכת (ולא תסכול, ברכה וכו'), כדי להציע לנסות אותה שוב
    is_system_question: bool = True


@dataclass
class StreamingAnswer:
    """תשובה שמוזרמת חלק אחרי חלק.

    עוברים עליה בלולאה כדי לקבל את הטקסט בהדרגה. אחרי שהלולאה מסתיימת, השדות
    full_text, sources ו־outcome מלאים.
    """

    full_text: str = ""
    # "(המידע מתוך ...)": נקבעת בקוד לפי הקטעים שהמודל ציטט, ולא בידי המודל
    source_line: str = ""
    sources: list[SourceReference] = field(default_factory=list)
    outcome: AnswerOutcome = AnswerOutcome.ANSWERED
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    question_analysis: QuestionAnalysis | None = None
    elapsed_seconds: float = 0.0
    _text_pieces: Iterator[str] = field(default_factory=lambda: iter(()), repr=False)

    def attach_text_stream(self, text_pieces: Iterator[str]) -> None:
        self._text_pieces = text_pieces

    def __iter__(self) -> Iterator[str]:
        started_at = time.perf_counter()
        for piece in self._text_pieces:
            self.full_text += piece
            yield piece
        self.full_text = format_answer_for_display(self.full_text)
        if not self.full_text:
            # תמיד מוצגת תשובה, גם אם משהו השתבש (post_build_fixes, תיקון 22)
            self.full_text = EMPTY_ANSWER_FALLBACK_MESSAGE
            yield EMPTY_ANSWER_FALLBACK_MESSAGE
        if self.source_line:
            yield f"\n\n{self.source_line}"
        self.elapsed_seconds = time.perf_counter() - started_at

    @property
    def display_text(self) -> str:
        """התשובה כפי שהיא מוצגת למשתמש: הטקסט ושורת המקור."""
        return "\n\n".join(text for text in (self.full_text, self.source_line) if text)

    def to_chat_turn(self, original_question: str) -> ChatTurn:
        """התור שנשמר בהיסטוריית השיחה (post_build_fixes, תיקונים 12–13):
        השאלה בניסוח המלא שלה ("מה זה אומר?" ← השאלה המלאה), והודעת תקלה לא נשמרת
        כדי ששאלת המשך לא תתייחס אליה."""
        question = (
            self.question_analysis.standalone_question
            if self.question_analysis
            else original_question
        )
        answer_text = "" if self.outcome == AnswerOutcome.SERVICE_UNAVAILABLE else self.full_text
        is_system_question = self.question_analysis is None or self.question_analysis.question_type in {
            QuestionType.SYSTEM_QUESTION,
            QuestionType.BUSINESS_INQUIRY,
        }
        return ChatTurn(
            question,
            answer_text,
            was_answered=self.outcome == AnswerOutcome.ANSWERED,
            is_system_question=is_system_question,
        )

    def read_complete_text(self) -> str:
        """קוראת את כל התשובה בבת אחת (שימושי בהערכה ובבדיקות)."""
        for _ in self:
            pass
        return self.full_text


def format_conversation_history(chat_history: list[ChatTurn]) -> str:
    """התורות האחרונות בשיחה, כטקסט קצר למודל."""
    recent_turns = chat_history[-config.CONVERSATION_HISTORY_TURNS :]
    lines = []
    for turn in recent_turns:
        lines.append(f"משתמש: {turn.question}")
        bot_answer = turn.answer[:MAX_HISTORY_ANSWER_LENGTH] or "(לא נענתה בגלל תקלה)"
        lines.append(f"בוט: {bot_answer}")
    return "\n".join(lines)


def is_answer_without_content(model_text: str) -> bool:
    """האם המודל סימן שאין תשובה, או כתב רק שורת מקור בלי תוכן (post_build_fixes, תיקון 19)."""
    if model_text.strip().startswith(NO_ANSWER_MARKER):
        return True
    return not format_answer_for_display(model_text)


def build_source_line(sources: list[SourceReference]) -> str:
    """שורת המקור שבסוף התשובה: עד שני מסמכים שהמודל ציטט, לפי סדר הרלוונטיות
    (הקטעים ממוינים מהרלוונטי ביותר). הקוד קובע אותה ולא המודל (post_build_fixes, תיקון 23)."""
    source_names = list(dict.fromkeys(source.source_name for source in sources))
    if not source_names:
        return ""
    return f"(המידע מתוך {' ו'.join(source_names[:MAX_SOURCES_IN_LINE])})"


def extract_cited_sources(
    answer_text: str, chunks_sent_to_model: list[KnowledgeChunk]
) -> list[SourceReference]:
    """המקורות שהמודל ציטט ([1], [2]). אם לא ציטט, כל הקטעים שנשלחו (README סעיף 22).

    כמה קטעים מאותו מסמך ומאותו סעיף מוצגים כמקור אחד.
    """
    cited_numbers = sorted({int(number) for number in CITATION_PATTERN.findall(answer_text)})
    valid_numbers = [
        number for number in cited_numbers if 1 <= number <= len(chunks_sent_to_model)
    ]
    if not valid_numbers:
        valid_numbers = list(range(1, len(chunks_sent_to_model) + 1))

    sources: list[SourceReference] = []
    for number in valid_numbers:
        chunk = chunks_sent_to_model[number - 1]
        source = SourceReference(
            citation_number=number,
            doc_id=chunk.doc_id,
            source_name=get_document_friendly_name(chunk.doc_id, chunk.doc_title),
            section=chunk.section,
        )
        if all((source.doc_id, source.section) != (s.doc_id, s.section) for s in sources):
            sources.append(source)
    return sources


def build_escalation_text(action: EscalationAction, is_business_inquiry: bool) -> str:
    """הטקסט שמתאים לפעולת ההפניה שהקוד החליט עליה."""
    if action == EscalationAction.SHOW_EMAIL:
        return BUSINESS_CONTACT_MESSAGE if is_business_inquiry else EMAIL_ESCALATION_MESSAGE
    if action == EscalationAction.OFFER_DEMO_USERS:
        return GUEST_DEMO_USERS_OFFER
    if action == EscalationAction.EMAIL_ALREADY_SHOWN:
        return EMAIL_ALREADY_SHOWN_MESSAGE
    return ""


def apply_escalation_action_to_tracker(
    action: EscalationAction, tracker: ConversationTracker
) -> None:
    if action == EscalationAction.SHOW_EMAIL:
        tracker.mark_email_shown()
    elif action == EscalationAction.OFFER_DEMO_USERS:
        tracker.mark_demo_users_offered()


class CheckInAnswerEngine:
    def __init__(
        self,
        retriever: HybridRetriever,
        answer_llms: list[LLM | LazyLoadedLlm],
        question_analysis_llm: LLM | None,
    ) -> None:
        self.retriever = retriever
        # המודלים לפי סדר הניסיון: הראשי ואחריו הגיבויים
        self.answer_llms = answer_llms
        self.question_analysis_llm = question_analysis_llm

    # ----- נקודת הכניסה -----

    def answer_question(
        self,
        question: str,
        role: UserRole,
        tracker: ConversationTracker,
        chat_history: list[ChatTurn] | None = None,
    ) -> StreamingAnswer:
        """עונה על שאלה. מחזירה תשובה מוזרמת; עוברים עליה בלולאה כדי לקבל את הטקסט."""
        answer = StreamingAnswer()
        answer.attach_text_stream(
            self.produce_answer_text(answer, question, role, tracker, chat_history or [])
        )
        return answer

    # ----- שלבי התשובה -----

    def analyze(self, question: str, conversation_history: str) -> QuestionAnalysis:
        if self.question_analysis_llm is None:
            return build_default_analysis(question)
        return analyze_question(question, conversation_history, self.question_analysis_llm)

    def produce_answer_text(
        self,
        answer: StreamingAnswer,
        question: str,
        role: UserRole,
        tracker: ConversationTracker,
        chat_history: list[ChatTurn],
    ) -> Iterator[str]:
        """מחליטה איזה סוג תשובה לתת, ומזרימה את הטקסט שלה."""
        conversation_history = format_conversation_history(chat_history)
        analysis = self.analyze(question, conversation_history)
        answer.question_analysis = analysis
        question_type = analysis.question_type

        if question_type == QuestionType.MANIPULATION_ATTEMPT:
            answer.outcome = AnswerOutcome.MANIPULATION_ATTEMPT
            yield pick_varied_message(MANIPULATION_ATTEMPT_MESSAGES, tracker.next_fixed_reply_index())
            return
        if question_type == QuestionType.SMALL_TALK:
            answer.outcome = AnswerOutcome.SMALL_TALK
            yield pick_varied_message(SMALL_TALK_MESSAGES, tracker.next_fixed_reply_index())
            return
        if question_type == QuestionType.CHEATING_ATTEMPT:
            # לא "לא מצאתי": מבהירים שאי אפשר לעקוף, ומפנים לדרך הנכונה (תיקון 30)
            answer.outcome = AnswerOutcome.MANIPULATION_ATTEMPT
            yield CHEATING_ATTEMPT_MESSAGE
            return
        if question_type == QuestionType.FRUSTRATION:
            answer.outcome = AnswerOutcome.FRUSTRATION
            yield self.build_frustration_reply(question, tracker, chat_history, conversation_history)
            return
        if question_type == QuestionType.CONTACT_REQUEST:
            answer.outcome = AnswerOutcome.CONTACT_REPLY
            yield self.build_contact_reply(role, tracker)
            return

        try:
            retrieval = self.retriever.retrieve(analysis.all_search_queries())
        except Exception:
            logger.exception("החיפוש נכשל")
            yield self.build_service_unavailable_reply(answer, tracker)
            return
        answer.retrieved_chunk_ids = [item.chunk.chunk_id for item in retrieval.chunks]

        # גם שאלה שסווגה כלא קשורה ממשיכה לחיפוש, למקרה שהסיווג טעה והיא כן במאגר
        if not retrieval.passed_relevance_threshold:
            yield self.build_reply_without_answer(answer, analysis, role, tracker)
            return

        chunks_for_model = [item.chunk for item in retrieval.chunks]
        yield from self.stream_grounded_answer(
            answer, analysis, role, tracker, chunks_for_model, conversation_history
        )
        is_answered_business_inquiry = (
            question_type == QuestionType.BUSINESS_INQUIRY
            and answer.outcome == AnswerOutcome.ANSWERED
        )
        if is_answered_business_inquiry:
            yield f"\n\n{self.build_business_contact_text(tracker)}"

    def stream_grounded_answer(
        self,
        answer: StreamingAnswer,
        analysis: QuestionAnalysis,
        role: UserRole,
        tracker: ConversationTracker,
        chunks_for_model: list[KnowledgeChunk],
        conversation_history: str,
    ) -> Iterator[str]:
        """מזרימה את תשובת המודל. אם המודל מסמן שאין תשובה, מחליפה בהודעה הקבועה."""
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=ANSWER_SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=build_answer_prompt(
                    analysis.standalone_question, role, chunks_for_model, conversation_history
                ),
            ),
        ]
        model_text_pieces = self.stream_chat_with_fallback(messages)
        try:
            # מחכים לתחילת הטקסט כדי לזהות את סימן "אין תשובה" לפני שמציגים משהו
            opening_text = self.read_opening_text(model_text_pieces)
            if opening_text.lstrip().startswith("("):
                # תשובה שמתחילה בשורת מקור: קוראים את כולה, כדי לוודא שיש בה תוכן
                opening_text += "".join(model_text_pieces)
            if is_answer_without_content(opening_text):
                for _ in model_text_pieces:  # מסיימים לקרוא את הזרם בשקט
                    pass
                yield self.build_reply_without_answer(answer, analysis, role, tracker)
                return
            yield opening_text
            full_model_text = opening_text
            for piece in model_text_pieces:
                full_model_text += piece
                yield piece
        except ServiceUnavailableError as error:
            logger.error("%s", error)
            yield self.build_service_unavailable_reply(answer, tracker)
            return

        answer.outcome = AnswerOutcome.ANSWERED
        answer.sources = extract_cited_sources(full_model_text, chunks_for_model)
        answer.source_line = build_source_line(answer.sources)
        tracker.record_answered_question()

    @staticmethod
    def read_opening_text(text_pieces: Iterator[str]) -> str:
        """קוראת מהזרם עד שיש מספיק תווים כדי לזהות את סימן "אין תשובה"."""
        opening_text = ""
        for piece in text_pieces:
            opening_text += piece
            if len(opening_text.strip()) >= len(NO_ANSWER_MARKER):
                break
        return opening_text

    def stream_chat_with_fallback(self, messages: list[ChatMessage]) -> Iterator[str]:
        """מזרימה מהמודל הראשון שמצליח. מודל שנכשל לפני שהחזיר טקסט מוחלף בבא בתור.

        אם כולם נכשלו בגלל מכסה, ממתינה כמה שניות ומנסה סבב נוסף, כי מגבלות רבות
        הן לדקה (post_build_fixes, תיקונים 8–9).
        """
        last_error: Exception | None = None
        for round_number in range(1, ROUNDS_OF_MODEL_ATTEMPTS + 1):
            for llm in self.answer_llms:
                produced_text = False
                try:
                    for response in llm.stream_chat(messages):
                        if response.delta:
                            produced_text = True
                            yield response.delta
                    return
                except Exception as error:
                    if produced_text:
                        raise ServiceUnavailableError(f"המודל נכשל באמצע התשובה: {error}") from error
                    logger.warning("המודל %s נכשל: %s", llm.metadata.model_name, str(error)[:200])
                    last_error = error
            is_last_round = round_number == ROUNDS_OF_MODEL_ATTEMPTS
            if not is_last_round and last_error is not None and is_quota_exceeded_error(last_error):
                time.sleep(config.QUOTA_RETRY_WAIT_SECONDS)
        raise ServiceUnavailableError(f"כל המודלים נכשלו: {last_error}")

    # ----- תשובות קבועות ומדרג ההפניות -----

    def build_reply_without_answer(
        self,
        answer: StreamingAnswer,
        analysis: QuestionAnalysis,
        role: UserRole,
        tracker: ConversationTracker,
    ) -> str:
        """אין תשובה במאגר. שאלה שלא קשורה ל־CheckIn מקבלת הסבר שהבוט עונה רק על
        המערכת ולא נספרת ככישלון; שאלה על המערכת מקבלת "לא נמצא מידע" (post_build_fixes, תיקון 5)."""
        if analysis.question_type == QuestionType.OFF_TOPIC:
            answer.outcome = AnswerOutcome.OFF_TOPIC
            return pick_varied_message(OFF_TOPIC_MESSAGES, tracker.next_fixed_reply_index())
        if analysis.question_type == QuestionType.BUSINESS_INQUIRY:
            # במסמכים אין מידע על רכישה, אבל לכוונת רכישה תמיד יש תשובה: הפנייה העסקית
            answer.outcome = AnswerOutcome.CONTACT_REPLY
            return self.build_business_contact_text(tracker)
        answer.outcome = AnswerOutcome.NOT_FOUND
        return self.build_not_found_reply(analysis.standalone_question, role, tracker)

    def build_not_found_reply(
        self, standalone_question: str, role: UserRole, tracker: ConversationTracker
    ) -> str:
        """הודעת "לא נמצא מידע", ובמידת הצורך גם הפניה לפי מדרג ההפניות."""
        tracker.record_unanswered_question(standalone_question)
        action = decide_action_after_unanswered_question(role, tracker, datetime.now())
        apply_escalation_action_to_tracker(action, tracker)
        escalation_text = build_escalation_text(action, is_business_inquiry=False)
        return "\n\n".join(text for text in (NOT_FOUND_MESSAGES[role], escalation_text) if text)

    def build_contact_reply(self, role: UserRole, tracker: ConversationTracker) -> str:
        """תשובה לבקשה מפורשת ליצור קשר."""
        action = decide_action_for_contact_request(role, tracker, datetime.now())
        apply_escalation_action_to_tracker(action, tracker)
        if action == EscalationAction.NONE:
            return CONTACT_REQUEST_DEFAULT_MESSAGES[role]
        return build_escalation_text(action, is_business_inquiry=False)

    def build_business_contact_text(self, tracker: ConversationTracker) -> str:
        """המייל לפנייה עסקית (או תזכורת שכבר נשלח)."""
        action = decide_action_for_business_inquiry(tracker)
        apply_escalation_action_to_tracker(action, tracker)
        return build_escalation_text(action, is_business_inquiry=True)

    def build_frustration_reply(
        self,
        message: str,
        tracker: ConversationTracker,
        chat_history: list[ChatTurn],
        conversation_history: str,
    ) -> str:
        """תגובה לתסכול או לחשד (post_build_fixes, תיקונים 24–25).

        מיד אחרי תשובה טובה: מתייחסים לחשד עצמו, בלי להתנצל. אחרי כישלון: התנצלות,
        הצעה לנסות שוב את השאלה האחרונה על המערכת, ומייל אם היו כישלונות.
        """
        if chat_history and chat_history[-1].was_answered:
            return self.respond_to_doubt_after_good_answer(message, conversation_history)

        message_index = tracker.next_fixed_reply_index()
        last_system_question = next(
            (turn.question for turn in reversed(chat_history) if turn.is_system_question), None
        )
        if last_system_question:
            reply = pick_varied_message(FRUSTRATION_MESSAGES_WITH_QUESTION, message_index)
            reply = reply.format(previous_question=last_system_question[:MAX_QUOTED_QUESTION_LENGTH])
        else:
            reply = pick_varied_message(FRUSTRATION_MESSAGES, message_index)
        action = decide_action_for_frustration(tracker)
        apply_escalation_action_to_tracker(action, tracker)
        escalation_text = build_escalation_text(action, is_business_inquiry=False)
        return "\n\n".join(text for text in (reply, escalation_text) if text)

    def respond_to_doubt_after_good_answer(self, message: str, conversation_history: str) -> str:
        """המודל מתייחס לחשד לפי התשובה הקודמת. אם הוא לא זמין, תגובה קבועה."""
        prompt = DOUBT_AFTER_ANSWER_PROMPT.format(
            conversation_history=conversation_history, message=message
        )
        try:
            return "".join(
                self.stream_chat_with_fallback([ChatMessage(role=MessageRole.USER, content=prompt)])
            )
        except ServiceUnavailableError:
            return DOUBT_AFTER_ANSWER_FALLBACK_MESSAGE

    def build_service_unavailable_reply(
        self, answer: StreamingAnswer, tracker: ConversationTracker
    ) -> str:
        answer.outcome = AnswerOutcome.SERVICE_UNAVAILABLE
        tracker.record_service_failure()
        return SERVICE_UNAVAILABLE_MESSAGE


def create_answer_engine(
    top_k: int = config.TOP_K,
    strip_hebrew_prefixes: bool = config.BM25_STRIP_HEBREW_PREFIXES,
) -> CheckInAnswerEngine:
    """בונה מנוע מוכן לעבודה. דורש שהטעינה (build_index) כבר רצה.

    הפרמטרים מאפשרים להשוות תצורות בהערכה בלי לשנות את .env.
    """
    chunks = load_chunks_from_json(config.CHUNKS_FILE)
    # כל חיבור ל־Google לוקח כשנייה, ולכן החיבורים נוצרים במקביל (post_build_fixes, תיקון 17)
    with ThreadPoolExecutor() as executor:
        vector_retriever = executor.submit(
            lambda: create_vector_retriever(
                create_embedding_model(), get_embedding_model_name(), config.CANDIDATES_PER_RETRIEVER
            )
        )
        answer_llms = executor.submit(create_answer_llms)
        question_analysis_llm = executor.submit(
            lambda: create_question_analysis_llm() if config.QUERY_REWRITING_ENABLED else None
        )
    return CheckInAnswerEngine(
        retriever=HybridRetriever(
            chunks, vector_retriever.result(), top_k=top_k, strip_hebrew_prefixes=strip_hebrew_prefixes
        ),
        answer_llms=answer_llms.result(),
        question_analysis_llm=question_analysis_llm.result(),
    )
