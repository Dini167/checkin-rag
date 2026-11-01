"""מעקב אחרי מצב השיחה: מתי התחילה, אילו שאלות לא נענו, והאם המייל כבר הוצג.

המודול לא תלוי ב־Streamlit. הממשק שומר מופע אחד לכל שיחה (ב־st.session_state),
וכשהבוט יחובר לאתר, האתר ישמור אותו באותה צורה.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from rag.hebrew_tokenizer import split_into_words

# שתי שאלות שחולקות לפחות 80% מהמילים נחשבות לאותה שאלה
DUPLICATE_QUESTION_SIMILARITY = 0.8


def calculate_word_overlap(first_text: str, second_text: str) -> float:
    """דמיון בין שני טקסטים לפי חפיפת מילים (Jaccard), בין 0 ל־1."""
    first_words = set(split_into_words(first_text))
    second_words = set(split_into_words(second_text))
    if not first_words or not second_words:
        return 0.0
    return len(first_words & second_words) / len(first_words | second_words)


@dataclass
class ConversationTracker:
    started_at: datetime = field(default_factory=datetime.now)
    # השאלות השונות על המערכת שלא נמצאה להן תשובה
    distinct_unanswered_questions: list[str] = field(default_factory=list)
    # כמה שאלות על המערכת ברצף נשארו בלי תשובה (מתאפס בתשובה מוצלחת)
    consecutive_unanswered_count: int = 0
    # תשובות שנכשלו בגלל תקלה (עומס או מכסה)
    service_failures_count: int = 0
    email_already_shown: bool = False
    demo_users_offered: bool = False
    # כמה הודעות קבועות כבר נשלחו, כדי לגוון את הניסוח
    fixed_replies_given: int = 0

    @property
    def has_failed_answers(self) -> bool:
        """האם בשיחה הזו כבר הייתה שאלה שלא נענתה, מכל סיבה."""
        return bool(self.distinct_unanswered_questions) or self.service_failures_count > 0

    def record_service_failure(self) -> None:
        self.service_failures_count += 1

    def next_fixed_reply_index(self) -> int:
        """מספר רץ להודעות קבועות, כדי לבחור בכל פעם ניסוח אחר."""
        self.fixed_replies_given += 1
        return self.fixed_replies_given - 1

    def minutes_since_start(self, now: datetime) -> float:
        return (now - self.started_at).total_seconds() / 60

    def is_repeat_of_unanswered_question(self, question: str) -> bool:
        """האם השאלה זהה, או כמעט זהה, לשאלה קודמת שלא נענתה."""
        return any(
            calculate_word_overlap(question, previous_question)
            >= DUPLICATE_QUESTION_SIMILARITY
            for previous_question in self.distinct_unanswered_questions
        )

    def record_unanswered_question(self, question: str) -> bool:
        """רושמת שאלה על המערכת שלא נענתה. שאלה חוזרת לא נספרת שוב.

        מחזירה True אם השאלה נספרה.
        """
        if self.is_repeat_of_unanswered_question(question):
            return False
        self.distinct_unanswered_questions.append(question)
        self.consecutive_unanswered_count += 1
        return True

    def record_answered_question(self) -> None:
        """תשובה מוצלחת מסיימת את רצף הכישלונות."""
        self.consecutive_unanswered_count = 0

    def mark_email_shown(self) -> None:
        self.email_already_shown = True

    def mark_demo_users_offered(self) -> None:
        self.demo_users_offered = True
