"""כללי מדרג ההפניות: מתי מציגים את המייל (README סעיפים 7, 14, 15).

כל ההחלטות כאן נעשות בקוד ולא במודל, ולכן הן יציבות ואי אפשר "לשכנע" אותן.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from config import (
    FAILURES_BEFORE_EMAIL,
    LONG_CONVERSATION_MINUTES,
    RECENT_FAILURES_IN_LONG_CONVERSATION,
)
from rag.conversation_tracker import ConversationTracker
from rag.prompts import UserRole


class EscalationAction(str, Enum):
    NONE = "none"  # ברירת מחדל: לנסח אחרת / לפנות לראש הצוות
    OFFER_DEMO_USERS = "offer_demo_users"
    SHOW_EMAIL = "show_email"
    EMAIL_ALREADY_SHOWN = "email_already_shown"


def has_reached_failure_limit(tracker: ConversationTracker) -> bool:
    """3 שאלות שונות על המערכת בלי תשובה."""
    return len(tracker.distinct_unanswered_questions) >= FAILURES_BEFORE_EMAIL


def is_long_conversation_without_progress(
    tracker: ConversationTracker, now: datetime
) -> bool:
    """עברו 30 דקות, וגם 2 השאלות האחרונות על המערכת נשארו בלי תשובה."""
    return (
        tracker.minutes_since_start(now) >= LONG_CONVERSATION_MINUTES
        and tracker.consecutive_unanswered_count >= RECENT_FAILURES_IN_LONG_CONVERSATION
    )


def is_email_allowed_for_registered_user(
    tracker: ConversationTracker, now: datetime
) -> bool:
    return has_reached_failure_limit(tracker) or is_long_conversation_without_progress(
        tracker, now
    )


def decide_action_after_unanswered_question(
    role: UserRole, tracker: ConversationTracker, now: datetime
) -> EscalationAction:
    """מה מוסיפים להודעת "לא נמצא מידע", אחרי שהשאלה כבר נרשמה כלא נענתה."""
    # גולש לא מקבל מייל דרך כישלונות, ומשתמשי הדמו לא קשורים לשאלה שאין לה תשובה
    # (post_build_fixes, תיקון 6). הם מוצעים רק כשהוא מתעניין איך המערכת נראית.
    if role == UserRole.GUEST or tracker.email_already_shown:
        return EscalationAction.NONE
    if not is_email_allowed_for_registered_user(tracker, now):
        return EscalationAction.NONE
    return EscalationAction.SHOW_EMAIL


def decide_action_for_contact_request(
    role: UserRole, tracker: ConversationTracker, now: datetime
) -> EscalationAction:
    """בקשה מפורשת ליצור קשר."""
    if tracker.email_already_shown:
        return EscalationAction.EMAIL_ALREADY_SHOWN
    if role == UserRole.GUEST:
        # גולש שכבר קיבל את משתמשי הדמו ועדיין מבקש ליצור קשר מקבל את המייל
        if tracker.demo_users_offered:
            return EscalationAction.SHOW_EMAIL
        return EscalationAction.OFFER_DEMO_USERS
    if is_email_allowed_for_registered_user(tracker, now):
        return EscalationAction.SHOW_EMAIL
    return EscalationAction.NONE


def decide_action_for_business_inquiry(tracker: ConversationTracker) -> EscalationAction:
    """כוונת רכישה מפורשת (רכישה, מחיר, הדגמה): אחרי התשובה מהמסמכים, או במקומה אם
    אין במסמכים מידע, מציגים את המייל כפנייה עסקית, בכל תפקיד (post_build_fixes, תיקונים 13, 18)."""
    if tracker.email_already_shown:
        return EscalationAction.EMAIL_ALREADY_SHOWN
    return EscalationAction.SHOW_EMAIL


def decide_action_for_frustration(tracker: ConversationTracker) -> EscalationAction:
    """תסכול אחרי תשובה שנכשלה הוא בדיוק המקרה הקיצוני שבו מציגים את המייל."""
    if tracker.has_failed_answers and not tracker.email_already_shown:
        return EscalationAction.SHOW_EMAIL
    return EscalationAction.NONE
