"""בדיקות לכללי שחרור המייל (README סעיפים 14, 15)."""

from __future__ import annotations

from datetime import datetime, timedelta

from rag.conversation_tracker import ConversationTracker
from rag.escalation_policy import (
    EscalationAction,
    decide_action_after_unanswered_question,
    decide_action_for_business_inquiry,
    decide_action_for_contact_request,
    decide_action_for_frustration,
)
from rag.prompts import UserRole

CONVERSATION_START = datetime(2026, 11, 1, 10, 0)
FIVE_MINUTES_LATER = CONVERSATION_START + timedelta(minutes=5)
FORTY_MINUTES_LATER = CONVERSATION_START + timedelta(minutes=40)

DIFFERENT_QUESTIONS = [
    "איך מייצאים את הנוכחות לאקסל?",
    "האם יש אפליקציה לאייפון?",
    "איך משנים את שפת הממשק לאנגלית?",
]


def create_tracker() -> ConversationTracker:
    return ConversationTracker(started_at=CONVERSATION_START)


def record_failures_and_decide(
    tracker: ConversationTracker, questions: list[str], role: UserRole, now: datetime
) -> EscalationAction:
    action = EscalationAction.NONE
    for question in questions:
        tracker.record_unanswered_question(question)
        action = decide_action_after_unanswered_question(role, tracker, now)
        if action == EscalationAction.SHOW_EMAIL:
            tracker.mark_email_shown()
    return action


def test_no_email_after_first_failure() -> None:
    action = record_failures_and_decide(
        create_tracker(), DIFFERENT_QUESTIONS[:1], UserRole.EMPLOYEE, FIVE_MINUTES_LATER
    )
    assert action == EscalationAction.NONE


def test_email_after_three_different_failures() -> None:
    action = record_failures_and_decide(
        create_tracker(), DIFFERENT_QUESTIONS, UserRole.EMPLOYEE, FIVE_MINUTES_LATER
    )
    assert action == EscalationAction.SHOW_EMAIL


def test_repeated_question_is_counted_once() -> None:
    tracker = create_tracker()
    same_question_three_times = [DIFFERENT_QUESTIONS[0]] * 3
    action = record_failures_and_decide(
        tracker, same_question_three_times, UserRole.EMPLOYEE, FIVE_MINUTES_LATER
    )
    assert action == EscalationAction.NONE
    assert len(tracker.distinct_unanswered_questions) == 1


def test_email_after_thirty_minutes_and_two_recent_failures() -> None:
    action = record_failures_and_decide(
        create_tracker(), DIFFERENT_QUESTIONS[:2], UserRole.EMPLOYEE, FORTY_MINUTES_LATER
    )
    assert action == EscalationAction.SHOW_EMAIL


def test_no_email_after_thirty_minutes_when_last_question_was_answered() -> None:
    tracker = create_tracker()
    tracker.record_unanswered_question(DIFFERENT_QUESTIONS[0])
    tracker.record_answered_question()
    action = record_failures_and_decide(
        tracker, DIFFERENT_QUESTIONS[1:2], UserRole.EMPLOYEE, FORTY_MINUTES_LATER
    )
    assert action == EscalationAction.NONE


def test_email_shown_only_once() -> None:
    tracker = create_tracker()
    record_failures_and_decide(tracker, DIFFERENT_QUESTIONS, UserRole.EMPLOYEE, FIVE_MINUTES_LATER)
    action = record_failures_and_decide(
        tracker, ["איך מחברים את המערכת לסלאק?"], UserRole.EMPLOYEE, FIVE_MINUTES_LATER
    )
    assert action == EscalationAction.NONE
    assert (
        decide_action_for_contact_request(UserRole.EMPLOYEE, tracker, FIVE_MINUTES_LATER)
        == EscalationAction.EMAIL_ALREADY_SHOWN
    )


def test_employee_contact_request_without_failures_gets_no_email() -> None:
    action = decide_action_for_contact_request(
        UserRole.EMPLOYEE, create_tracker(), FIVE_MINUTES_LATER
    )
    assert action == EscalationAction.NONE


def test_guest_gets_no_email_and_no_demo_offer_from_failures() -> None:
    action = record_failures_and_decide(
        create_tracker(), DIFFERENT_QUESTIONS, UserRole.GUEST, FORTY_MINUTES_LATER
    )
    assert action == EscalationAction.NONE


def test_guest_gets_demo_first_then_email_when_insisting() -> None:
    tracker = create_tracker()
    first_action = decide_action_for_contact_request(UserRole.GUEST, tracker, FIVE_MINUTES_LATER)
    assert first_action == EscalationAction.OFFER_DEMO_USERS
    tracker.mark_demo_users_offered()
    second_action = decide_action_for_contact_request(UserRole.GUEST, tracker, FIVE_MINUTES_LATER)
    assert second_action == EscalationAction.SHOW_EMAIL


def test_business_inquiry_gets_email_once() -> None:
    tracker = create_tracker()
    assert decide_action_for_business_inquiry(tracker) == EscalationAction.SHOW_EMAIL
    tracker.mark_email_shown()
    assert decide_action_for_business_inquiry(tracker) == EscalationAction.EMAIL_ALREADY_SHOWN


def test_frustration_after_one_failure_gets_no_email() -> None:
    tracker = create_tracker()
    tracker.record_unanswered_question(DIFFERENT_QUESTIONS[0])
    assert decide_action_for_frustration(UserRole.EMPLOYEE, tracker, FIVE_MINUTES_LATER) == EscalationAction.NONE


def test_frustration_after_three_failures_gets_email_once() -> None:
    tracker = create_tracker()
    for question in DIFFERENT_QUESTIONS:
        tracker.record_unanswered_question(question)
    assert decide_action_for_frustration(UserRole.EMPLOYEE, tracker, FIVE_MINUTES_LATER) == EscalationAction.SHOW_EMAIL
    tracker.mark_email_shown()
    assert decide_action_for_frustration(UserRole.EMPLOYEE, tracker, FIVE_MINUTES_LATER) == EscalationAction.NONE
