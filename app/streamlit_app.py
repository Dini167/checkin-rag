"""ממשק הצ'אט ב־Streamlit. הממשק רק מציג; כל הלוגיקה נמצאת ב־rag/.

הרצה (מתיקיית הפרויקט):
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

# Streamlit מריץ את הקובץ מתוך app/, ולכן מוסיפים את תיקיית הפרויקט לנתיב הייבוא
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.conversation_tracker import ConversationTracker  # noqa: E402
from rag.engine import (  # noqa: E402
    ChatTurn,
    CheckInAnswerEngine,
    SourceReference,
    StreamingAnswer,
    create_answer_engine,
)
from rag.markdown_formatting import format_answer_for_display  # noqa: E402
from rag.model_providers import MissingApiKeyError  # noqa: E402
from rag.prompts import ROLE_DISPLAY_NAMES, UserRole  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

RIGHT_TO_LEFT_STYLE = """
<style>
    .stApp, .stMarkdown, .stChatMessage, .stRadio, .stCaption,
    [data-testid="stSidebar"], [data-testid="stChatInput"] textarea {
        direction: rtl;
        text-align: right;
    }
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {
        direction: rtl;
        text-align: right;
    }
    [data-testid="stMarkdownContainer"] ol, [data-testid="stMarkdownContainer"] ul {
        padding-right: 1.5rem;
        padding-left: 0;
        margin-right: 0;
    }
</style>
"""

ROLE_SELECTOR_OPTIONS = [UserRole.EMPLOYEE, UserRole.TEAM_LEAD, UserRole.GUEST]


@dataclass
class DisplayedMessage:
    """הודעה שמוצגת במסך: מי כתב, הטקסט, והמקורות (לתשובות הבוט)."""

    author: str  # "user" או "assistant"
    text: str
    sources: list[SourceReference] = field(default_factory=list)


@st.cache_resource(show_spinner="טוען את מאגר הידע...")
def load_answer_engine() -> CheckInAnswerEngine:
    """המנוע נטען פעם אחת ומשותף לכל השיחות."""
    return create_answer_engine()


def initialize_conversation_state() -> None:
    """יוצרת את מצב השיחה בפעם הראשונה שהדף נטען."""
    if "conversation_tracker" not in st.session_state:
        st.session_state.conversation_tracker = ConversationTracker()
        st.session_state.chat_history = []  # list[ChatTurn], נשלח למנוע
        st.session_state.displayed_messages = []  # list[DisplayedMessage]


def reset_conversation_state() -> None:
    for key in ("conversation_tracker", "chat_history", "displayed_messages"):
        st.session_state.pop(key, None)
    initialize_conversation_state()


def format_source_line(source: SourceReference) -> str:
    return f"{source.source_name} — סעיף: {source.section}"


def render_sources(sources: list[SourceReference]) -> None:
    """המקורות מוסתרים מאחורי כפתור קטן, ומוצגים רק למי שלוחץ (post_build_fixes, תיקון 2)."""
    if not sources:
        return
    with st.popover("מקורות"):
        for source in sources:
            st.markdown(f"- {format_source_line(source)}")


def render_previous_messages() -> None:
    for message in st.session_state.displayed_messages:
        with st.chat_message(message.author):
            st.markdown(message.text)
            render_sources(message.sources)


def render_sidebar() -> UserRole:
    """בורר התפקיד וכפתור שיחה חדשה. מחזירה את התפקיד שנבחר."""
    with st.sidebar:
        st.header("הגדרות")
        selected_role = st.radio(
            "התפקיד שלי",
            options=ROLE_SELECTOR_OPTIONS,
            format_func=lambda role: ROLE_DISPLAY_NAMES[role],
        )
        st.caption(
            "בשלב הזה התפקיד נבחר כאן. כשהבוט יחובר לאתר, הוא יגיע אוטומטית מההתחברות."
        )
        if st.button("שיחה חדשה"):
            reset_conversation_state()
            st.rerun()
    return selected_role


def render_streaming_answer(answer: StreamingAnswer) -> None:
    """מציגה את התשובה בהדרגה. בכל עדכון מכינים את הטקסט שהגיע עד עכשיו להצגה
    (רשימות תקינות, בלי סימוני מקורות), כבר בזמן ההזרמה (docs/post_build_fixes.md)."""
    answer_placeholder = st.empty()
    text_so_far = ""
    for piece in answer:
        text_so_far += piece
        answer_placeholder.markdown(format_answer_for_display(text_so_far))
    answer_placeholder.markdown(answer.display_text)


def handle_new_question(
    engine: CheckInAnswerEngine, question: str, role: UserRole
) -> None:
    """מציגה את השאלה, מזרימה את התשובה ושומרת את שתיהן בהיסטוריה."""
    st.session_state.displayed_messages.append(DisplayedMessage("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    answer = engine.answer_question(
        question=question,
        role=role,
        tracker=st.session_state.conversation_tracker,
        chat_history=st.session_state.chat_history,
    )
    with st.chat_message("assistant"):
        render_streaming_answer(answer)
        render_sources(answer.sources)

    st.session_state.chat_history.append(answer.to_chat_turn(question))
    st.session_state.displayed_messages.append(
        DisplayedMessage("assistant", answer.display_text, answer.sources)
    )


def main() -> None:
    st.set_page_config(page_title="עוזר CheckIn", page_icon="🕘")
    st.markdown(RIGHT_TO_LEFT_STYLE, unsafe_allow_html=True)
    st.title("עוזר CheckIn")
    st.caption("שאלות על מערכת הנוכחות. התשובות מבוססות על מאגר הידע בלבד.")

    initialize_conversation_state()
    role = render_sidebar()

    try:
        engine = load_answer_engine()
    except (MissingApiKeyError, FileNotFoundError, RuntimeError) as error:
        st.error(str(error))
        st.stop()

    render_previous_messages()
    question = st.chat_input("מה תרצו לשאול?")
    if question:
        handle_new_question(engine, question, role)


main()
