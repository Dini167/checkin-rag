"""הערכת המערכת: Retrieval (Recall@k), איכות תשובה, טיפול בתפקיד, מדרג ההפניות וזמן תגובה.

הרצה:
    python -m eval.run_eval                          # תצורת ברירת המחדל
    python -m eval.run_eval --top-k 3 --name top3    # השוואת תצורות
    python -m eval.run_eval --strip-prefixes --name prefixes
    python -m eval.run_eval --retrieval-only         # רק Recall@k, בלי מודל שפה

התוצאות נשמרות ב־eval/results/<name>.json. שאלות שכבר נענו באותה תצורה לא נשלחות
שוב למודל (מטמון), כדי לחסוך במכסה החינמית (README סעיף 20).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import config
from ingestion.audience_tagging import SENSITIVE_KEYWORDS
from rag.conversation_tracker import ConversationTracker
from rag.engine import AnswerOutcome, ChatTurn, CheckInAnswerEngine, create_answer_engine
from rag.model_providers import create_fallback_llm
from rag.prompts import UserRole

logger = logging.getLogger(__name__)

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

# התנהגות צפויה ← תוצאה צפויה מהמנוע (בדיקה אוטומטית, בלי שופט)
EXPECTED_OUTCOME_BY_BEHAVIOR = {
    "answer": AnswerOutcome.ANSWERED,
    "role_redirect": AnswerOutcome.ANSWERED,
    "sensitive_no_mention": AnswerOutcome.ANSWERED,
    "sensitive_honest": AnswerOutcome.ANSWERED,
    "not_found": AnswerOutcome.NOT_FOUND,
    "off_topic": AnswerOutcome.OFF_TOPIC,
    "manipulation": AnswerOutcome.MANIPULATION_ATTEMPT,
    "business_email": AnswerOutcome.ANSWERED,  # קודם עונים מהמסמכים, ואז מייל
}
BEHAVIORS_FOR_JUDGE = {"answer", "role_redirect", "sensitive_honest", "sensitive_no_mention"}

JUDGE_PROMPT = """\
יש לבדוק תשובה של בוט שעונה על שאלות לגבי מערכת נוכחות בשם CheckIn.
תפקיד השואל: {role}
השאלה: {question}
התנהגות נדרשת: {behavior_description}
תשובה צפויה (בקירוב): {expected_answer}
המקטעים שהבוט קיבל:
{chunks}

תשובת הבוט:
{answer}

להחזיר JSON בלבד:
{{"faithful": true/false, "correct": true/false, "role_handling_ok": true/false, "comment": "..."}}
faithful: כל מה שבתשובה נתמך במקטעים, בלי המצאות.
correct: התשובה עונה נכון על השאלה בהתאם לתשובה הצפויה.
role_handling_ok: התשובה עומדת בהתנהגות הנדרשת לתפקיד השואל.
"""

BEHAVIOR_DESCRIPTIONS = {
    "answer": "לענות רגיל מתוך המקטעים.",
    "role_redirect": "השואל עובד ושואל על פעולה של ראש צוות: להסביר שזו פעולה של ראש הצוות, "
    "ומה העובד כן יכול לעשות, בלי לתת את השלבים של ראש הצוות.",
    "sensitive_no_mention": "לא להזכיר ביוזמה את המעקב של ראש הצוות אחרי הצוות "
    "('מי במשמרת', דוח נוכחות, נקודות אדומות).",
    "sensitive_honest": "לענות בכנות, בקצרה ובטון ניטרלי, בלי להכחיש ובלי פירוט מסכים מיותר.",
}


def load_evaluation_set() -> dict:
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def retrieved_documents_match(
    retrieved_chunk_ids: list[str], engine: CheckInAnswerEngine, expected_sources: list[dict]
) -> bool:
    """האם אחד המקורות הצפויים (מסמך, ואופציונלית סעיף) נמצא בין הקטעים שאוחזרו."""
    chunks_by_id = engine.retriever.chunks_by_id
    for chunk_id in retrieved_chunk_ids:
        chunk = chunks_by_id[chunk_id]
        for expected in expected_sources:
            if chunk.doc_id == expected["doc_id"] and expected.get("section", chunk.section) == chunk.section:
                return True
    return False


def measure_retrieval_recall(engine: CheckInAnswerEngine, questions: list[dict]) -> dict:
    """Recall@k: בכמה מהשאלות שיש להן מקור צפוי, המקור נמצא בין k הקטעים הראשונים.

    החיפוש נעשה על השאלה המקורית בלבד, כדי למדוד את ה־Retrieval בנפרד מהמודל.
    """
    questions_with_sources = [question for question in questions if question["expected_sources"]]
    misses = []
    for question in questions_with_sources:
        retrieval = engine.retriever.retrieve([question["question"]])
        retrieved_ids = [item.chunk.chunk_id for item in retrieval.chunks]
        if not retrieved_documents_match(retrieved_ids, engine, question["expected_sources"]):
            misses.append(question["id"])
    hits = len(questions_with_sources) - len(misses)
    return {"recall_at_k": hits / len(questions_with_sources), "misses": misses}


def contains_support_email(text: str) -> bool:
    return config.SUPPORT_EMAIL in text


def check_behavior_automatically(question: dict, outcome: AnswerOutcome, answer_text: str) -> bool:
    """בדיקות שלא צריכות שופט: סוג התוצאה, המייל ומונחים רגישים."""
    behavior = question["expected_behavior"]
    if outcome != EXPECTED_OUTCOME_BY_BEHAVIOR[behavior]:
        return False
    if behavior == "business_email":
        return contains_support_email(answer_text)
    if contains_support_email(answer_text):
        return False  # במקרים אחרים המייל לא אמור להופיע בשאלה ראשונה
    if behavior == "sensitive_no_mention":
        return not any(keyword in answer_text for keyword in SENSITIVE_KEYWORDS)
    return True


def judge_answer(judge_llm, question: dict, answer_text: str, chunk_texts: list[str]) -> dict:
    """שופט ממודל אחר (Groq) מעריך נאמנות, נכונות וטיפול בתפקיד."""
    prompt = JUDGE_PROMPT.format(
        role=question["role"],
        question=question["question"],
        behavior_description=BEHAVIOR_DESCRIPTIONS[question["expected_behavior"]],
        expected_answer=question["expected_answer"],
        chunks="\n\n".join(chunk_texts),
        answer=answer_text,
    )
    try:
        response_text = judge_llm.complete(prompt).text
        return json.loads(response_text[response_text.index("{") : response_text.rindex("}") + 1])
    except Exception as error:
        logger.warning("השופט נכשל בשאלה %s: %s", question["id"], error)
        return {"error": str(error)}


def evaluate_single_question(
    engine: CheckInAnswerEngine, judge_llm, question: dict
) -> dict:
    """מריצה שאלה אחת בשיחה חדשה ומחזירה את כל המדדים שלה."""
    started_at = time.perf_counter()
    answer = engine.answer_question(question["question"], UserRole(question["role"]), ConversationTracker())
    answer_text = answer.read_complete_text()
    latency_seconds = time.perf_counter() - started_at

    result = {
        "id": question["id"],
        "role": question["role"],
        "question": question["question"],
        "answer": answer_text,
        "outcome": answer.outcome.value,
        "sources": [f"{source.source_name} — {source.section}" for source in answer.sources],
        "latency_seconds": round(latency_seconds, 2),
        "behavior_ok": check_behavior_automatically(question, answer.outcome, answer_text),
    }
    if judge_llm is not None and question["expected_behavior"] in BEHAVIORS_FOR_JUDGE:
        chunk_texts = [engine.retriever.chunks_by_id[i].text for i in answer.retrieved_chunk_ids]
        result["judge"] = judge_answer(judge_llm, question, answer_text, chunk_texts)
    return result


def evaluate_scenario(engine: CheckInAnswerEngine, scenario: dict) -> dict:
    """שיחה של כמה תורות: בודקת בכל תור אם המייל הופיע כצפוי."""
    tracker = ConversationTracker()
    chat_history: list[ChatTurn] = []
    email_checks = []
    last_answer = None
    for turn_question, email_expected in zip(scenario["turns"], scenario["email_expected_in_turn"]):
        last_answer = engine.answer_question(turn_question, UserRole(scenario["role"]), tracker, chat_history)
        answer_text = last_answer.read_complete_text()
        chat_history.append(last_answer.to_chat_turn(turn_question))
        email_checks.append(contains_support_email(answer_text) == email_expected)

    result = {"id": scenario["id"], "name": scenario["name"], "passed": all(email_checks), "turns": email_checks}
    if "expected_sources_last_turn" in scenario:
        result["passed"] = result["passed"] and retrieved_documents_match(
            last_answer.retrieved_chunk_ids, engine, scenario["expected_sources_last_turn"]
        )
    return result


def load_cached_results(results_path: Path) -> dict[str, dict]:
    if not results_path.exists():
        return {}
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    return {result["id"]: result for result in saved.get("questions", [])}


def summarize_answer_results(question_results: list[dict]) -> dict:
    judged = [
        result["judge"]
        for result in question_results
        if "judge" in result and "error" not in result["judge"]
    ]
    latencies = [result["latency_seconds"] for result in question_results]

    def judged_rate(metric: str) -> float | None:
        return sum(bool(judge.get(metric)) for judge in judged) / len(judged) if judged else None

    return {
        "behavior_accuracy": sum(result["behavior_ok"] for result in question_results) / len(question_results),
        "faithfulness": judged_rate("faithful"),
        "correctness": judged_rate("correct"),
        "role_handling": judged_rate("role_handling_ok"),
        "average_latency_seconds": round(statistics.mean(latencies), 2),
        "max_latency_seconds": max(latencies),
    }


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="הערכת מערכת ה־RAG")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--strip-prefixes", action="store_true", help="הורדת קידומות ב־BM25")
    parser.add_argument("--retrieval-only", action="store_true", help="רק Recall@k")
    parser.add_argument("--no-judge", action="store_true", help="בלי שופט Groq")
    parser.add_argument("--pause-seconds", type=float, default=4.0, help="המתנה בין שאלות (מכסה)")
    parser.add_argument("--name", default="default", help="שם התצורה לקובץ התוצאות")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_command_line_arguments()
    evaluation_set = load_evaluation_set()
    engine = create_answer_engine(top_k=arguments.top_k, strip_hebrew_prefixes=arguments.strip_prefixes)
    results_path = RESULTS_DIR / f"{arguments.name}.json"

    report: dict = {
        "configuration": {"top_k": arguments.top_k, "strip_prefixes": arguments.strip_prefixes},
        "retrieval": measure_retrieval_recall(engine, evaluation_set["questions"]),
    }
    print(f"Recall@{arguments.top_k}: {report['retrieval']['recall_at_k']:.0%}  (פספוסים: {report['retrieval']['misses']})")

    if not arguments.retrieval_only:
        judge_llm = None if arguments.no_judge else create_fallback_llm()
        cached_results = load_cached_results(results_path)
        question_results = []
        for question in evaluation_set["questions"]:
            if question["id"] in cached_results:
                question_results.append(cached_results[question["id"]])
                continue
            question_results.append(evaluate_single_question(engine, judge_llm, question))
            print(f"{question['id']}: {'✓' if question_results[-1]['behavior_ok'] else '✗'}")
            time.sleep(arguments.pause_seconds)
        report["questions"] = question_results
        report["answer_summary"] = summarize_answer_results(question_results)
        report["scenarios"] = [evaluate_scenario(engine, scenario) for scenario in evaluation_set["scenarios"]]
        print(json.dumps(report["answer_summary"], ensure_ascii=False, indent=2))
        for scenario_result in report["scenarios"]:
            print(f"{scenario_result['id']} {scenario_result['name']}: {'✓' if scenario_result['passed'] else '✗'}")

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"התוצאות נשמרו ב־{results_path}")


if __name__ == "__main__":
    main()
