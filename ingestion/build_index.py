"""סקריפט הטעינה: מסמכים ← קטעים ← תיוג ← דוחות ← embeddings ב־Chroma.

הרצה:
    python -m ingestion.build_index              # טעינה מלאה
    python -m ingestion.build_index --reports-only  # רק קטעים ודוחות, בלי embeddings

אפשר להריץ שוב ושוב: קטעים שלא השתנו לא נשלחים שוב ל־embedding, ואין כפילויות.
"""

from __future__ import annotations

import argparse
import logging
import sys

import config
from ingestion.audience_tagging import load_tagging_overrides, tag_chunks
from ingestion.chunk_builder import build_document_chunks
from ingestion.docx_reader import load_word_documents
from ingestion.reports import build_prefix_report, build_tagging_report, write_report
from rag.hebrew_tokenizer import create_hebrew_tokenizer
from rag.knowledge_chunk import KnowledgeChunk, save_chunks_to_json
from rag.model_providers import (
    MissingApiKeyError,
    create_embedding_model,
    get_embedding_model_name,
    is_quota_exceeded_error,
)
from rag.vector_store import open_chroma_collection, sync_chunks_to_collection

logger = logging.getLogger(__name__)

TAGGING_REPORT_FILE = config.DOCS_DIR / "tagging_report.md"
PREFIX_REPORT_FILE = config.DOCS_DIR / "prefix_report.md"


def build_tagged_chunks() -> list[KnowledgeChunk]:
    """טוענת את המסמכים, מחלקת לקטעים ומתייגת."""
    documents = load_word_documents(config.DATA_DIR)
    chunks = [chunk for document in documents for chunk in build_document_chunks(document)]
    tag_chunks(chunks, load_tagging_overrides(config.AUDIENCE_OVERRIDES_FILE))
    logger.info("נבנו %d קטעים מתוך %d מסמכים", len(chunks), len(documents))
    return chunks


def write_review_reports(chunks: list[KnowledgeChunk]) -> None:
    """כותבת את דוח התיוג ואת דוח הקידומות לבדיקה ידנית."""
    write_report(build_tagging_report(chunks), TAGGING_REPORT_FILE)
    tokenizer = create_hebrew_tokenizer(
        [chunk.text for chunk in chunks], config.PREFIX_EXCEPTIONS_FILE, strip_prefixes=True
    )
    write_report(build_prefix_report(tokenizer), PREFIX_REPORT_FILE)
    logger.info("הדוחות נכתבו: %s, %s", TAGGING_REPORT_FILE, PREFIX_REPORT_FILE)


def store_chunk_embeddings(chunks: list[KnowledgeChunk]) -> None:
    """מחשבת embeddings לקטעים החדשים ושומרת ב־Chroma."""
    embedding_model_name = get_embedding_model_name()
    logger.info("מודל embedding: %s", embedding_model_name)
    collection = open_chroma_collection(embedding_model_name)
    sync_chunks_to_collection(chunks, collection, create_embedding_model())


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="טעינת המסמכים למאגר הידע")
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="בונה קטעים ודוחות בלבד, בלי embeddings (לא צריך מפתח API)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_command_line_arguments()

    chunks = build_tagged_chunks()
    save_chunks_to_json(chunks, config.CHUNKS_FILE)
    write_review_reports(chunks)
    if arguments.reports_only:
        return 0

    try:
        store_chunk_embeddings(chunks)
    except MissingApiKeyError as error:
        logger.error("%s", error)
        return 1
    except Exception as error:
        if is_quota_exceeded_error(error):
            logger.error(
                "המכסה החינמית של ה־embedding נגמרה. הקטעים שכבר נשמרו לא יחושבו "
                "שוב, אז אפשר פשוט להריץ שוב מאוחר יותר."
            )
            return 1
        raise
    logger.info("הטעינה הסתיימה בהצלחה")
    return 0


if __name__ == "__main__":
    sys.exit(main())
