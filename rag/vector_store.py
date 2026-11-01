"""גישה ל־Chroma: אוסף נפרד לכל מודל embedding, וסנכרון הקטעים בלי כפילויות."""

from __future__ import annotations

import logging
import math
import re
import time

import chromadb
from chromadb.api.models.Collection import Collection
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from rag.knowledge_chunk import KnowledgeChunk
from rag.model_providers import is_quota_exceeded_error

logger = logging.getLogger(__name__)

COLLECTION_NAME_PREFIX = "checkin"
EMBEDDING_BATCH_SIZE = 20
QUOTA_WAIT_SECONDS = 60
QUOTA_RETRY_ATTEMPTS = 4


def build_collection_name(embedding_model_name: str) -> str:
    """שם אוסף שכולל את שם המודל. Chroma מתירה רק אותיות לועזיות, ספרות, נקודה, מקף וקו תחתון."""
    safe_model_name = re.sub(r"[^a-zA-Z0-9._-]", "-", embedding_model_name)
    return f"{COLLECTION_NAME_PREFIX}_{safe_model_name}".strip("-_.")


def open_chroma_collection(embedding_model_name: str) -> Collection:
    """פותחת (או יוצרת) את האוסף של המודל, עם מרחק קוסינוס."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=build_collection_name(embedding_model_name),
        configuration={"hnsw": {"space": "cosine"}},
    )


def create_llama_index_vector_store(collection: Collection) -> ChromaVectorStore:
    return ChromaVectorStore(chroma_collection=collection)


def convert_llama_index_score_to_cosine(score: float) -> float:
    """ממירה את הציון ש־LlamaIndex מחזיר מ־Chroma לדמיון קוסינוס.

    ChromaVectorStore מחזיר exp(-distance), ובמרחב קוסינוס distance = 1 - cosine.
    לכן cosine = 1 + ln(score). כך הסף בהגדרות הוא דמיון קוסינוס רגיל (0 עד 1).
    """
    if score <= 0:
        return 0.0
    return 1.0 + math.log(score)


def convert_chunk_to_text_node(chunk: KnowledgeChunk) -> TextNode:
    return TextNode(id_=chunk.chunk_id, text=chunk.text, metadata=chunk.to_metadata())


def read_stored_content_hashes(collection: Collection) -> dict[str, str]:
    """מחזירה מזהה קטע ← hash של התוכן, לכל הקטעים שכבר שמורים באוסף."""
    stored = collection.get(include=["metadatas"])
    return {
        chunk_id: str(metadata.get("content_hash", ""))
        for chunk_id, metadata in zip(stored["ids"], stored["metadatas"] or [])
    }


def embed_and_store_batch(
    batch: list[KnowledgeChunk],
    vector_store: ChromaVectorStore,
    embedding_model: BaseEmbedding,
) -> None:
    """מחשבת embeddings לקבוצה אחת ושומרת אותה מיד, כך שהתקדמות לא הולכת לאיבוד.

    המכסה החינמית מוגבלת לדקה. בחריגה מחכים דקה ומנסים שוב את אותה קבוצה.
    """
    nodes = [convert_chunk_to_text_node(chunk) for chunk in batch]
    for attempt in range(1, QUOTA_RETRY_ATTEMPTS + 1):
        try:
            embeddings = embedding_model.get_text_embedding_batch(
                [node.get_content() for node in nodes]
            )
            break
        except Exception as error:
            if not is_quota_exceeded_error(error) or attempt == QUOTA_RETRY_ATTEMPTS:
                raise
            logger.warning("חריגה ממכסת הדקה, ממתינים %d שניות", QUOTA_WAIT_SECONDS)
            time.sleep(QUOTA_WAIT_SECONDS)
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding
    vector_store.add(nodes)


def sync_chunks_to_collection(
    chunks: list[KnowledgeChunk],
    collection: Collection,
    embedding_model: BaseEmbedding,
) -> None:
    """מסנכרנת את האוסף עם הקטעים: מוסיפה חדשים, מעדכנת ששונו, ומוחקת שנעלמו.

    רק קטעים חדשים או ששונו נשלחים ל־embedding, כדי לחסוך במכסה החינמית.
    """
    stored_hashes = read_stored_content_hashes(collection)
    current_chunk_ids = {chunk.chunk_id for chunk in chunks}

    chunks_to_embed = [
        chunk for chunk in chunks if stored_hashes.get(chunk.chunk_id) != chunk.content_hash
    ]
    obsolete_chunk_ids = [
        chunk_id for chunk_id in stored_hashes if chunk_id not in current_chunk_ids
    ]
    changed_chunk_ids = [
        chunk.chunk_id for chunk in chunks_to_embed if chunk.chunk_id in stored_hashes
    ]

    ids_to_delete = obsolete_chunk_ids + changed_chunk_ids
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    vector_store = create_llama_index_vector_store(collection)
    for batch_start in range(0, len(chunks_to_embed), EMBEDDING_BATCH_SIZE):
        batch = chunks_to_embed[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        embed_and_store_batch(batch, vector_store, embedding_model)
        logger.info("נשמרו %d/%d קטעים", batch_start + len(batch), len(chunks_to_embed))

    logger.info(
        "סנכרון האוסף: %d חדשים/עודכנו, %d נמחקו, %d ללא שינוי",
        len(chunks_to_embed),
        len(obsolete_chunk_ids),
        len(chunks) - len(chunks_to_embed),
    )
