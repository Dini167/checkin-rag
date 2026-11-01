"""חיפוש היברידי: וקטורי (לפי משמעות) + BM25 (לפי מילים), מאוחד ב־Reciprocal Rank Fusion.

הסף (README סעיפים 9, 19): הציון המאוחד לא מכויל, ולכן הסף מוצב על דמיון הקוסינוס
של החיפוש הווקטורי. אם אף קטע לא עובר אותו, אין תשובה, ולא קוראים למודל.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever

import config
from rag.bm25_index import HebrewBM25Index
from rag.hebrew_tokenizer import create_hebrew_tokenizer
from rag.knowledge_chunk import KnowledgeChunk
from rag.vector_store import (
    convert_llama_index_score_to_cosine,
    create_llama_index_vector_store,
    open_chroma_collection,
)

logger = logging.getLogger(__name__)

# הקבוע המקובל ב־Reciprocal Rank Fusion
RECIPROCAL_RANK_CONSTANT = 60


@dataclass
class RankedList:
    """תוצאות של שיטת חיפוש אחת לשאילתה אחת: מזהי קטעים מהטוב לפחות טוב."""

    chunk_ids: list[str]


@dataclass
class RetrievedChunk:
    chunk: KnowledgeChunk
    fused_score: float
    # הדמיון הגבוה ביותר בחיפוש הווקטורי (None אם נמצא רק ב־BM25)
    best_cosine_similarity: float | None


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    passed_relevance_threshold: bool


def fuse_ranked_lists(ranked_lists: list[RankedList]) -> dict[str, float]:
    """Reciprocal Rank Fusion: כל רשימה תורמת 1/(60+מיקום) לכל קטע שמופיע בה."""
    fused_scores: dict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list.chunk_ids, start=1):
            fused_scores[chunk_id] += 1.0 / (RECIPROCAL_RANK_CONSTANT + rank)
    return dict(fused_scores)


def create_vector_retriever(
    embedding_model: BaseEmbedding, embedding_model_name: str, top_n: int
) -> BaseRetriever:
    """Retriever וקטורי של LlamaIndex מעל האוסף ב־Chroma."""
    collection = open_chroma_collection(embedding_model_name)
    if collection.count() == 0:
        raise RuntimeError(
            f"האוסף של {embedding_model_name} ריק. הריצו קודם: python -m ingestion.build_index"
        )
    vector_index = VectorStoreIndex.from_vector_store(
        create_llama_index_vector_store(collection), embed_model=embedding_model
    )
    return vector_index.as_retriever(similarity_top_k=top_n)


class HybridRetriever:
    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        vector_retriever: BaseRetriever,
        top_k: int = config.TOP_K,
        candidates_per_retriever: int = config.CANDIDATES_PER_RETRIEVER,
        relevance_threshold: float = config.RELEVANCE_THRESHOLD,
        strip_hebrew_prefixes: bool = config.BM25_STRIP_HEBREW_PREFIXES,
    ) -> None:
        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.vector_retriever = vector_retriever
        tokenizer = create_hebrew_tokenizer(
            [chunk.text for chunk in chunks], config.PREFIX_EXCEPTIONS_FILE, strip_hebrew_prefixes
        )
        self.bm25_index = HebrewBM25Index(chunks, tokenizer)
        self.top_k = top_k
        self.candidates_per_retriever = candidates_per_retriever
        self.relevance_threshold = relevance_threshold

    def search_by_meaning(self, query: str) -> tuple[RankedList, dict[str, float]]:
        """חיפוש וקטורי. מחזיר את הדירוג ואת דמיון הקוסינוס של כל קטע."""
        nodes_with_scores = self.vector_retriever.retrieve(query)
        cosine_by_chunk_id = {
            node.node.node_id: convert_llama_index_score_to_cosine(node.score or 0.0)
            for node in nodes_with_scores
        }
        return RankedList(list(cosine_by_chunk_id)), cosine_by_chunk_id

    def search_by_keywords(self, query: str) -> RankedList:
        results = self.bm25_index.search(query, self.candidates_per_retriever)
        return RankedList([result.chunk_id for result in results])

    def run_vector_searches_in_parallel(
        self, queries: list[str]
    ) -> list[tuple[RankedList, dict[str, float]]]:
        """כל ניסוח דורש embedding משלו. מריצים במקביל כדי לא להאט את התשובה."""
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            return list(executor.map(self.search_by_meaning, queries))

    def retrieve(self, queries: list[str]) -> RetrievalResult:
        """מחפשת בכל הניסוחים, בשתי השיטות, ומאחדת לרשימה אחת של top_k קטעים."""
        ranked_lists: list[RankedList] = []
        best_cosine_by_chunk_id: dict[str, float] = {}

        for ranked_list, cosine_by_chunk_id in self.run_vector_searches_in_parallel(queries):
            ranked_lists.append(ranked_list)
            for chunk_id, cosine in cosine_by_chunk_id.items():
                best_cosine_by_chunk_id[chunk_id] = max(
                    cosine, best_cosine_by_chunk_id.get(chunk_id, 0.0)
                )
        for query in queries:
            ranked_lists.append(self.search_by_keywords(query))

        fused_scores = fuse_ranked_lists(ranked_lists)
        top_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[: self.top_k]
        retrieved_chunks = [
            RetrievedChunk(
                chunk=self.chunks_by_id[chunk_id],
                fused_score=fused_scores[chunk_id],
                best_cosine_similarity=best_cosine_by_chunk_id.get(chunk_id),
            )
            for chunk_id in top_chunk_ids
        ]
        highest_cosine = max(best_cosine_by_chunk_id.values(), default=0.0)
        passed_threshold = highest_cosine >= self.relevance_threshold
        logger.info(
            "Retrieval: %d קטעים, דמיון מרבי %.3f, סף %.2f, %s",
            len(retrieved_chunks),
            highest_cosine,
            self.relevance_threshold,
            "עבר" if passed_threshold else "לא עבר",
        )
        return RetrievalResult(chunks=retrieved_chunks, passed_relevance_threshold=passed_threshold)
