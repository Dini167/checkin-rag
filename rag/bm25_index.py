"""חיפוש מילולי BM25 עם ה־tokenizer העברי (README סעיפים 10, 13).

BM25Retriever של LlamaIndex מקבל רק stemmer לאנגלית וביטוי רגולרי, ולא tokenizer
מותאם. לכן משתמשים ישירות בספריית bm25s (שעליה הוא בנוי) עם מילים שפורקו מראש.
האינדקס נבנה בזיכרון בכל עלייה של המערכת: זה לוקח שבריר שנייה ל־216 קטעים.
"""

from __future__ import annotations

from dataclasses import dataclass

import bm25s

from rag.hebrew_tokenizer import HebrewPrefixTokenizer
from rag.knowledge_chunk import KnowledgeChunk


@dataclass
class KeywordSearchResult:
    chunk_id: str
    bm25_score: float


class HebrewBM25Index:
    def __init__(self, chunks: list[KnowledgeChunk], tokenizer: HebrewPrefixTokenizer) -> None:
        self.tokenizer = tokenizer
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.bm25 = bm25s.BM25()
        tokenized_chunks = [tokenizer.tokenize(chunk.text) for chunk in chunks]
        self.bm25.index(tokenized_chunks, show_progress=False)

    def search(self, query: str, top_n: int) -> list[KeywordSearchResult]:
        """מחזירה עד top_n קטעים עם ציון חיובי, מהגבוה לנמוך."""
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []
        top_n = min(top_n, len(self.chunk_ids))
        chunk_positions, scores = self.bm25.retrieve(
            [query_tokens], k=top_n, show_progress=False
        )
        return [
            KeywordSearchResult(self.chunk_ids[position], float(score))
            for position, score in zip(chunk_positions[0], scores[0])
            if score > 0
        ]
