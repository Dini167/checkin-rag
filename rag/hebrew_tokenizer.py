"""פירוק טקסט עברי למילים עבור BM25, עם הורדת קידומות זהירה (README סעיף 13).

הכלל: מורידים אות שימוש אחת (ו, ה, ב, ל, מ, ש, כ) רק אם מה שנשאר הוא מילה
שמופיעה במסמכים ויש בה לפחות 3 אותיות. המילה המקורית תמיד נשמרת.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

HEBREW_PREFIX_LETTERS = "והבלמשכ"
MINIMUM_WORD_LENGTH_AFTER_STRIPPING = 3
# מילה: רצף של אותיות (עברית ולועזית) וספרות. גרשיים ומקפים מפרידים בין מילים.
WORD_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def split_into_words(text: str) -> list[str]:
    """מפרקת טקסט למילים באותיות קטנות, בלי מילים של אות אחת."""
    return [word for word in WORD_PATTERN.findall(text.lower()) if len(word) > 1]


def build_corpus_vocabulary(texts: list[str]) -> set[str]:
    """אוסף כל המילים שמופיעות במסמכים."""
    vocabulary: set[str] = set()
    for text in texts:
        vocabulary.update(split_into_words(text))
    return vocabulary


def load_prefix_exceptions(exceptions_path: Path) -> set[str]:
    """טוענת את רשימת המילים שאסור להוריד מהן קידומת."""
    if not exceptions_path.exists():
        return set()
    return set(json.loads(exceptions_path.read_text(encoding="utf-8"))["words"])


@dataclass
class PrefixDecision:
    """החלטה על מילה אחת, לדוח הבדיקה."""

    original_word: str
    word_without_prefix: str
    accepted: bool
    reason: str


class HebrewPrefixTokenizer:
    def __init__(
        self,
        corpus_vocabulary: set[str],
        prefix_exceptions: set[str],
        strip_prefixes: bool,
    ) -> None:
        self.corpus_vocabulary = corpus_vocabulary
        self.prefix_exceptions = prefix_exceptions
        self.strip_prefixes = strip_prefixes

    def decide_prefix_stripping(self, word: str) -> PrefixDecision | None:
        """בודקת אם להוריד קידומת ממילה. None אם המילה לא מתחילה באות שימוש."""
        if word[0] not in HEBREW_PREFIX_LETTERS:
            return None
        word_without_prefix = word[1:]
        if word in self.prefix_exceptions:
            return PrefixDecision(word, word_without_prefix, False, "ברשימת החריגים")
        if len(word_without_prefix) < MINIMUM_WORD_LENGTH_AFTER_STRIPPING:
            return PrefixDecision(word, word_without_prefix, False, "קצרה מדי")
        if word_without_prefix not in self.corpus_vocabulary:
            return PrefixDecision(word, word_without_prefix, False, "לא קיימת במסמכים")
        return PrefixDecision(word, word_without_prefix, True, "קיימת במסמכים")

    def tokenize(self, text: str) -> list[str]:
        """מחזירה את המילים בטקסט, ולצד כל מילה גם את הצורה בלי קידומת אם אושרה."""
        tokens: list[str] = []
        for word in split_into_words(text):
            tokens.append(word)
            if not self.strip_prefixes:
                continue
            decision = self.decide_prefix_stripping(word)
            if decision is not None and decision.accepted:
                tokens.append(decision.word_without_prefix)
        return tokens

    def list_all_prefix_decisions(self) -> list[PrefixDecision]:
        """ההחלטות לכל המילים במסמכים שמתחילות באות שימוש (לדוח הבדיקה)."""
        decisions = [
            self.decide_prefix_stripping(word) for word in sorted(self.corpus_vocabulary)
        ]
        return [decision for decision in decisions if decision is not None]


def create_hebrew_tokenizer(
    chunk_texts: list[str], exceptions_path: Path, strip_prefixes: bool
) -> HebrewPrefixTokenizer:
    """בונה tokenizer שאוצר המילים שלו הוא כל המילים במסמכים."""
    return HebrewPrefixTokenizer(
        corpus_vocabulary=build_corpus_vocabulary(chunk_texts),
        prefix_exceptions=load_prefix_exceptions(exceptions_path),
        strip_prefixes=strip_prefixes,
    )
