"""בדיקות להורדת הקידומות הזהירה (README סעיף 13)."""

from __future__ import annotations

from rag.hebrew_tokenizer import HebrewPrefixTokenizer, split_into_words

CORPUS_VOCABULARY = {"בקשות", "הבקשות", "מחלה", "בקשה", "קשה", "שכר"}


def create_tokenizer(prefix_exceptions: set[str] | None = None) -> HebrewPrefixTokenizer:
    return HebrewPrefixTokenizer(
        corpus_vocabulary=CORPUS_VOCABULARY,
        prefix_exceptions=prefix_exceptions or set(),
        strip_prefixes=True,
    )


def test_split_into_words_separates_quotes_and_drops_single_letters() -> None:
    assert split_into_words('נכנסים ל"הנוכחות שלי"') == ["נכנסים", "הנוכחות", "שלי"]


def test_prefix_removed_when_remaining_word_exists() -> None:
    assert create_tokenizer().tokenize("הבקשות") == ["הבקשות", "בקשות"]


def test_prefix_kept_when_remaining_word_does_not_exist() -> None:
    # "חלה" לא מופיעה במסמכים, ולכן "מחלה" נשארת כמו שהיא
    assert create_tokenizer().tokenize("מחלה") == ["מחלה"]


def test_prefix_kept_when_remaining_word_too_short() -> None:
    assert create_tokenizer().tokenize("שכר") == ["שכר"]


def test_exception_list_prevents_wrong_stripping() -> None:
    assert create_tokenizer().tokenize("בקשה") == ["בקשה", "קשה"]
    assert create_tokenizer({"בקשה"}).tokenize("בקשה") == ["בקשה"]


def test_stripping_disabled_returns_original_words_only() -> None:
    tokenizer = HebrewPrefixTokenizer(CORPUS_VOCABULARY, set(), strip_prefixes=False)
    assert tokenizer.tokenize("הבקשות") == ["הבקשות"]
