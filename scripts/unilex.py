#!/usr/bin/env python3
"""Parser and pronunciation selector for Unilex ``.out`` (MNCL) lexicons.

A lexicon file has one header line (``MNCL``) followed by entries shaped like::

    ("word" POS (((syl phones) stress) ((syl phones) stress) ...))

where ``POS`` is a bare symbol (``nn``), a nested list (``(dt full)``), or a
pipe-joined set (``nnp|vbz``). Each syllable is ``((phone ...) stress)``.

Phones are passed through verbatim -- this module never interprets them, so it
works for any accent's Unilex file (edi, gam, rpx, ... all 14).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Union

logger = logging.getLogger(__name__)

# Vowel phones vary by accent; this small set is only used for the optional
# function-word full/reduced heuristic and can be overridden by the caller.
_DEFAULT_VOWEL_STARTS = ("a", "e", "i", "o", "u", "@", "ai", "ei", "ou", "au", "oo", "ii", "uu")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Syllable:
    """One syllable: its phones and its stress marker (as a string)."""

    phones: tuple[str, ...]
    stress: str


@dataclass(frozen=True)
class Entry:
    """One lexicon entry for a word: a POS label plus a syllabified pronunciation."""

    pos: str
    reduction: str | None  # 'full' / 'reduced' when present, else None
    syllables: tuple[Syllable, ...]


@dataclass
class Lexicon:
    """A parsed Unilex lexicon: word -> list of entries."""

    entries: dict[str, list[Entry]] = field(default_factory=dict)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self.entries

    def get(self, word: str) -> list[Entry]:
        return self.entries.get(word.lower(), [])

    def __len__(self) -> int:
        return len(self.entries)


# --------------------------------------------------------------------------- #
# S-expression reader (minimal, tailored to the MNCL grammar)
# --------------------------------------------------------------------------- #
SExpr = Union[str, list["SExpr"]]  # noqa: UP007


def _tokenize(line: str) -> list[str]:
    tokens: list[str] = []
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if ch.isspace():
            i += 1
        elif ch in "()":
            tokens.append(ch)
            i += 1
        elif ch == '"':
            j = i + 1
            while j < n and line[j] != '"':
                j += 1
            tokens.append(line[i : j + 1])  # keep quotes to mark a string atom
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace() and line[j] not in "()":
                j += 1
            tokens.append(line[i:j])
            i = j
    return tokens


def _read(tokens: Sequence[str], pos: int) -> tuple[SExpr, int]:
    tok = tokens[pos]
    if tok == "(":
        items: list[SExpr] = []
        pos += 1
        while tokens[pos] != ")":
            item, pos = _read(tokens, pos)
            items.append(item)
        return items, pos + 1  # consume ')'
    return tok, pos + 1


def _parse_line(line: str) -> tuple[str, Entry] | None:
    """Parse one entry line into ``(word, Entry)`` or ``None`` if not an entry."""
    line = line.strip()
    if not line.startswith("("):
        return None  # header or blank
    tree, _ = _read(_tokenize(line), 0)
    if not isinstance(tree, list) or len(tree) < 3:
        return None

    word_atom, pos_node, pron_node = tree[0], tree[1], tree[2]
    if not (isinstance(word_atom, str) and word_atom.startswith('"')):
        return None
    word = word_atom.strip('"').lower()

    pos, reduction = _read_pos(pos_node)

    syllables: list[Syllable] = []
    if isinstance(pron_node, list):
        for syl in pron_node:
            # syl == [ [phone, phone, ...], stress ]
            if not (isinstance(syl, list) and len(syl) == 2):
                continue
            phones_node, stress_node = syl
            if isinstance(phones_node, list):
                phones = tuple(p for p in phones_node if isinstance(p, str))
            elif isinstance(phones_node, str):
                phones = (phones_node,)
            else:
                phones = ()
            stress = stress_node if isinstance(stress_node, str) else ""
            syllables.append(Syllable(phones=phones, stress=stress))

    if not syllables:
        return None
    return word, Entry(pos=pos, reduction=reduction, syllables=tuple(syllables))


def _read_pos(node: SExpr) -> tuple[str, str | None]:
    """Return ``(pos, reduction)`` from a bare/nested/pipe POS node."""
    if isinstance(node, str):
        return node, None
    if isinstance(node, list) and node:
        head = node[0] if isinstance(node[0], str) else str(node[0])
        reduction = None
        if len(node) > 1 and isinstance(node[1], str) and node[1] in ("full", "reduced"):
            reduction = node[1]
        return head, reduction
    return str(node), None


def load_lexicon(path: str | Path) -> Lexicon:
    """Load a Unilex ``.out`` file into a :class:`Lexicon`."""
    path = Path(path)
    lex = Lexicon()
    n_lines = n_entries = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            n_lines += 1
            parsed = _parse_line(line)
            if parsed is None:
                continue
            word, entry = parsed
            lex.entries.setdefault(word, []).append(entry)
            n_entries += 1
    logger.info("loaded %s: %d words, %d entries from %d lines",
                path.name, len(lex), n_entries, n_lines)
    return lex


# --------------------------------------------------------------------------- #
# Pronunciation selection (this is where homographs are resolved)
# --------------------------------------------------------------------------- #
class PronunciationSelector(Protocol):
    """Chooses one entry for a word given optional sentence context."""

    def select(
        self, word: str, entries: list[Entry], next_word: str | None, pos_tag: str | None
    ) -> Entry: ...


class FirstEntrySelector:
    """Deterministic default: take the first entry.

    Applies the standard function-word rule for words that carry full/reduced
    variants (e.g. 'the', 'a', 'to', 'of'): choose 'full' before a vowel-initial
    next word, otherwise 'reduced'. This has no external dependencies.
    """

    def __init__(self, vowel_starts: Sequence[str] = _DEFAULT_VOWEL_STARTS) -> None:
        self._vowels = tuple(vowel_starts)

    def select(
        self, word: str, entries: list[Entry], next_word: str | None, pos_tag: str | None
    ) -> Entry:
        reductions = {e.reduction for e in entries if e.reduction}
        if {"full", "reduced"} <= reductions:
            want = "full" if self._starts_with_vowel(next_word, entries) else "reduced"
            for e in entries:
                if e.reduction == want:
                    return e
        return entries[0]

    def _starts_with_vowel(self, next_word: str | None, entries: list[Entry]) -> bool:
        # We don't have the next word's phones here without a second lookup, so
        # fall back to its orthography, which is correct the large majority of
        # the time for the closed function-word set.
        if not next_word:
            return False
        return next_word[0].lower() in ("a", "e", "i", "o", "u")


class PosAwareSelector:
    """Resolve homographs (read, record, ...) using a POS tag when available.

    Narrows to the POS-matching entries, then still applies the function-word
    full/reduced rule among them (so 'the' before a vowel vs consonant is kept),
    falling back to first-entry behaviour when nothing matches.
    """

    def __init__(self, fallback: PronunciationSelector | None = None) -> None:
        self._fallback = fallback or FirstEntrySelector()

    def select(
        self, word: str, entries: list[Entry], next_word: str | None, pos_tag: str | None
    ) -> Entry:
        candidates = entries
        if pos_tag:
            want = pos_tag.lower()
            matched = [e for e in entries if e.pos == want]
            if not matched:  # coarse match on the nn*/vb* family
                matched = [e for e in entries if e.pos[:2] == want[:2]]
            if matched:
                candidates = matched
        # Apply the full/reduced vowel rule among the surviving candidates.
        return self._fallback.select(word, candidates, next_word, pos_tag)


def entry_to_tokens(entry: Entry) -> str:
    """Render one entry as the training token string for a single word.

    e.g. Syllables [(m i),1] [(s t @r r),0] -> ``"1 m i - 0 s t @r r"``.
    """
    syl_strings = [
        " ".join((syl.stress, *syl.phones)).strip()
        for syl in entry.syllables
    ]
    return " - ".join(syl_strings)


if __name__ == "__main__":  # simple manual check
    import sys

    logging.basicConfig(level=logging.INFO)
    lex = load_lexicon(sys.argv[1])
    for w in ("mister", "quilter", "the", "read", "record"):
        print(w, "->", [entry_to_tokens(e) for e in lex.get(w)])
