#!/usr/bin/env python3
"""Parser for the raw Unilex *master* lexicon (the ``headword:variant:POS:...`` file).

This is a different, richer format than the per-accent ``.out`` files:

    headword:variant:POS: { pronunciation } :{morphology}:frequency

e.g.::

    read:1:VB/NN/NNP/VBP: { r * ii d } :{read}:94567
    read:2:VBN/VBD: { r * e d } :{read}:94567
    the:1,unstressed-and-before-consonant:DT: { dh @ } :{the}:16006650

Why it matters: homographs are already separate POS-tagged variants, and the
function-word context (before-vowel / before-consonant / stressed) is stated in
the variant field -- so the two things ``prepare_data`` was approximating are
explicit here. It also carries frequency counts.

Caveat: the phones are accent-neutral Unilex keysymbols (``ii``, ``ou``, ``OR``,
``E5`` ...), not the realised edi/gam/rpx phones in the ``.out`` files. Training
on this yields an accent-neutral base frontend; producing the 14 realised
accents needs Unilex's accent-transform step, which is not in this file.

The parser returns the same :class:`~unilex.Lexicon` / :class:`~unilex.Entry`
types as the ``.out`` parser, so it is a drop-in source for ``prepare_data.py``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from unilex import Entry, Lexicon, Syllable

logger = logging.getLogger(__name__)

# Markers that delimit morpheme regions but are not phones; dropped for phones,
# except '.' (syllable boundary), '*'/'~' (stress) and '$' (syllabic marker).
_REGION_CHARS = str.maketrans({c: " " for c in "{}<>="})
_STRESS_PRIMARY = "*"
_STRESS_SECONDARY = "~"
_SYLLABIC = "$"


def _parse_pron(pron: str) -> list[Syllable]:
    """Parse the ``{ ... }`` pronunciation body into stress-marked syllables."""
    cleaned = pron.translate(_REGION_CHARS)
    syllables: list[Syllable] = []
    for chunk in cleaned.split("."):
        tokens = chunk.split()
        if not tokens:
            continue
        stress = "0"
        phones: list[str] = []
        for tok in tokens:
            if tok == _STRESS_PRIMARY:
                stress = "1"
            elif tok == _STRESS_SECONDARY:
                stress = "2"
            elif tok == _SYLLABIC:
                if phones:  # attach syllabic marker to the previous phone
                    phones[-1] = phones[-1] + _SYLLABIC
            else:
                # keysymbol; take the first of an "ii/@" alternation for determinism
                phone = tok.split("/")[0]
                phone = phone.strip("[]")  # [E50] -> E50
                if phone:
                    phones.append(phone)
        if phones:
            syllables.append(Syllable(phones=tuple(phones), stress=stress))
    return syllables


def _split_pos(pos_field: str) -> list[str]:
    """Expand a ``VB/NN/NNP|POS`` POS field into individual lowercase tags."""
    tags: list[str] = []
    for alt in re.split(r"[/|]", pos_field):
        alt = alt.strip()
        if alt:
            tags.append(alt.lower())
    return tags or ["x"]


def _reduction_from_variant(variant: str) -> str | None:
    """Map the variant description onto full/reduced for the function-word rule."""
    v = variant.lower()
    if "before-vowel" in v or "stressed" in v and "unstressed" not in v:
        return "full"
    if "before-consonant" in v or "unstressed" in v:
        return "reduced"
    return None


def parse_master_line(line: str) -> tuple[str, list[Entry], int] | None:
    """Parse one master line into ``(headword, entries, frequency)``.

    One line becomes several entries -- one per POS tag -- so that a POS-aware
    selector can resolve homographs. The format is six colon-separated fields:
    ``head:variant:POS: { pron } :{morph}:freq``.
    """
    line = line.rstrip("\r\n")
    if not line or ":" not in line:
        return None
    fields = line.split(":")
    if len(fields) != 6:
        return None
    head, variant, pos_field, pron_field, _morph, freq_field = fields

    head = head.strip().lower()
    variant = variant.strip()
    # Field 3 is the full pronunciation; braces are internal morpheme markers,
    # so parse the whole field (they are stripped in _parse_pron), not just the
    # interior of the outermost brace pair.
    pron = pron_field

    freq = 0
    freq_match = re.search(r"(\d+)", freq_field)
    if freq_match:
        freq = int(freq_match.group(1))

    syllables = tuple(_parse_pron(pron))
    if not syllables:
        return None
    reduction = _reduction_from_variant(variant)
    entries = [
        Entry(pos=tag, reduction=reduction, syllables=syllables)
        for tag in _split_pos(pos_field.strip())
    ]
    return head, entries, freq


def load_master_lexicon(path: str | Path) -> tuple[Lexicon, dict[str, int]]:
    """Load the master lexicon into a :class:`Lexicon` plus a frequency map.

    Returns ``(lexicon, freq_by_word)`` where ``freq_by_word`` gives the corpus
    frequency of each headword (useful for coverage-prioritised corpora).
    """
    path = Path(path)
    lex = Lexicon()
    freqs: dict[str, int] = {}
    n_lines = n_ok = n_bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            n_lines += 1
            parsed = parse_master_line(line)
            if parsed is None:
                if line.strip():
                    n_bad += 1
                continue
            head, entries, freq = parsed
            lex.entries.setdefault(head, []).extend(entries)
            freqs[head] = max(freqs.get(head, 0), freq)
            n_ok += 1
    logger.info("loaded master %s: %d words, %d entry-groups (%d unparsed) from %d lines",
                path.name, len(lex), n_ok, n_bad, n_lines)
    return lex, freqs


if __name__ == "__main__":
    import sys

    from unilex import entry_to_tokens

    logging.basicConfig(level=logging.INFO)
    lex, freqs = load_master_lexicon(sys.argv[1])
    for w in ("read", "record", "the", "photograph", "running", "international"):
        entries = lex.get(w)
        rendered = sorted({f"{e.pos}:{entry_to_tokens(e)}" for e in entries})
        print(f"{w} (freq {freqs.get(w)}): {rendered}")
