#!/usr/bin/env python3
"""Generate a corpus that covers the lexicon, from the Unilex master file.

The email's second goal is "a large text corpus covering every word in our
lexicon". This emits exactly that: every headword (optionally above a frequency
threshold), either one word per line or wrapped in a simple carrier phrase so
the examples have sentence context and a phrase break.

Feed the output straight into prepare_data.py to get guaranteed coverage.

Usage:
    python coverage_corpus.py --lexicon unilex --out corpus.txt --mode carrier
    python coverage_corpus.py --lexicon unilex --out corpus.txt --min-freq 5 --sort freq
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from collections.abc import Sequence
from pathlib import Path

from unilex_master import load_master_lexicon

logger = logging.getLogger("coverage_corpus")

# Fixed carrier templates over-represent their own words, so they are offered
# only as an explicit opt-in. The default 'sample' mode avoids that skew.
CARRIER_TEMPLATES = (
    "we say the word {w} again",
    "the word is {w} today",
    "please repeat {w} once more",
)
# Emit headwords prepare_data can consume: letters with apostrophes (kept) and
# hyphens (prepare_data splits these into their component words). Entries with
# '=', digits, or other symbols are genuinely uncoverable and reported.
EMITTABLE = re.compile(r"^'?[a-z]+(?:['\-][a-z]+)*'?$")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lexicon", required=True, help="Path to the Unilex master file.")
    p.add_argument("--out", required=True, help="Output corpus text file.")
    p.add_argument("--mode", choices=["sample", "words", "carrier"], default="sample",
                   help="'sample' (default): target word among frequency-sampled real words "
                        "(natural distribution, no skew). 'words': one word per line. "
                        "'carrier': fixed carrier phrases (skewed; opt-in only).")
    p.add_argument("--fillers", type=int, default=6,
                   help="Number of filler words per line in 'sample' mode (default 6).")
    p.add_argument("--min-freq", type=int, default=0, help="Skip headwords rarer than this.")
    p.add_argument("--limit", type=int, default=0, help="Cap number of target headwords (0 = all).")
    p.add_argument("--sort", choices=["freq", "alpha"], default="alpha",
                   help="Order of target headwords in the output.")
    p.add_argument("--seed", type=int, default=1234, help="Seed for 'sample' filler selection.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    _lex, freqs = load_master_lexicon(args.lexicon)
    total_headwords = len(freqs)
    words: list[str] = [
        w for w in freqs
        if EMITTABLE.match(w) and freqs[w] >= args.min_freq
    ]
    excluded = total_headwords - sum(1 for w in freqs if EMITTABLE.match(w))
    if args.sort == "freq":
        words.sort(key=lambda w: freqs[w], reverse=True)
    else:
        words.sort()
    if args.limit:
        words = words[: args.limit]

    rng = random.Random(args.seed)
    # Filler pool + frequency weights (shared across lines; sampled with
    # replacement so each word appears as filler in proportion to its frequency,
    # reproducing a natural unigram distribution rather than a fixed carrier).
    filler_pool = words
    filler_weights = [max(freqs[w], 1) for w in filler_pool]

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for i, target in enumerate(words):
            if args.mode == "words":
                f.write(target + "\n")
            elif args.mode == "carrier":
                template = CARRIER_TEMPLATES[i % len(CARRIER_TEMPLATES)]
                f.write(template.format(w=target) + "\n")
            else:  # sample
                if args.fillers > 0 and len(filler_pool) > 1:
                    line_words = rng.choices(filler_pool, weights=filler_weights, k=args.fillers)
                else:
                    line_words = []
                line_words.append(target)  # guarantee the target appears -> coverage
                rng.shuffle(line_words)
                f.write(" ".join(line_words) + "\n")

    emit_frac = 100 * (total_headwords - excluded) / max(total_headwords, 1)
    logger.info("wrote %d lines to %s (mode=%s, min_freq=%d)",
                len(words), out, args.mode, args.min_freq)
    logger.info("emittable headwords: %.1f%% of %d (%d excluded: contain '=', digits, or other symbols)",
                emit_frac, total_headwords, excluded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
