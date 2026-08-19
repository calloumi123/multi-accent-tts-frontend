#!/usr/bin/env python3
"""Turn a raw text corpus + Unilex lexicons into model-ready training data.

This is the data-collection / cleaning step that was missing from the research
repo. Given plain text and one ``.out`` lexicon per accent, it produces the
parallel ``src-*.txt`` / ``tgt-*.txt`` splits, the vocab files, and a coverage
report -- in the exact format the training code expects.

Pipeline per input line:
  1. Reject lines needing upstream normalisation (digits) unless --allow-digits.
  2. Tokenise into words (letters + apostrophes); split on hyphens; mark a
     phrase break (_B) wherever source punctuation (, . ; : ! ? -- ...) falls.
  3. Look every word up in *every* requested accent's lexicon. If any word is
     out-of-vocabulary in any accent, drop the line (default) so all accents
     stay parallel. OOV words are logged for the coverage report.
  4. Select one pronunciation per word (homograph handling; see --selector).
  5. Emit uppercase src and per-accent phone tgt, split into train/val/test,
     then build vocab.

Example
-------
    python prepare_data.py \
        --input corpus.txt \
        --lexicon edi:/lex/unilex-edi.out gam:/lex/unilex-gam.out rpx:/lex/unilex-rpx.out \
        --output-dir dataset --corpus-name libri960 \
        --selector pos --val-frac 0.05 --test-frac 0.05 --seed 1234
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from unilex import (
    Entry,
    FirstEntrySelector,
    Lexicon,
    PosAwareSelector,
    PronunciationSelector,
    entry_to_tokens,
    load_lexicon,
)
from unilex_master import load_master_lexicon

logger = logging.getLogger("prepare_data")


def load_any_lexicon(path: Path) -> Lexicon:
    """Load a lexicon, auto-detecting the ``.out`` (MNCL) vs master format."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            s = line.strip()
            if not s:
                continue
            if s.startswith("(") or s == "MNCL":
                return load_lexicon(path)
            if s.count(":") >= 4 and "{" in s:
                lex, _freqs = load_master_lexicon(path)
                return lex
            break
    return load_lexicon(path)

WORD_RE = re.compile(r"[A-Za-z']+")
BREAK_PUNCT = set(",.;:!?()[]{}\u2014\u2013\u2026")  # , . ; : ! ? ( ) — – …
WORD_SEP = " + "
BREAK = "_B"


@dataclass
class Example:
    """One cleaned training example, shared across accents."""

    words: list[str]          # lowercase words, hyphen-split, punctuation-free
    break_after: list[bool]   # phrase break after word i (last is always True)
    pos_tags: list[str] | None = None


@dataclass
class Stats:
    total_lines: int = 0
    kept: int = 0
    dropped_digits: int = 0
    dropped_empty: int = 0
    dropped_oov: int = 0
    oov_counter: Counter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.oov_counter is None:
            self.oov_counter = Counter()


# --------------------------------------------------------------------------- #
# Cleaning / tokenising
# --------------------------------------------------------------------------- #
def clean_line(line: str, *, allow_digits: bool, stats: Stats) -> Example | None:
    """Tokenise one raw line into an :class:`Example`, or ``None`` if dropped."""
    if any(ch.isdigit() for ch in line) and not allow_digits:
        stats.dropped_digits += 1
        return None

    # Split into "chunks" on whitespace and hyphens; track break punctuation.
    words: list[str] = []
    break_after: list[bool] = []
    pending_break = False

    # Replace hyphens with spaces so hyphenated words become separate tokens
    # (keeps src within the A-Z/space/apostrophe charset used at training time).
    normalised = line.replace("-", " ")
    # Walk char-by-char grouping words, noting punctuation that follows a word.
    idx = 0
    n = len(normalised)
    while idx < n:
        m = WORD_RE.match(normalised, idx)
        if m:
            token = m.group(0)  # keep apostrophes; lexicon has 'cause, a's, rachel's
            if pending_break and words:
                break_after[-1] = True
                pending_break = False
            if token:
                words.append(token.lower())
                break_after.append(False)
            idx = m.end()
        else:
            if normalised[idx] in BREAK_PUNCT:
                pending_break = True
            idx += 1

    if not words:
        stats.dropped_empty += 1
        return None
    break_after[-1] = True  # terminal break
    return Example(words=words, break_after=break_after)


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
def _candidate_forms(word: str) -> list[str]:
    """Lookup forms to try for a surface token, most specific first.

    Keeps real leading-apostrophe entries ('em, 'cause) by trying the token
    verbatim first, then handles plural possessives (dogs') and quote-wrapped
    tokens ('word') by stripping edge apostrophes.
    """
    forms = [word]
    if word.endswith("'") and len(word) > 1:
        forms.append(word[:-1])          # dogs' -> dogs
    stripped = word.strip("'")
    if stripped and stripped not in forms:
        forms.append(stripped)           # 'word' -> word
    return forms


def lookup_entries(lex: Lexicon, word: str) -> list[Entry]:
    """Return entries for the first candidate form present in the lexicon."""
    for form in _candidate_forms(word):
        entries = lex.get(form)
        if entries:
            return entries
    return []


def resolve_example(
    example: Example,
    lexicons: dict[str, Lexicon],
    selector: PronunciationSelector,
    stats: Stats,
) -> dict[str, str] | None:
    """Return ``{accent: tgt_string}`` for an example, or ``None`` if any OOV.

    A word must resolve in *every* accent, so all accents stay parallel.
    """
    # OOV check across all accents. Record every offending word (not just the
    # first) so the coverage report is complete, then drop the line.
    oov_found = False
    for word in example.words:
        for accent, lex in lexicons.items():
            if not lookup_entries(lex, word):
                stats.oov_counter[f"{accent}:{word}"] += 1
                oov_found = True
    if oov_found:
        stats.dropped_oov += 1
        return None

    tgt_by_accent: dict[str, str] = {}
    for accent, lex in lexicons.items():
        word_tokens: list[str] = []
        for i, word in enumerate(example.words):
            entries = lookup_entries(lex, word)
            next_word = example.words[i + 1] if i + 1 < len(example.words) else None
            pos_tag = example.pos_tags[i] if example.pos_tags else None
            entry: Entry = selector.select(word, entries, next_word, pos_tag)
            word_tokens.append(entry_to_tokens(entry))
        tgt_by_accent[accent] = _assemble_tgt(word_tokens, example.break_after)
    return tgt_by_accent


def _assemble_tgt(word_tokens: list[str], break_after: list[bool]) -> str:
    """Join word token-strings with ``+`` inside phrases and ``_B`` at breaks."""
    phrases: list[list[str]] = []
    current: list[str] = []
    for tok, is_break in zip(word_tokens, break_after):
        current.append(tok)
        if is_break:
            phrases.append(current)
            current = []
    if current:
        phrases.append(current)
    phrase_strs = [WORD_SEP.join(p) for p in phrases if p]
    return (" " + BREAK + " ").join(phrase_strs) + " " + BREAK


def example_to_src(example: Example) -> str:
    """Uppercase, space-joined source text (matches the training charset)."""
    return " ".join(w.upper() for w in example.words)


# --------------------------------------------------------------------------- #
# POS tagging (optional, for homograph handling)
# --------------------------------------------------------------------------- #
def get_tagger() -> Callable[[list[str]], list[str]]:
    """Return a function mapping a word list to POS tags, using nltk."""
    try:
        import nltk  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "--selector pos needs nltk. Install with: pip install nltk\n"
            "then: python -m nltk.downloader averaged_perceptron_tagger punkt"
        ) from exc

    def tag(words: list[str]) -> list[str]:
        return [t for _, t in nltk.pos_tag(words)]

    return tag


# --------------------------------------------------------------------------- #
# Streaming writer (splits + vocab in a single pass, no full-corpus buffering)
# --------------------------------------------------------------------------- #
class DatasetWriter:
    """Writes src/tgt splits and accumulates vocab counts line by line.

    Assigns each example to a split by a seeded random draw, so the whole corpus
    never needs to be held in memory (split sizes are approximate, not exact).
    """

    def __init__(
        self,
        accents: Sequence[str],
        output_dir: Path,
        corpus_name: str,
        val_frac: float,
        test_frac: float,
        seed: int,
    ) -> None:
        self._accents = list(accents)
        self._output_dir = output_dir
        self._val_frac = val_frac
        self._test_frac = test_frac
        self._rng = random.Random(seed)
        self.counts = {"train": 0, "val": 0, "test": 0}
        self._char_counter: Counter = Counter()
        self._phone_counter: Counter = Counter()
        self._word_counter: Counter = Counter()

        self._handles: dict[tuple[str, str, str], object] = {}
        for accent in self._accents:
            data_dir = output_dir / f"unilex_{accent}" / "charlex" / corpus_name / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            for split in ("train", "val", "test"):
                for kind in ("src", "tgt"):
                    self._handles[(accent, kind, split)] = open(  # noqa: SIM115
                        data_dir / f"{kind}-{split}.txt", "w", encoding="utf-8"
                    )

    def _pick_split(self) -> str:
        r = self._rng.random()
        if r < self._test_frac:
            return "test"
        if r < self._test_frac + self._val_frac:
            return "val"
        return "train"

    def add(self, src_line: str, tgt_by_accent: dict[str, str]) -> None:
        split = self._pick_split()
        self.counts[split] += 1
        self._char_counter.update(src_line.replace(" ", "#"))
        self._word_counter.update(src_line.split())
        for accent in self._accents:
            self._handles[(accent, "src", split)].write(src_line + "\n")  # type: ignore[attr-defined]
            tgt = tgt_by_accent[accent]
            self._handles[(accent, "tgt", split)].write(tgt + "\n")  # type: ignore[attr-defined]
            self._phone_counter.update(tgt.split())

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()  # type: ignore[attr-defined]

    def write_vocab(self) -> dict[str, int]:
        vocab_dir = self._output_dir / "vocab"
        vocab_dir.mkdir(parents=True, exist_ok=True)

        def dump(counter: Counter, name: str) -> int:
            items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
            with (vocab_dir / name).open("w", encoding="utf-8") as f:
                for tok, cnt in items:
                    f.write(f"{tok}\t{cnt}\n")
            return len(items)

        return {
            "src": dump(self._char_counter, "src.vocab"),
            "tgt": dump(self._phone_counter, "tgt.vocab"),
            "word": dump(self._word_counter, "src.word.vocab"),
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_lexicon_args(specs: Sequence[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"--lexicon expects accent:path, got {spec!r}")
        accent, path = spec.split(":", 1)
        out[accent] = Path(path)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Raw text file, one sentence/paragraph per line.")
    p.add_argument(
        "--lexicon", required=True, nargs="+", metavar="ACCENT:PATH",
        help="One or more accent:lexicon.out pairs, e.g. edi:/lex/unilex-edi.out",
    )
    p.add_argument("--output-dir", required=True, help="Root directory for generated data.")
    p.add_argument("--corpus-name", default="corpus", help="Corpus subdir name (default: corpus).")
    p.add_argument("--selector", choices=["first", "pos"], default="first",
                   help="Pronunciation selection. 'pos' resolves homographs via nltk tags.")
    p.add_argument("--oov", choices=["drop", "error"], default="drop",
                   help="What to do when a word is missing from a lexicon.")
    p.add_argument("--allow-digits", action="store_true",
                   help="Keep lines containing digits (default: drop; normalise upstream).")
    p.add_argument("--min-words", type=int, default=1, help="Drop examples shorter than this.")
    p.add_argument("--max-words", type=int, default=0, help="Drop examples longer than this (0 = no limit).")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--report", default=None, help="Optional path to write a JSON coverage report.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def write_train_config(
    accents: Sequence[str],
    output_dir: Path,
    corpus_name: str,
    have_val: bool,
    have_test: bool,
) -> Path:
    """Emit a ready-to-run train.py config pointing at the generated data.

    Data paths are absolute so training works from any working directory.
    Model hyper-parameters mirror the repo's multi-accent r=1 h512 recipe.
    """
    def data_path(accent: str, kind: str, split: str) -> str:
        return str(
            (output_dir / f"unilex_{accent}" / "charlex" / corpus_name / "data" / f"{kind}-{split}.txt").resolve()
        )

    data_block: dict[str, dict[str, dict[str, str]]] = {}
    for accent in accents:
        block = {"corpus_1": {"path_src": data_path(accent, "src", "train"),
                              "path_tgt": data_path(accent, "tgt", "train")}}
        if have_val:
            block["valid"] = {"path_src": data_path(accent, "src", "val"),
                              "path_tgt": data_path(accent, "tgt", "val")}
        if have_test:
            block["test"] = {"path_src": data_path(accent, "src", "test"),
                             "path_tgt": data_path(accent, "tgt", "test")}
        data_block[accent] = block

    config = {
        "experiment_name": "frontend_" + "_".join(accents),
        "run_description": "auto-generated by prepare_data.py",
        "epochs": 300,
        "batch_size": 32,
        "save_step": 10000,
        "batch_group_size": 100,
        "max_seq_len": 600,
        "learning_rate": 0.00005,
        "r": 1,
        "has_postnet": False,
        "has_stopnet": False,
        "has_prenet": False,
        "enc_embedding_dim": 512,
        "dec_embedding_dim": 512,
        "enc_hidden_dim": 512,
        "dec_hidden_dim": 512,
        "post_hidden_dim": 512,
        "attn_type": "gmm",
        "num_mixtures": 5,
        "gmm_version": "v2",
        "data": data_block,
        "src_vocab": "vocab/src.vocab",
        "tgt_vocab": "vocab/tgt.vocab",
        "word_vocab": "vocab/src.word.vocab",
    }
    path = output_dir / "train_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    lex_paths = parse_lexicon_args(args.lexicon)
    accents = list(lex_paths.keys())
    logger.info("loading %d lexicon(s): %s", len(accents), ", ".join(accents))
    lexicons = {a: load_any_lexicon(p) for a, p in lex_paths.items()}

    selector: PronunciationSelector = (
        PosAwareSelector() if args.selector == "pos" else FirstEntrySelector()
    )

    stats = Stats()
    tagger = get_tagger() if args.selector == "pos" else None
    output_dir = Path(args.output_dir)
    writer = DatasetWriter(
        accents, output_dir, args.corpus_name,
        args.val_frac, args.test_frac, args.seed,
    )

    try:
        with open(args.input, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stats.total_lines += 1
                line = line.strip()
                if not line:
                    stats.dropped_empty += 1
                    continue
                ex = clean_line(line, allow_digits=args.allow_digits, stats=stats)
                if ex is None:
                    continue
                if len(ex.words) < args.min_words:
                    stats.dropped_empty += 1
                    continue
                if args.max_words and len(ex.words) > args.max_words:
                    stats.dropped_empty += 1
                    continue
                if tagger is not None:
                    ex.pos_tags = tagger(ex.words)
                resolved = resolve_example(ex, lexicons, selector, stats)
                if resolved is None:
                    if args.oov == "error":
                        raise SystemExit("OOV word encountered and --oov error is set.")
                    continue
                stats.kept += 1
                writer.add(example_to_src(ex), resolved)
    finally:
        writer.close()

    if stats.kept == 0:
        logger.error("no usable examples produced; check input and lexicons")
        return 1

    counts = writer.counts
    vocab_sizes = writer.write_vocab()
    config_path = write_train_config(
        accents, output_dir, args.corpus_name,
        have_val=args.val_frac > 0, have_test=args.test_frac > 0,
    )
    logger.info("wrote train config to %s (run: python train.py --config_path %s --output_path ./runs)",
                config_path, config_path)

    # Coverage report
    top_oov = stats.oov_counter.most_common(25)
    coverage_rate = round(stats.kept / max(stats.total_lines, 1), 4)
    report = {
        "input_lines": stats.total_lines,
        "kept": stats.kept,
        "dropped": {
            "needs_normalisation_digits": stats.dropped_digits,
            "empty_or_length_filtered": stats.dropped_empty,
            "out_of_vocabulary": stats.dropped_oov,
        },
        "coverage_rate": coverage_rate,
        "splits": counts,
        "vocab_sizes": vocab_sizes,
        "accents": accents,
        "top_oov": top_oov,
    }
    logger.info("kept %d/%d lines (%.1f%%) | splits=%s | vocab src=%d tgt=%d",
                stats.kept, stats.total_lines, 100 * coverage_rate,
                counts, vocab_sizes["src"], vocab_sizes["tgt"])
    if top_oov:
        logger.info("top OOV (accent:word): %s", ", ".join(f"{w}({c})" for w, c in top_oov[:10]))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("wrote coverage report to %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
