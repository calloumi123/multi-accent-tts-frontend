#!/usr/bin/env python3
"""Fetch and clean a raw text corpus into one-sentence-per-line, ready for prepare_data.

Handles the quirks of each common source so the output drops straight into
``prepare_data.py``. Downloads (``--url``) run on your machine; you can also point
``--input`` at a file you already have.

Sources
-------
  leipzig           Leipzig Corpora Collection archive or *-sentences.txt.
                    Strips the leading "<id>\\t" column (otherwise prepare_data's
                    digit filter would drop every line).
  librispeech-norm  LibriSpeech normalised LM text (.txt/.txt.gz), already one
                    sentence per line (uppercase, NO punctuation -> no breaks).
  books             Raw prose (LibriSpeech book corpus, Gutenberg, ...); splits
                    paragraphs into sentences.
  plain             A plain text file, one sentence per line already.
  tsv               Generic tab-separated file; keeps the last column.
  hf                A Hugging Face dataset, streamed (needs `pip install datasets`).

Examples
--------
    # Leipzig 1M news (download, strip IDs)
    python fetch_corpus.py --source leipzig \
        --url https://downloads.wortschatz-leipzig.de/corpora/eng_news_2020_1M.tar.gz \
        --out corpus.txt

    # LibriSpeech normalised LM text (first 2M lines)
    python fetch_corpus.py --source librispeech-norm \
        --url https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz \
        --out corpus.txt --limit 2000000

    # PG19 (Project Gutenberg) via Hugging Face, sentence-split
    python fetch_corpus.py --source hf --hf-dataset pg19 --hf-split train \
        --hf-text-field text --split-sentences --limit 500000 --out corpus.txt
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import re
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import IO

logger = logging.getLogger("fetch_corpus")

# Split on sentence-final punctuation followed by whitespace + a capital/quote.
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def _open_source(url: str | None, input_path: str | None) -> Path:
    """Return a local path for the source, downloading the URL if needed."""
    if input_path:
        return Path(input_path)
    if not url:
        raise SystemExit("provide --input or --url")
    suffix = "".join(Path(url).suffixes) or ".dat"
    tmp = Path(tempfile.mkdtemp()) / ("download" + suffix)
    logger.info("downloading %s -> %s", url, tmp)
    urllib.request.urlretrieve(url, tmp)
    return tmp


def _open_text(path: Path) -> IO[str]:
    """Open a plain or gzipped text file as UTF-8 text."""
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Per-source line iterators (each yields raw sentence strings)
# --------------------------------------------------------------------------- #
def _iter_leipzig(path: Path) -> Iterator[str]:
    """Yield sentences from a Leipzig archive or *-sentences.txt, ID column stripped."""
    if path.suffix in (".gz", ".tgz") and tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as tar:
            member = next((m for m in tar.getmembers() if m.name.endswith("-sentences.txt")), None)
            if member is None:
                raise SystemExit("no *-sentences.txt found in the Leipzig archive")
            fobj = tar.extractfile(member)
            if fobj is None:
                raise SystemExit("could not read the sentences file from the archive")
            stream = io.TextIOWrapper(fobj, encoding="utf-8", errors="replace")
            yield from _strip_id_column(stream)
    else:
        with _open_text(path) as handle:
            yield from _strip_id_column(handle)


def _strip_id_column(lines: Iterable[str]) -> Iterator[str]:
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        # Leipzig format is "<id>\t<sentence>"; keep everything after the first tab.
        yield line.split("\t", 1)[1] if "\t" in line else line


def _iter_plain(path: Path, split_sentences: bool) -> Iterator[str]:
    with _open_text(path) as handle:
        if split_sentences:
            yield from _split_stream(handle)
        else:
            for line in handle:
                line = line.strip()
                if line:
                    yield line


def _iter_tsv(path: Path) -> Iterator[str]:
    with _open_text(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line:
                yield line.split("\t")[-1]


def _iter_hf(dataset: str, config: str | None, split: str, text_field: str,
             split_sentences: bool) -> Iterator[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("--source hf needs the datasets library: pip install datasets") from exc
    ds = load_dataset(dataset, config, split=split, streaming=True)
    for row in ds:
        text = row.get(text_field, "")
        if not text:
            continue
        if split_sentences:
            yield from _split_text(text)
        else:
            for line in str(text).splitlines():
                line = line.strip()
                if line:
                    yield line


def _split_stream(handle: Iterable[str]) -> Iterator[str]:
    """Sentence-split a stream of prose lines, joining wrapped lines into paragraphs."""
    buffer: list[str] = []
    for line in handle:
        line = line.strip()
        if not line:
            if buffer:
                yield from _split_text(" ".join(buffer))
                buffer = []
        else:
            buffer.append(line)
    if buffer:
        yield from _split_text(" ".join(buffer))


def _split_text(text: str) -> Iterator[str]:
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if sentence:
            yield sentence


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def iter_sentences(args: argparse.Namespace) -> Iterator[str]:
    if args.source == "hf":
        yield from _iter_hf(args.hf_dataset, args.hf_config, args.hf_split,
                            args.hf_text_field, args.split_sentences)
        return
    path = _open_source(args.url, args.input)
    if args.source == "leipzig":
        yield from _iter_leipzig(path)
    elif args.source == "librispeech-norm":
        yield from _iter_plain(path, split_sentences=False)
    elif args.source == "books":
        yield from _iter_plain(path, split_sentences=True)
    elif args.source == "plain":
        yield from _iter_plain(path, split_sentences=args.split_sentences)
    elif args.source == "tsv":
        yield from _iter_tsv(path)
    else:  # pragma: no cover
        raise SystemExit(f"unknown source: {args.source}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True,
                   choices=["leipzig", "librispeech-norm", "books", "plain", "tsv", "hf"])
    p.add_argument("--url", default=None, help="Download URL for the corpus.")
    p.add_argument("--input", default=None, help="Local path to the corpus (instead of --url).")
    p.add_argument("--out", required=True, help="Output file (one sentence per line).")
    p.add_argument("--limit", type=int, default=0, help="Max sentences to write (0 = all).")
    p.add_argument("--min-chars", type=int, default=1, help="Skip sentences shorter than this.")
    p.add_argument("--split-sentences", action="store_true",
                   help="Split paragraphs into sentences (default for 'books').")
    p.add_argument("--hf-dataset", default=None, help="Hugging Face dataset name (for --source hf).")
    p.add_argument("--hf-config", default=None, help="Hugging Face dataset config.")
    p.add_argument("--hf-split", default="train", help="Hugging Face split (default: train).")
    p.add_argument("--hf-text-field", default="text", help="Text column name (default: text).")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    written = 0
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for sentence in iter_sentences(args):
            sentence = " ".join(sentence.split())  # collapse whitespace
            if len(sentence) < args.min_chars:
                continue
            f.write(sentence + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break

    logger.info("wrote %d sentences to %s", written, out)
    if args.source == "librispeech-norm":
        logger.info("note: this source has no punctuation, so no _B phrase breaks will be produced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
