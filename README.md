# TTS frontend pipeline


Grapheme→phoneme TTS frontend: Tacotron-style seq2seq with GMM attention,
~10M parameters, trained on **three realised accents**:

| Accent | Variety | Example |
|---|---|---|
| `edi` | Edinburgh Scottish (rhotic, tapped r) | `run` → `t^ uh n` |
| `gam` | General American (rhotic) | `bath` → `b a th` |
| `rpx` | RP (non-rhotic) | `bath` → `b aa th`, `for` → `f @` |

All three lexicons share the identical 116,739 headwords, so training on all
three costs no coverage. Each sentence becomes three aligned training items
(one `src`, three `tgt`), and the model learns a 32-dim accent embedding.

## Layout

| Path | What |
|---|---|
| `scripts/` | data pipeline (`prepare_data.py`, `coverage_corpus.py`, `unilex_master.py`, `export_bundle.py`, `fetch_corpus.py`, `unilex.py`) |
| `unilex` | the master lexicon — 116,739 words, 118,374 entry-groups |
| `dist/` | the `multi_accent_frontend` wheel, for testing the exported bundle |
| `train_frontend.ipynb` | the Colab training notebook |

## Running it

Open `train_frontend.ipynb` in Colab. It clones this repo inside the Colab VM
(so no uploads from your own machine) and writes checkpoints out to Google
Drive. Set `GIT_REPO` in the params cell, and add a `GH_TOKEN` secret in Colab's
🔑 Secrets panel with `repo` scope.

The training code itself (`train.py`, `datasets.py`, `layers/`) is cloned
separately from the public research repo; it is not vendored here.

## Do not commit checkpoints

A checkpoint is 115 MB (40 MB of weights plus Adam's two moment tensors), above
GitHub's 100 MB per-file limit. `.gitignore` already excludes them.

## Licensing and provenance

**Code** (`scripts/`, `train_frontend.ipynb`, `dist/`) — MIT, see `LICENSE`.
The copyright notice of the upstream project is retained as MIT requires.

**`unilex`** — the Unisyn/Unilex master lexicon, originating from CSTR,
University of Edinburgh. It is included here under the repository author's
academic-use permission as an Edinburgh student. It is **third-party data and is
not covered by the MIT licence above** — that licence applies to the code only,
and no relicensing of the lexicon is claimed or implied. If you are not covered
by your own Unisyn permission, obtain the lexicon from CSTR directly rather than
from this repository.

**Training code** — `train.py`, `datasets.py` and the model layers are cloned at
runtime from the upstream research repo and are not vendored here.
