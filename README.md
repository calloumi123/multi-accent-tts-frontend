# TTS frontend pipeline

> ## ⚠️ KEEP THIS REPO PRIVATE
> `unilex` is a licensed CSTR / University of Edinburgh lexicon and is **not
> freely redistributable**. Publishing it would be a licence violation, and the
> copy here may be employer-licensed. Private repo only.

Grapheme→phoneme TTS frontend: Tacotron-style seq2seq with GMM attention,
~10M parameters.

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
