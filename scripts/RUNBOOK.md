# Data prep + training runbook

End-to-end path from raw text to a trained, packaged model. The scripts in this
folder are designed to sit in the **research repo root** (next to `train.py`,
`datasets.py`, and `notebooks/lex/*.out`).

```
raw text  ──prepare_data.py──▶  dataset/ + train_config.json + vocab/ + coverage report
                                        │
                                   train.sh (train.py)
                                        │
                                  runs/<experiment>/  (checkpoints + vocab)
                                        │
                                 export_bundle.py
                                        │
                                   bundle/  ──▶  Frontend.from_pretrained(...)  /  HF Hub
```

## 1. Prepare data

You can feed either format of lexicon; the loader auto-detects them:

- **Per-accent `.out` files** (`notebooks/lex/unilex-{edi,gam,rpx}.out`) —
  already realised into that accent's phones. Use these for the edi/gam/rpx
  multi-accent model.
- **The Unilex master file** (a single `headword:variant:POS:{pron}:...` file) —
  accent-neutral keysymbols, but with homograph POS variants, the
  function-word context rule, and frequency counts all explicit. Use this for an
  accent-neutral base frontend and for guaranteed coverage (below).

Point the script at a raw text corpus and one lexicon per accent:

```bash
python scripts/prepare_data.py \
  --input corpus.txt \
  --lexicon edi:notebooks/lex/unilex-edi.out \
            gam:notebooks/lex/unilex-gam.out \
            rpx:notebooks/lex/unilex-rpx.out \
  --output-dir dataset --corpus-name libri960 \
  --selector pos \
  --val-frac 0.05 --test-frac 0.05 --seed 1234 \
  --report dataset/coverage.json
```

To train an accent-neutral base model straight from the master file:

```bash
python scripts/prepare_data.py \
  --input corpus.txt --lexicon base:unilex \
  --output-dir dataset --corpus-name unilex_master --selector pos \
  --val-frac 0.05 --test-frac 0.05
```

### Guaranteeing coverage of the whole lexicon

The master file lets you emit a corpus that covers every headword — the email's
"max out coverage" goal:

```bash
python scripts/coverage_corpus.py --lexicon unilex --out corpus.txt
# then run prepare_data.py on corpus.txt
```

The default `--mode sample` places each target word among frequency-sampled
real words, so the filler distribution matches the lexicon's natural unigram
frequencies instead of a handful of repeated carrier words. `--mode words`
emits one word per line; `--mode carrier` uses fixed carrier phrases but
over-represents their words, so it is opt-in only. Use `--min-freq` / `--sort
freq` to focus on the common vocabulary first.

Note: coverage now spans ~100% of headwords (all but a handful with digits or the
"=" symbol); hyphenated entries are split into their component words.
Possessives and quote-wrapped tokens (`dogs'`, `'word'`) are resolved by
`prepare_data` rather than dropped.

What prepare_data does: rejects lines needing upstream normalisation (digits), tokenises,
marks phrase breaks (`_B`) from source punctuation, looks every word up in every
accent (dropping a line if any word is OOV in any accent, so accents stay
parallel), selects one pronunciation per word, and writes the parallel
`src-*.txt` / `tgt-*.txt` splits, the `vocab/` files, a **coverage report**, and
a ready-to-run `train_config.json`.

Read `dataset/coverage.json` to see coverage and the top OOV words per accent —
this is your lever for the "max out coverage" goal: feed a corpus that exercises
every lexicon entry and drive OOV toward zero.

### Corpus collection tips
- **Coverage over volume.** To cover the whole lexicon, a word-list or
  sentence set that hits every headword beats more of the same common words.
  Generating one sentence per lexicon entry (e.g. simple carrier phrases) is a
  legitimate way to guarantee 100% coverage.
- **Normalise first.** Digits, currency, and abbreviations are dropped by
  default (`--allow-digits` to keep). Expand them upstream; the model learns
  orthography→pronunciation, not text normalisation.
- **Punctuation drives prosody.** Keep the original punctuation in the raw
  text — that is what becomes `_B` phrase breaks. Stripped, clean text yields a
  model with no phrase-break structure.

## 2. Train

Install the training deps and run from the research repo root:

```bash
pip install torch numpy tqdm tensorboard ipython
./scripts/train.sh dataset/train_config.json runs
```

Resume from a checkpoint:

```bash
./scripts/train.sh dataset/train_config.json runs runs/<experiment>/step_XXXX.pth.tar
```

**Resources.** This is a Tacotron-style seq2seq with GMM attention; the repo's
recipe is ~300 epochs at batch 32 over libri960-scale data. Plan for a single
modern GPU (16–24 GB) running for hours-to-days depending on corpus size. It
does not train usefully on CPU. Watch attention alignment in TensorBoard — a
clean monotonic diagonal is the sign it is learning; if attention never aligns,
lower the learning rate or the reduction factor `r`.

## 3. Export + use

Turn a finished run into a bundle the pip package loads:

```bash
python scripts/export_bundle.py \
  --run-dir runs/frontend_edi_gam_rpx-<timestamp>-<hash> \
  --config dataset/train_config.json \
  --checkpoint best.pth.tar \
  --out bundle
```

Then, with `pip install multi-accent-frontend`:

```python
from multi_accent_frontend import Frontend
fe = Frontend.from_pretrained("bundle")
fe.transcribe("hello world", accent="gam")
```

Or push to the Hub for `from_pretrained("<org>/<repo>")`:

```bash
huggingface-cli upload <your-org>/multi-accent-frontend bundle .
```

## Known approximations (matter if you compare to the original checkpoints)

The original private training data was generated with pronunciation-selection
rules this pipeline approximates. A model trained here is a valid frontend, but
it will **not** be byte-identical to the authors' data:

- **Homographs** (`read`, `record`, ...): `--selector pos` uses nltk POS tags to
  pick the right entry; `--selector first` ignores POS. Neither is a full
  homograph disambiguator — that is one of the open research directions.
- **Function words** (`the`, `a`, `to`, `of`): a simple vowel-context rule picks
  full vs reduced. The authors' rule may differ.
- **Phrase breaks**: inferred from source punctuation only. There is no learned
  break predictor here — another open research direction.

All three are isolated in `unilex.py` (`PronunciationSelector`) and
`prepare_data.py` (break logic), so you can swap in better strategies without
touching the rest of the pipeline.

## The master lexicon and accents

`unilex_master.py` parses the raw Unilex master file. Two things to know:

- It **resolves homographs and function-word context for you** — those are
  explicit variant fields in the master data (`read:1:VB {r ii d}` vs
  `read:2:VBN {r e d}`; `the:1,before-consonant` vs `the:2,before-vowel`). With
  `--selector pos`, prepare_data uses these directly, so the two big
  approximations above largely go away for master-sourced data.
- Its phones are **accent-neutral keysymbols**, not realised edi/gam/rpx phones.
  A model trained on master-sourced targets is a valid *base* frontend, but it
  is not one of the 14 realised accents. Producing those needs Unilex's
  accent-transform (postlex) step, which is not in this file — that transform is
  what the shipped `.out` files already had applied. If you have the postlex
  rules, apply them to the master pronunciations before `entry_to_tokens`; the
  parser keeps enough structure (syllables + stress) to do so.
