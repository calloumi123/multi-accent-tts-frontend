# Running the pipeline on real datasets

Step-by-step for each corpus. Every path is the same shape:

```
fetch_corpus.py  →  clean sentences  →  prepare_data.py  →  dataset/ + train_config.json
                                                          →  train.sh  →  export_bundle.py  →  Frontend
```

You only need three things installed to prepare data: Python 3.10+, `pip install
numpy`, and (for `--selector pos`) `pip install nltk` then
`python -m nltk.downloader averaged_perceptron_tagger`. Training additionally
needs the research repo and a GPU (see the end).

Throughout, `LEX` is your lexicon. Use the Unilex **master** file you have:

```bash
export LEX=unilex          # accent-neutral keysymbols (all 116k words)
# or a realised accent from the research repo:
# export LEX=notebooks/lex/unilex-edi.out
```

Two rules that apply to every dataset:

- **Keep punctuation.** It becomes the `_B` phrase breaks. Sources without
  punctuation (LibriSpeech-norm) give a model with no break structure.
- **Check the coverage report** (`--report`) before scaling up. Web/news text
  has more OOV (proper nouns) and more digit-lines dropped than book text.

---

## 1. Leipzig Corpora Collection  ⭐ easiest "a million sentences"

Fixed-size files (10K / 100K / 1M sentences), one per line, punctuated.

```bash
# download + strip the leading "<id>\t" column (or every line would be dropped)
python scripts/fetch_corpus.py --source leipzig \
    --url https://downloads.wortschatz-leipzig.de/corpora/eng_news_2020_1M.tar.gz \
    --out corpus.txt

python scripts/prepare_data.py \
    --input corpus.txt --lexicon base:$LEX \
    --output-dir dataset --corpus-name leipzig_news \
    --selector pos --val-frac 0.02 --test-frac 0.02 \
    --report dataset/coverage.json
```

Pick other corpora (web, Wikipedia, other years/sizes) from the download page:
https://wortschatz.uni-leipzig.de/en/download/English . Swap the `--url`.

---

## 2. LibriSpeech LM text  ⭐ same source as the repo's own data

Two flavours from https://www.openslr.org/11/ :

**(a) Normalised LM text** — ~40M sentences, already uppercase, **no
punctuation** (so no phrase breaks). Matches the repo's existing `src` format.

```bash
python scripts/fetch_corpus.py --source librispeech-norm \
    --url https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz \
    --out corpus.txt --limit 2000000
python scripts/prepare_data.py --input corpus.txt --lexicon base:$LEX \
    --output-dir dataset --corpus-name libri_norm --selector first \
    --val-frac 0.02 --test-frac 0.02 --report dataset/coverage.json
```

**(b) Book corpus** — the raw Gutenberg books *with* punctuation. Use this if
you want `_B` breaks:

```bash
python scripts/fetch_corpus.py --source books \
    --url https://www.openslr.org/resources/11/librispeech-lm-corpus.tgz \
    --out corpus.txt --limit 2000000
# (then prepare_data as above)
```

---

## 3. Project Gutenberg / PG19  ⭐ clean, punctuated, read-speech style

Best content match for a TTS frontend. Easiest via Hugging Face streaming:

```bash
pip install datasets
python scripts/fetch_corpus.py --source hf \
    --hf-dataset pg19 --hf-split train --hf-text-field text \
    --split-sentences --limit 500000 --out corpus.txt
python scripts/prepare_data.py --input corpus.txt --lexicon base:$LEX \
    --output-dir dataset --corpus-name pg19 --selector pos \
    --val-frac 0.02 --test-frac 0.02 --report dataset/coverage.json
```

---

## 4. WMT News Crawl  (tens of millions of news sentences/year)

Download a year's `news.YEAR.en.shuffled.deduped.gz` from
https://data.statmt.org/news-crawl/ (already one sentence per line, punctuated):

```bash
python scripts/fetch_corpus.py --source plain \
    --url https://data.statmt.org/news-crawl/en/news.2020.en.shuffled.deduped.gz \
    --out corpus.txt --limit 2000000
# prepare_data as above (news has more OOV proper nouns -- check the report)
```

---

## 5. Any Hugging Face dataset (C4, cc_news, wikitext, …)

Stream without downloading everything; give the dataset name, split and text
column:

```bash
python scripts/fetch_corpus.py --source hf \
    --hf-dataset allenai/c4 --hf-config en --hf-split train \
    --hf-text-field text --split-sentences --limit 1000000 --out corpus.txt
```

---

## 6. Your own file

```bash
# one sentence per line already:
python scripts/fetch_corpus.py --source plain --input my.txt --out corpus.txt
# paragraphs that need splitting:
python scripts/fetch_corpus.py --source plain --input my.txt --split-sentences --out corpus.txt
# tab-separated (keep last column):
python scripts/fetch_corpus.py --source tsv --input my.tsv --out corpus.txt
```

---

## Reading the coverage report

`dataset/coverage.json` tells you what happened:

```json
{ "kept": 1897430, "input_lines": 2000000,
  "dropped": { "needs_normalisation_digits": 71203,
               "empty_or_length_filtered": 1044,
               "out_of_vocabulary": 30323 },
  "coverage_rate": 0.9487,
  "top_oov": [["base:covid", 512], ["base:reranked", 88]] }
```

- High `needs_normalisation_digits` → your text has many numbers; expand them
  upstream, or accept the drops.
- `top_oov` lists the words that cost you sentences. If the tail matters, top up
  coverage for just those words with `coverage_corpus.py` and concatenate.

---

## Train, then export

Training uses the research repo's `train.py` (needs a GPU; see RUNBOOK.md):

```bash
git clone https://github.com/sunsiqitos/multi_accent_s2s_frontend
cp scripts/*.py scripts/*.sh multi_accent_s2s_frontend/    # data scripts alongside train.py
cd multi_accent_s2s_frontend
pip install torch numpy tqdm tensorboard ipython
./train.sh /path/to/dataset/train_config.json runs

python export_bundle.py --run-dir runs/<experiment> \
    --config /path/to/dataset/train_config.json --out bundle
```

Then, with `pip install multi-accent-frontend`:

```python
from multi_accent_frontend import Frontend
Frontend.from_pretrained("bundle").transcribe("hello world", accent="base")
```
