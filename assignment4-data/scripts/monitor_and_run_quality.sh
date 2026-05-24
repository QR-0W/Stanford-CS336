#!/usr/bin/env bash
set -u

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
TEST_PY="${TEST_PY:-$PY}"
URL="https://downloads.cs.stanford.edu/nlp/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz"
WIKI_URLS="$ROOT/data/enwiki-20240420-extracted_urls.txt.gz"
LOG="${QUALITY_RUN_LOG:-$ROOT/data/quality_full_run.log}"
LOCK="${QUALITY_RUN_LOCK:-$ROOT/data/quality_full_run.lock}"
STAMP="${QUALITY_RUN_STAMP:-$ROOT/data/quality_full_run.done}"
URL_OUTPUT="${QUALITY_URL_OUTPUT:-data/wiki_positive_urls.txt}"
WARC_PREFIX="${QUALITY_WARC_PREFIX:-data/wiki_positive_urls}"
TRAIN_OUTPUT="${QUALITY_TRAIN_OUTPUT:-data/quality_classifier.full.train.txt}"
MODEL_OUTPUT="${QUALITY_MODEL_OUTPUT:-cs336_data/assets/quality_classifier.bin}"

mkdir -p "$ROOT/data" "$ROOT/cs336_data/assets"
exec >> "$LOG" 2>&1

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[$(date -Is)] another quality full run is already active; exiting"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

echo "[$(date -Is)] quality full runner started"
echo "[$(date -Is)] robust range-download + monitor for $WIKI_URLS"

"$PY" scripts/range_download.py \
  --url "$URL" \
  --output "$WIKI_URLS" \
  --log "$ROOT/data/wiki_urls.download.log"

echo "[$(date -Is)] wiki URL file is complete; starting quality classifier run"

cd "$ROOT"
NUM_URLS=${NUM_URLS:-200}
MAX_POS=${MAX_POS:-200}
MAX_NEG=${MAX_NEG:-200}

echo "[$(date -Is)] sampling $NUM_URLS wikipedia reference URLs"
"$PY" scripts/sample_wiki_urls.py \
  --input data/enwiki-20240420-extracted_urls.txt.gz \
  --output "$URL_OUTPUT" \
  --num-urls "$NUM_URLS"

echo "[$(date -Is)] fetching sampled wiki reference pages as WARC"
rm -f "$WARC_PREFIX.warc.gz" "$WARC_PREFIX.warc.warc.gz" "$WARC_PREFIX.warc.warc.gz.open"
wget --timeout=10 --tries=1 --wait=0.1 --random-wait \
  -i "$URL_OUTPUT" \
  --warc-file="$WARC_PREFIX" \
  -O /dev/null || true

WARC="$WARC_PREFIX.warc.gz"
if [ ! -f "$WARC" ] && [ -f "$WARC_PREFIX.warc.warc.gz" ]; then
  WARC="$WARC_PREFIX.warc.warc.gz"
fi

if [ ! -f "$WARC" ]; then
  echo "[$(date -Is)] ERROR: wiki positive WARC was not created"
  exit 1
fi

echo "[$(date -Is)] training quality classifier with WARC=$WARC MAX_POS=$MAX_POS MAX_NEG=$MAX_NEG"
"$PY" scripts/train_quality_classifier.py \
  --positive-warcs "$WARC" \
  --negative-wets data/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz \
  --max-positive-docs "$MAX_POS" \
  --max-negative-docs "$MAX_NEG" \
  --apply-gopher \
  --train-output "$TRAIN_OUTPUT" \
  --model-output "$MODEL_OUTPUT" \
  --epoch 25 \
  --lr 0.5 \
  --word-ngrams 2

echo "[$(date -Is)] running quality tests"
"$TEST_PY" -m pytest tests/test_quality.py -v

echo "[$(date -Is)] running full pytest"
"$TEST_PY" -m pytest -v

echo "[$(date -Is)] quality full runner finished"
touch "$STAMP"
