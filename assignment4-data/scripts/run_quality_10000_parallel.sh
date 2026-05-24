#!/usr/bin/env bash
set -u

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
TEST_PY="${TEST_PY:-$PY}"
URL_SOURCE="$ROOT/data/enwiki-20240420-extracted_urls.txt.gz"
LOG="$ROOT/data/quality_full_10000_parallel_run.log"
LOCK="$ROOT/data/quality_full_10000_parallel.lock"
STAMP="$ROOT/data/quality_full_10000_parallel.done"
URL_OUTPUT="data/wiki_positive_urls_10000.txt"
SHARD_DIR="data/wiki_positive_urls_10000_shards"
TRAIN_OUTPUT="data/quality_classifier.full_10000.train.txt"
MODEL_OUTPUT="cs336_data/assets/quality_classifier.bin"
NUM_URLS=${NUM_URLS:-10000}
MAX_POS=${MAX_POS:-10000}
MAX_NEG=${MAX_NEG:-10000}
JOBS=${JOBS:-8}

mkdir -p "$ROOT/data" "$ROOT/cs336_data/assets"
exec >> "$LOG" 2>&1

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[$(date -Is)] another parallel quality run is already active; exiting"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

cd "$ROOT"
echo "[$(date -Is)] parallel quality run started jobs=$JOBS num_urls=$NUM_URLS max_pos=$MAX_POS max_neg=$MAX_NEG"

gzip -t "$URL_SOURCE"

if [ ! -f "$URL_OUTPUT" ] || [ "$(wc -l < "$URL_OUTPUT")" -lt "$NUM_URLS" ]; then
  echo "[$(date -Is)] sampling $NUM_URLS wikipedia reference URLs"
  "$PY" scripts/sample_wiki_urls.py \
    --input data/enwiki-20240420-extracted_urls.txt.gz \
    --output "$URL_OUTPUT" \
    --num-urls "$NUM_URLS"
fi

rm -rf "$SHARD_DIR"
mkdir -p "$SHARD_DIR"
"$PY" - <<'PY'
from pathlib import Path
root = Path('data')
urls = Path('data/wiki_positive_urls_10000.txt').read_text(encoding='utf-8').splitlines()
jobs = int(__import__('os').environ.get('JOBS', '8'))
out = Path('data/wiki_positive_urls_10000_shards')
out.mkdir(parents=True, exist_ok=True)
for i in range(jobs):
    shard = urls[i::jobs]
    (out / f'shard_{i:02d}.txt').write_text('\n'.join(shard) + ('\n' if shard else ''), encoding='utf-8')
print(f'split {len(urls)} urls into {jobs} shards')
PY

fetch_shard() {
  local shard_file="$1"
  local shard_name
  shard_name="$(basename "$shard_file" .txt)"
  local prefix="$SHARD_DIR/$shard_name"
  echo "[$(date -Is)] fetching $shard_file -> $prefix.warc.gz"
  wget --timeout=10 --tries=1 --wait=0.1 --random-wait \
    -i "$shard_file" \
    --warc-file="$prefix" \
    -O /dev/null \
    > "$prefix.wget.log" 2>&1 || true
  echo "[$(date -Is)] finished $shard_file"
}
export -f fetch_shard
export SHARD_DIR

for shard in "$SHARD_DIR"/shard_*.txt; do
  while [ "$(jobs -pr | wc -l)" -ge "$JOBS" ]; do
    sleep 1
  done
  fetch_shard "$shard" &
done
wait

echo "[$(date -Is)] all shards fetched"
mapfile -t WARCS < <(ls "$SHARD_DIR"/*.warc.gz 2>/dev/null || true)
if [ "${#WARCS[@]}" -eq 0 ]; then
  echo "[$(date -Is)] ERROR: no WARC shards were created"
  exit 1
fi
printf '[%s] WARC shards: %s\n' "$(date -Is)" "${#WARCS[@]}"
ls -lh "$SHARD_DIR"/*.warc.gz

echo "[$(date -Is)] training quality classifier"
"$PY" scripts/train_quality_classifier.py \
  --positive-warcs "${WARCS[@]}" \
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

echo "[$(date -Is)] parallel quality run finished"
touch "$STAMP"
