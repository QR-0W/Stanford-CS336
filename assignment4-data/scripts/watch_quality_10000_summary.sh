#!/usr/bin/env bash
set -u

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
DONE="$ROOT/data/quality_full_10000_parallel.done"
LOG="$ROOT/data/quality_full_10000_parallel_run.log"
SUMMARY="$ROOT/data/quality_full_10000.summary.txt"

while [ ! -f "$DONE" ]; do
  sleep 60
done

{
  echo "quality_full_10000 completed at $(date -Is)"
  echo
  echo "== Artifacts =="
  ls -lh "$ROOT"/data/wiki_positive_urls_10000_shards/*.warc.gz 2>/dev/null || true
  ls -lh "$ROOT/data/quality_classifier.full_10000.train.txt" "$ROOT/cs336_data/assets/quality_classifier.bin" 2>/dev/null || true
  echo
  echo "== Label counts =="
  ROOT="$ROOT" "$PY" - <<'PY'
from collections import Counter
import os
from pathlib import Path
p = Path(os.environ['ROOT']) / 'data/quality_classifier.full_10000.train.txt'
counts = Counter()
if p.exists():
    for line in p.open(encoding='utf-8', errors='replace'):
        if line.startswith('__label__'):
            counts[line.split(maxsplit=1)[0]] += 1
print(dict(counts))
PY
  echo
  echo "== Fixture predictions =="
  ROOT="$ROOT" "$PY" - <<'PY'
import os
from pathlib import Path
import fasttext
root = Path(os.environ['ROOT'])
model_path = root / 'cs336_data/assets/quality_classifier.bin'
if model_path.exists():
    model = fasttext.load_model(str(model_path))
    for rel in ['tests/fixtures/low_quality_cc.txt', 'tests/fixtures/high_quality_wiki_reference.txt']:
        text = (root / rel).read_text(encoding='utf-8')
        labels, probs = model.predict(text.replace('\n', ' '), k=2)
        print(rel, [(label.replace('__label__', ''), float(prob)) for label, prob in zip(labels, probs)])
PY
  echo
  echo "== Test summary lines =="
  grep -E "tests/test_quality.py|=+ .*passed|FAILED|ERROR|quality full runner finished|parallel quality run finished" "$LOG" | tail -80 || true
} > "$SUMMARY" 2>&1
