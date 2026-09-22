#!/usr/bin/env bash
# Downloads the paper corpus used by the benchmark harness (arXiv mirror).
# Stored under benchmarks/corpus/, which is gitignored. Redistribution is
# covered by arXiv's preprint licensing; keep the corpus local.
set -euo pipefail

CORPUS="$(cd "$(dirname "$0")/corpus" && pwd)"
mkdir -p "$CORPUS"

declare -A PAPERS=(
  [1301.3781]="word2vec"
  [1706.03762]="attention"
  [1810.04805]="bert"
  [2010.11929]="vit"
  [1512.03385]="resnet"
)

for id in "${!PAPERS[@]}"; do
  name="${PAPERS[$id]}"
  dest="$CORPUS/$id-$name.pdf"
  if [[ -f "$dest" ]]; then
    echo "skip $dest (exists)"
    continue
  fi
  echo "fetching $name ($id)"
  curl -fL --retry 3 -o "$dest" "https://arxiv.org/pdf/$id"
done

echo "done: $(ls "$CORPUS" | wc -l) pdfs in $CORPUS"