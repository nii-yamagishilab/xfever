#!/bin/bash

set -ex

for lang in 'en' 'es' 'fr' 'id' 'ja' 'zh'; do
  # Uncompress
  if [[ ! -d "${lang}" ]]; then
    tar xvfz "${lang}.tgz"
  fi
done

wc -l {en,es,fr,id,ja,zh}/*.jsonl

if [[ ! -d 'mixed' ]]; then
  python build_mixed_examples.py --dirpath .
fi

if [[ ! -d 'para' ]]; then
  python build_parallel_examples.py --dirpath .
fi
