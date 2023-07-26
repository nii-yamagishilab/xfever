#!/bin/bash

set -ex

for lang in 'en' 'es' 'fr' 'id' 'ja' 'zh'; do
  # Uncompress
  if [[ ! -d "${lang}" ]]; then
    tar xvfz "${lang}.tgz"
  fi

  # Create toy data
  mkdir -p toy/"${lang}"
  for split in train dev test; do
    if [[ ! -f toy/"${lang}"/"${split}".jsonl ]]; then
      python permute.py --in_file "${lang}"/"${split}".jsonl --out_file _"${lang}"_"${split}".perm.jsonl
      if [[ "${split}" == 'train' ]]; then
        num=1000
      else
        num=300
      fi
      head -n "${num}" _"${lang}"_"${split}".perm.jsonl > toy/"${lang}"/"${split}".jsonl
      rm -f _"${lang}"_"${split}".perm.jsonl
    fi
  done
done

wc -l {en,es,fr,id,ja,zh}/*.jsonl
wc -l toy/{en,es,fr,id,ja,zh}/*.jsonl

if [[ ! -d 'mixed' ]]; then
  python build_mixed_examples.py --dirpath .
  python build_mixed_examples.py --dirpath toy
fi

if [[ ! -d 'para' ]]; then
  python build_parallel_examples.py --dirpath .
  python build_parallel_examples.py --dirpath toy
fi
