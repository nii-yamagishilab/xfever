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
    if [[ ! -f toy/"${lang}"/"${split}".sonl ]]; then
      python permute.py --in_file "${lang}"/"${split}".jsonl --out_file _"${lang}"_"${split}".perm.jsonl
      if [[ "${split}" == 'train' ]]; then
        num=2000
      else
        num=600
      fi
      head -n "${num}" _"${lang}"_"${split}".perm.jsonl > toy/"${lang}"/"${split}".sonl
      rm -f _"${lang}"_"${split}".perm.jsonl
    fi
  done
done

wc -l {en,es,fr,id,ja,zh}/*.jsonl
wc -l toy/{en,es,fr,id,ja,zh}/*.sonl
