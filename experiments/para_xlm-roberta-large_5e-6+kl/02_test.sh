#!/bin/bash
#SBATCH --job-name=test
#SBATCH --out=test.%A_%a.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --array=1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate xfever
fi

set -ex

# Get current dirname
expr="$(basename "$PWD")"

pretrained_lr="${expr#*_}"
pretrained="${pretrained_lr%_*}"
max_len=128
model_dir="${pretrained}-${max_len}-mod"

unset -v latest

for file in "${model_dir}/checkpoints"/*.ckpt; do
  [[ "${file}" -nt "${latest}" ]] && latest="${file}"
done

if [[ -z "${latest}" ]]; then
  echo "Cannot find any checkpoint in ${model_dir}"
  exit
fi

for lang in 'en' 'es' 'fr' 'id' 'ja' 'zh'; do
  in_dir=../data/"${lang}"
  out_dir="${pretrained}-${max_len}-${lang}-out"
  mkdir -p "${out_dir}"

  for fname in 'test' 'test.6h' 'test.6h.human'; do
    in_file="${in_dir}"/"${fname}".jsonl
    out_file="${out_dir}"/"${fname}".prob
    if [[ -f "${in_file}" && ! -f "${out_file}" ]]; then
      HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python '../../predict.py' \
        --checkpoint_file "${latest}" \
        --in_file "${in_file}" \
        --out_file "${out_file}" \
        --batch_size 128 \
        --gpus 1

      python '../../evaluate.py' \
        --gold_file "${in_file}" \
        --prob_file "${out_file}" \
        --out_file "${out_dir}"/eval."${fname}".txt
    fi
  done
done
