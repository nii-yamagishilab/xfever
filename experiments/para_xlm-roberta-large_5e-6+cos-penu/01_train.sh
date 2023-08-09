#!/bin/bash
#SBATCH --job-name=train
#SBATCH --out=train.%A_%a.log
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

seed=3435

# Get current dirname
expr="$(basename "$PWD")"
data="${expr%%_*}"
data_dir="../data/${data}"
pretrained_lr_reg="${expr#*_}"
pretrained="${pretrained_lr_reg%_*}"
lr_reg="${pretrained_lr_reg#*_}"
lr="${lr_reg%+*}"
reg="${lr_reg#*+}"

max_len=128
model_dir="${pretrained}-${max_len}-mod"

if [[ "${data}" == 'para' ]]; then
  model_name='consistency'
else
  model_name='base'
fi

if [[ -d "${model_dir}" ]]; then
  echo "${model_dir} exists! Skip training."
  exit
fi

python '../../train.py' \
  --data_dir "${data_dir}" \
  --default_root_dir "${model_dir}" \
  --model_name "${model_name}" \
  --pretrained_model_name "${pretrained}" \
  --max_seq_length "${max_len}" \
  --seed "${seed}" \
  --cache_dir "/local/$(whoami)" \
  --overwrite_cache \
  --max_epochs 10 \
  --learning_rate "${lr}" \
  --consistency_reg_func2 "${reg}" \
  --lambda_consistency2 1.0 \
  --train_batch_size 32 \
  --accumulate_grad_batches 1 \
  --adafactor \
  --warmup_ratio 0.02 \
  --gradient_clip_val 1.0 \
  --precision 16 \
  --deterministic true \
  --gpus 1
