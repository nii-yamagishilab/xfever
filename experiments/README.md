The model checkpoints, example outputs, and preprocessed data are available at https://zenodo.org/records/8206962.

## Getting started with checkpoints

You may try reproducing our results with the provided model checkpoints. For example, let's try a model with non-parallel training and mBERT.

First, prepare data:
```shell
conda activate xfever
cd xfever/experiments/data/
sh 01_prepare.sh
cd ..
```

Then, download a model checkpoint and uncompress it:
```shell
wget https://zenodo.org/records/8206962/files/mixed_bert-base-multilingual-cased_2e-5.tar.gz
tar xvfz mixed_bert-base-multilingual-cased_2e-5.tar.gz
```

Move into the experiment folder, and create a temporary directory to store existing outputs:
```shell
cd mixed_bert-base-multilingual-cased_2e-5/
mkdir -p tmp
mv *-out tmp/
```

Next, try to make predictions on the Japanese test set.
```shell
mkdir -p bert-base-multilingual-cased-128-ja-out
python ../../predict.py --checkpoint_file bert-base-multilingual-cased-128-mod/checkpoints/epoch=0-step=33387-val_acc=0.8759.ckpt --in_file ../data/ja/test.jsonl --out_file bert-base-multilingual-cased-128-ja-out/test.prob --batch_size 128 --gpus 1
```

If everything works properly, you should see something like:
```shell
model_name: base
Using 16bit native Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Create features from [../data/ja/test.jsonl]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 11710/11710 [00:00<00:00, 12144.60it/s]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Predicting DataLoader 0: 100%|██████████████████████████████████████████████████████████████████████████████████| 92/92 [00:08<00:00, 11.22it/s]
Prediction took '0:00:12.765408'
Save output probabilities to bert-base-multilingual-cased-128-ja-out/test.prob
```

Finally, evaluate the predicted results:
```shell
python ../../evaluate.py --gold_file ../data/ja/test.jsonl --prob_file bert-base-multilingual-cased-128-ja-out/test.prob --out_file bert-base-multilingual-cased-128-ja-out/eval.test.txt
```

You should see:
```shell
Confusion Matrix:
      S     R     N
S  3633   258   128
R   473  3627   258
N   285   303  2745

          S     R     N
Prec:  82.7  86.6  87.7
Rec:   90.4  83.2  82.4
F1:    86.4  84.9  84.9

ACC: 85.4
ECE: 4.2
```

The above ACC and ECE scores correspond to the column "ja" in Tables 4 and 6 in [our paper](https://aclanthology.org/2023.rocling-1.1/).


## Training, prediction, and evaluation

Each subdirectory name in `experiments/` is roughly in the format of `[method]_[llm]_[params]`, which will be parsed by the scripts inside it. Each subdirectory contains ordered scripts: `01_train.sh` and `02_test.sh`. 
If you run them sequentially, you should be able to reproduce the results in the paper.


### Step 1: Train

Let's try zero-shot learning with the English training/dev sets and mBERT.
```shell
cd en_bert-base-multilingual-cased_2e-5/
sbatch [option] 01_train.sh
```

The script `01_train.sh` aims to work with the `sbatch` command on Slurm.
It can also work with the normal `sh` command with a slight modification.

### Step 2: Predict on the test set

```shell
sbatch [option] 02_test.sh
```
This script submits a job to Slurm, which runs predictions/evaluations on all the test sets.

If everything works properly, we should see something like:
```
tail -n 2 bert-base-multilingual-cased-128-*-out/eval.test.txt
==> bert-base-multilingual-cased-128-en-out/eval.test.txt <==
ACC: 87.9
ECE: 6.0

==> bert-base-multilingual-cased-128-es-out/eval.test.txt <==
ACC: 83.7
ECE: 8.5

==> bert-base-multilingual-cased-128-fr-out/eval.test.txt <==
ACC: 84.3
ECE: 7.9

==> bert-base-multilingual-cased-128-id-out/eval.test.txt <==
ACC: 82.6
ECE: 9.2

==> bert-base-multilingual-cased-128-ja-out/eval.test.txt <==
ACC: 72.4
ECE: 14.6

==> bert-base-multilingual-cased-128-zh-out/eval.test.txt <==
ACC: 82.1
ECE: 8.6

```
The above ACC and ECE scores correspond to the row "Zero-shot $J_z$" in Tables 4 and 6 in [our paper](https://aclanthology.org/2023.rocling-1.1/).
