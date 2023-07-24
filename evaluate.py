# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Authors: Yi-Chen Chang (yichen@nlplab.cc), Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import io
import jsonlines
import pandas as pd
import numpy as np
from processors import FactVerificationProcessor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def load_pred_labels(filename):
    probs = np.loadtxt(filename, dtype=np.float64)
    pred_labels = np.argmax(probs, axis=1)
    i2label = {
        i: label for i, label in enumerate(FactVerificationProcessor().get_labels())
    }
    return [i2label[pred][0] for pred in pred_labels]


def load_gold_labels(filename):
    return [line["label"][0] for line in jsonlines.open(filename)]


def read_gold_labels(filename):
    labels = []
    for line in jsonlines.open(filename):
        if "gold_label" in line:
            labels.append(line["gold_label"][0])
        elif "label" in line:
            labels.append(line["label"][0])
        else:
            raise KeyError("Cannot find label field")
    label2index = {
        label: i for i, label in enumerate(FactVerificationProcessor().get_labels())
    }
    return [label2index[label] for label in labels]


def ece_score(py, y_test, n_bins):
    # Modified from https://github.com/sirius8050/Expected-Calibration-Error/blob/master/ECE.py
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)

    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]

    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))

    return ece / sum(Bm)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--prob_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--n_bins", type=int, default=20)
    args = parser.parse_args()
    return args


def main():
    args = build_args()
    gold_labels = load_gold_labels(args.gold_file)
    pred_labels = load_pred_labels(args.prob_file)
    pred_probs = np.loadtxt(args.prob_file, dtype=np.float64)
    labels = FactVerificationProcessor().get_labels()
    prec = (
        precision_score(
            gold_labels, pred_labels, labels=labels, average=None, zero_division=0
        )
        * 100.0
    )
    rec = (
        recall_score(
            gold_labels, pred_labels, labels=labels, average=None, zero_division=0
        )
        * 100.0
    )
    f1 = (
        f1_score(gold_labels, pred_labels, labels=labels, average=None, zero_division=0)
        * 100.0
    )

    acc = accuracy_score(gold_labels, pred_labels) * 100.0
    mat = confusion_matrix(gold_labels, pred_labels, labels=labels)
    df = pd.DataFrame(mat, columns=labels, index=labels)
    df2 = pd.DataFrame([prec, rec, f1], columns=labels, index=["Prec:", "Rec:", "F1:"])
    y_test = read_gold_labels(args.gold_file)
    ece = ece_score(pred_probs, y_test, args.n_bins)

    results = "\n".join(
        [
            "Confusion Matrix:",
            f"{df}",
            "",
            f"{df2.round(1)}",
            "",
            f"ACC: {acc.round(1)}",
            f"ECE: {ece*100:.1f}",
        ]
    )

    print(results)

    with io.open(args.out_file, "w", encoding="utf-8") as f:
        f.write(results + "\n")


if __name__ == "__main__":
    main()
