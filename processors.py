# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Authors: Yi-Chen Chang (yichen@nlplab.cc), Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import io
import jsonlines
import re
import unicodedata
import numpy as np
from transformers import InputExample, DataProcessor
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

tokenizer = None


def convert_example_to_features(
    example,
    max_length,
    label_map,
    set_type,
    enable_data_augmentation,
    translate_dict,
    languages,
):
    if max_length is None:
        max_length = tokenizer.max_len

    text_a = example.text_a
    text_b = example.text_b

    if set_type in ["train", "dev"] and enable_data_augmentation:
        text_a, translated_text_a = text_a.split("[SEP]")
        text_b, translated_text_b = text_b.split("[SEP]")

        inputs = tokenizer.encode_plus(
            text_a,
            text_b,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            truncation_strategy="only_second",
        )

        translated_inputs = tokenizer.encode_plus(
            translated_text_a,
            translated_text_b,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            truncation_strategy="only_second",
        )

        inputs["input_ids_lang"] = translated_inputs["input_ids"]
        inputs["attention_mask_lang"] = translated_inputs["attention_mask"]
        if "token_type_ids" in translated_inputs:
            inputs["token_type_ids_lang"] = translated_inputs["token_type_ids"]

    else:
        inputs = tokenizer.encode_plus(
            text_a,
            text_b,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            truncation_strategy="only_second",
        )

    label = label_map[example.label]

    return {**inputs, "label": label}


def convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(
    examples,
    tokenizer,
    set_type,
    max_length=None,
    label_list=None,
    threads=8,
    enable_data_augmentation=False,
    translation_path=None,
    languages="",
):

    if label_list is None:
        processor = FactVerificationProcessor()
        label_list = processor.get_labels()

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    translate_dict = None

    threads = min(threads, cpu_count())
    with Pool(
        threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)
    ) as p:
        annotate_ = partial(
            convert_example_to_features,
            max_length=max_length,
            label_map=label_map,
            set_type=set_type,
            enable_data_augmentation=enable_data_augmentation,
            translate_dict=translate_dict,
            languages=languages,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
            )
        )

    if isinstance(features[0], list):
        new_features = []
        for item in features:
            for i in range(len(item)):
                new_features.append(item[i])

        features = new_features

    return features


def compute_metrics(probs, gold_labels):
    assert len(probs) == len(gold_labels)
    pred_labels = np.argmax(probs, axis=1)
    return {"acc": (gold_labels == pred_labels).mean()}


def process_claim(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r" \-LSB\-.*?\-RSB\-", "", text)
    text = re.sub(r"\-LRB\- \-RRB\- ", "", text)
    text = re.sub(" -LRB-", " ( ", text)
    text = re.sub("-RRB-", " )", text)
    text = re.sub("--", "-", text)
    text = re.sub("``", '"', text)
    text = re.sub("''", '"', text)
    return text


def process_title(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub("_", " ", text)
    text = re.sub(" -LRB-", " ( ", text)
    text = re.sub("-RRB-", " )", text)
    text = re.sub("-COLON-", ":", text)
    return text


def process_evidence(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub(" -LSB-.*-RSB-", " ", text)
    text = re.sub(" -LRB- -RRB- ", " ", text)
    text = re.sub("-LRB-", "(", text)
    text = re.sub("-RRB-", ")", text)
    text = re.sub("-COLON-", ":", text)
    text = re.sub("_", " ", text)
    text = re.sub(r"\( *\,? *\)", "", text)
    text = re.sub(r"\( *[;,]", "(", text)
    text = re.sub("--", "-", text)
    text = re.sub("``", '"', text)
    text = re.sub("''", '"', text)
    return text


class FactVerificationProcessor(DataProcessor):
    def get_labels(self):
        """See base class."""
        return ["S", "R", "N"]  # SUPPORTS, REFUTES, NOT ENOUGH INFO

    def get_dummy_label(self):
        return "N"

    def get_length(self, filepath):
        return sum(1 for line in io.open(filepath, "r", encoding="utf8"))

    def get_examples(
        self, filepath, set_type, check_data, training=True, use_title=True
    ):
        examples = []
        count = 0
        for (i, line) in enumerate(jsonlines.open(filepath)):
            guid = f"{set_type}-{i}"
            if check_data:
                claim = line["claim"]
                evidence = line["evidence"]
            else:
                claim = process_claim(line["claim"])
                evidence = process_evidence(line["evidence"])

            if use_title and "page" in line:
                title = process_title(line["page"])

            if "label" in line:
                label = line["label"][0]
            else:
                label = self.get_dummy_label()

            text_a = claim
            text_b = f"{title} : {evidence}" if use_title else evidence

            if check_data:
                if "[SEP]" not in text_a or "[SEP]" not in text_b:
                    count += 1

            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                )
            )
        if check_data:
            print(f"Amount of skipped data: {count}")

        return examples
