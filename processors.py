# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Authors: Yi-Chen Chang (yichen@nlplab.cc), Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import os
import io
import jsonlines
import re
import unicodedata
import numpy as np
import random
from transformers import InputExample, DataProcessor  # , InputFeatures
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

tokenizer = None


def do_code_switch(text, code_switch_languages, bilingual_dict):
    tokens = text.split(" ")
    num_tokens = len(tokens)
    num_switched_tokens = 0
    new_tokens = []
    for token in tokens:
        if (
            random.uniform(0, 1) > 0.5
            and (num_switched_tokens / num_tokens) <= 0.3
            and token.lower() in bilingual_dict["ja"]
        ):
            new_token = bilingual_dict["ja"][token.lower()][
                random.randint(0, len(bilingual_dict["ja"][token.lower()]) - 1)
            ]
            new_tokens.append(new_token)
            num_switched_tokens += 1
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)


def convert_example_to_features(
    example,
    max_length,
    label_map,
    set_type,
    code_switch_languages,
    bilingual_dict,
    enable_data_augmentation,
    translate_dict,
    languages,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if code_switch_languages:
        text_a = do_code_switch(example.text_a, code_switch_languages, bilingual_dict)
        text_b = do_code_switch(example.text_b, code_switch_languages, bilingual_dict)
    else:
        text_a = example.text_a
        text_b = example.text_b

    if set_type in ["train", "dev"] and enable_data_augmentation:

        text_a, translated_text_a = text_a.split("[SEP]")
        text_b, translated_text_b = text_b.split("[SEP]")

        # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
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
        # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
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
    code_switch_languages="",
    bilingual_dicts_path=None,
    enable_data_augmentation=False,
    translation_path=None,
    languages="",
):

    if label_list is None:
        processor = FactVerificationProcessor()
        label_list = processor.get_labels()

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    bilingual_dict = {}
    if code_switch_languages:

        assert bilingual_dicts_path is not None

        cs_langs = code_switch_languages.split("+")
        for lang in cs_langs:
            with open(
                os.path.join(bilingual_dicts_path, f"en-{lang}.txt"),
                "r",
                encoding="utf-8",
            ) as reader:
                raw = reader.readlines()
            bilingual_dict[lang] = {}
            for line in raw:
                line = line.strip()
                try:
                    token_en, token_lang = line.split("\t")
                except ValueError:
                    token_en, token_lang = line.split(" ")
                if token_en not in bilingual_dict[lang]:
                    bilingual_dict[lang][token_en] = [token_lang]
                else:
                    bilingual_dict[lang][token_en].append(token_lang)

    translate_dict = None

    # if set_type in ["train", "dev"] and enable_data_augmentation:

    #     with open(
    #         os.path.join(translation_path, f"translate-{set_type}-dict.json"), "r"
    #     ) as file:
    #         translate_dict = json.load(file)

    threads = min(threads, cpu_count())
    with Pool(
        threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)
    ) as p:
        annotate_ = partial(
            convert_example_to_features,
            max_length=max_length,
            label_map=label_map,
            set_type=set_type,
            code_switch_languages=code_switch_languages,
            bilingual_dict=bilingual_dict,
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
                    # continue

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
