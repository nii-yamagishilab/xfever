# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Authors: Yi-Chen Chang (yichen@nlplab.cc), Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import numpy as np
import pytorch_lightning as pl
import torch
from argparse import Namespace
from filelock import FileLock
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
from lightning_base import BaseTransformer, generic_train
from modeling_base import BaseModel, ConsistencyModel
from processors import (
    FactVerificationProcessor,
    compute_metrics,
    convert_examples_to_features,
)

MODEL_NAMES_MAPPING = {
    "base": BaseModel,
    "consistency": ConsistencyModel,
}


class FactVerificationTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        num_labels = len(FactVerificationProcessor().get_labels())

        rank_zero_info(f"model_name: {hparams.model_name}")
        model = MODEL_NAMES_MAPPING[hparams.model_name](hparams, num_labels)

        super().__init__(
            hparams,
            num_labels=num_labels,
            model=model,
            config=None if model is None else model.config,
        )

    def create_features(self, set_type, filepath):
        rank_zero_info(f"Create features from [{filepath}]")
        hparams = self.hparams
        processor = FactVerificationProcessor()

        check_data = (
            hparams.compute_consistency_reg
            if hasattr(hparams, "compute_consistency_reg")
            else False
        )

        examples = processor.get_examples(
            filepath, set_type, check_data, self.training, hparams.use_title
        )

        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            set_type=set_type,
            max_length=hparams.max_seq_length,
            label_list=processor.get_labels(),
            threads=hparams.num_workers,
            enable_data_augmentation=(
                hparams.compute_consistency_reg
                if hasattr(hparams, "compute_consistency_reg")
                else None
            ),
        )

        num_examples = processor.get_length(filepath)

        def empty_tensor_1():
            return torch.empty(num_examples, dtype=torch.long)

        def empty_tensor_2():
            return torch.empty((num_examples, hparams.max_seq_length), dtype=torch.long)

        input_ids = empty_tensor_2()
        attention_mask = empty_tensor_2()
        token_type_ids = empty_tensor_2()
        labels = empty_tensor_1()

        if (
            set_type in ["train", "dev"]
            and hasattr(hparams, "compute_consistency_reg")
            and hparams.compute_consistency_reg
        ):
            input_ids_lang = empty_tensor_2()
            attention_mask_lang = empty_tensor_2()
            token_type_ids_lang = empty_tensor_2()

        for i, feature in enumerate(features):
            # For English data
            input_ids[i] = torch.tensor(feature["input_ids"])
            attention_mask[i] = torch.tensor(feature["attention_mask"])
            if "token_type_ids" in feature and feature["token_type_ids"] is not None:
                token_type_ids[i] = torch.tensor(feature["token_type_ids"])
            labels[i] = torch.tensor(feature["label"])

            if (
                set_type in ["train", "dev"]
                and hasattr(hparams, "compute_consistency_reg")
                and hparams.compute_consistency_reg
            ):
                # For other langauges
                input_ids_lang[i] = torch.tensor(feature["input_ids_lang"])
                attention_mask_lang[i] = torch.tensor(feature["attention_mask_lang"])
                if (
                    "token_type_ids_lang" in feature
                    and feature["token_type_ids_lang"] is not None
                ):
                    token_type_ids_lang[i] = torch.tensor(
                        feature["token_type_ids_lang"]
                    )

        if (
            set_type in ["train", "dev"]
            and hasattr(hparams, "compute_consistency_reg")
            and hparams.compute_consistency_reg
        ):
            return [
                input_ids,
                attention_mask,
                token_type_ids,
                labels,
                input_ids_lang,
                attention_mask_lang,
                token_type_ids_lang,
            ]

        else:
            return [input_ids, attention_mask, token_type_ids, labels]

    def cached_feature_file(self, mode):
        dirname = "xfever_" + Path(self.hparams.data_dir).parts[-1]
        feat_dirpath = Path(self.hparams.cache_dir) / dirname
        feat_dirpath.mkdir(parents=True, exist_ok=True)
        pt = self.hparams.pretrained_model_name.replace("/", "__")
        return (
            feat_dirpath
            / f"cached_{mode}_{pt}_{self.hparams.model_name}_{self.hparams.seed}"
        )

    def prepare_data(self):
        if self.training:
            for dataset_type in ["train", "dev"]:
                if dataset_type == "dev" and self.hparams.skip_validation:
                    continue
                cached_feature_file = self.cached_feature_file(dataset_type)
                lock_path = cached_feature_file.with_suffix(".lock")
                with FileLock(lock_path):
                    if (
                        cached_feature_file.exists()
                        and not self.hparams.overwrite_cache
                    ):
                        rank_zero_info(f"Feature file [{cached_feature_file}] exists!")
                        continue

                    filepath = Path(self.hparams.data_dir) / f"{dataset_type}.jsonl"
                    assert filepath.exists(), f"Cannot find [{filepath}]"
                    feature_list = self.create_features(dataset_type, filepath)
                    rank_zero_info(f"\u2728 Saving features to [{cached_feature_file}]")
                    torch.save(feature_list, cached_feature_file)

    def init_parameters(self):
        base_name = self.config.model_type  # e.g., bert, roberta, ...
        no_init = [base_name]
        rank_zero_info(f"\U0001F4A5 Force no_init to [{base_name}]")
        if self.hparams.load_weights:
            no_init += ["classifier"]
            rank_zero_info("\U0001F4A5 Force no_init to [classifier]")
        if self.hparams.no_init:
            no_init += self.hparams.no_init
            rank_zero_info(f"\U0001F4A5 Force no_init to {self.hparams.no_init}")
        for n, p in self.model.named_parameters():
            if any(ni in n for ni in no_init):
                continue
            rank_zero_info(f"Initialize [{n}]")
            if "bias" not in n:
                if hasattr(self.config, "initializer_range"):
                    p.data.normal_(mean=0.0, std=self.config.initializer_range)
                else:
                    p.data.normal_(mean=0.0, std=0.02)
            else:
                p.data.zero_()

    def get_dataloader(self, mode, batch_size, num_workers):
        if self.training and mode == "dev" and self.hparams.skip_validation:
            return None
        cached_feature_file = self.cached_feature_file(mode)
        assert cached_feature_file.exists(), f"Cannot find [{cached_feature_file}]"
        feature_list = torch.load(cached_feature_file)
        shuffle = True if "train" in mode and self.training else False
        rank_zero_info(
            f"Load features from [{cached_feature_file}] with shuffle={shuffle}"
        )
        return DataLoader(
            TensorDataset(*feature_list),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def build_inputs(self, batch):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.config.model_type not in {"distilbert", "bart"}:
            inputs["token_type_ids"] = (
                batch[2]
                if self.config.model_type in ["bert", "xlnet", "albert"]
                else None
            )
        if (
            hasattr(self.hparams, "compute_consistency_reg")
            and self.hparams.compute_consistency_reg
            and len(batch) > 4
        ):
            inputs["input_ids_lang"] = batch[4]
            inputs["attention_mask_lang"] = batch[5]
            if self.config.model_type not in {"distilbert", "bart"}:
                inputs["token_type_ids_lang"] = (
                    batch[6]
                    if self.config.model_type in ["bert", "xlnet", "albert"]
                    else None
                )
        if (
            hasattr(self.hparams, "consistency_reg_func1")
            and self.hparams.consistency_reg_func1
        ):
            inputs["consistency_reg_func1"] = self.hparams.consistency_reg_func1
        if (
            hasattr(self.hparams, "consistency_reg_func2")
            and self.hparams.consistency_reg_func2
        ):
            inputs["consistency_reg_func2"] = self.hparams.consistency_reg_func2
        if (
            hasattr(self.hparams, "lambda_consistency1")
            and self.hparams.lambda_consistency1
        ):
            inputs["lambda_consistency1"] = self.hparams.lambda_consistency1
        if (
            hasattr(self.hparams, "lambda_consistency2")
            and self.hparams.lambda_consistency2
        ):
            inputs["lambda_consistency2"] = self.hparams.lambda_consistency2
        return inputs

    def base_training_step(self, inputs, batch_idx):
        outputs = self(**inputs)
        log_dict = {
            "train_loss": outputs.loss.detach().cpu(),
            "lr": self.lr_scheduler.get_last_lr()[-1],
        }
        if outputs.loss_en:
            log_dict["train_loss_en"] = outputs.loss_en.detach().cpu()
        if outputs.loss_lang:
            log_dict["train_loss_lang"] = outputs.loss_lang.detach().cpu()
        if outputs.reg_consistency1:
            log_dict["train_reg_consistency1"] = outputs.reg_consistency1.detach().cpu()
        if outputs.reg_consistency2:
            log_dict["train_reg_consistency2"] = outputs.reg_consistency2.detach().cpu()
        self.log_dict(log_dict)
        return outputs.loss

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            inputs = self.build_inputs(batch["train1"])
        else:
            inputs = self.build_inputs(batch)

        return self.base_training_step(inputs, batch_idx)

    def validation_step(self, batch, batch_idx):
        inputs = self.build_inputs(batch)
        outputs = self(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return {
            "loss": outputs.loss.detach().cpu(),
            "probs": probs.detach().cpu().numpy(),
            "labels": inputs["labels"].detach().cpu().numpy(),
        }

    def predict_step(self, batch, batch_idx):
        inputs = self.build_inputs(batch)
        outputs = self(**inputs)
        # return outputs.penultimate_layer.detach().cpu()
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs

    def validation_epoch_end(self, outputs):
        avg_loss = (
            torch.stack([x["loss"] for x in outputs]).mean().detach().cpu().item()
        )
        labels = np.concatenate([x["labels"] for x in outputs], axis=0)
        probs = np.concatenate([x["probs"] for x in outputs], axis=0)
        results = {
            **{"loss": avg_loss},
            **compute_metrics(probs, labels),
        }
        self.log_dict({f"val_{k}": torch.tensor(v) for k, v in results.items()})

    def load_weights(self, checkpoint):
        rank_zero_info(f"Loading model weights from [{checkpoint}]")
        checkpoint = torch.load(
            checkpoint,
            map_location=lambda storage, loc: storage,
        )

        ckpt_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
        ckpt_dict = {k: v for k, v in ckpt_dict.items() if k in model_dict}
        assert len(ckpt_dict), "Cannot find shareable weights"
        model_dict.update(ckpt_dict)
        self.load_state_dict(model_dict)

    @staticmethod
    def add_model_specific_args(parser):
        BaseTransformer.add_model_specific_args(parser)
        parser.add_argument("--cache_dir", type=str, default="/tmp")
        parser.add_argument("--overwrite_cache", action="store_true")
        parser.add_argument("--save_all_checkpoints", action="store_true")
        parser.add_argument("--max_seq_length", type=int, default=128)
        parser.add_argument("--use_title", action="store_true")
        parser.add_argument("--load_weights", type=str, default=None)
        parser.add_argument("--no_init", nargs="+", default=[])
        parser.add_argument("--classifier_dropout_prob", type=float, default=0.1)
        return parser


def build_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = FactVerificationTransformer.add_model_specific_args(parser)
    return parser.parse_args()


def main():
    args = build_args()

    if args.seed > 0:
        pl.seed_everything(args.seed)

    model = FactVerificationTransformer(args)

    if args.load_weights is not None:
        model.load_weights(args.load_weights)

    ckpt_dirpath = Path(args.default_root_dir) / "checkpoints"
    ckpt_dirpath.mkdir(parents=True, exist_ok=True)

    monitor, mode, ckpt_filename = None, "min", "{epoch}-{step}"
    dev_filepath = Path(args.data_dir) / "dev.jsonl"
    if dev_filepath.exists() and not args.skip_validation:
        monitor, mode = "val_acc", "max"
        ckpt_filename = "{epoch}-{step}-{" + monitor + ":.4f}"

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dirpath,
            filename=ckpt_filename,
            monitor=monitor,
            mode=mode,
            save_top_k=-1 if args.save_all_checkpoints else 1,
        )
    )

    if monitor is not None:
        callbacks.append(
            EarlyStopping(monitor=monitor, mode=mode, patience=args.patience)
        )

    generic_train(model, args, callbacks)


if __name__ == "__main__":
    main()
