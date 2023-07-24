# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Authors: Yi-Chen Chang (yichen@nlplab.cc), Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.file_utils import ModelOutput
from utils import (
    KL,
    symmetric_KLDiv,
    symmetric_cross_entropy,
    JSDiv,
    JSDis,
    CosSimilarityLoss,
    EuclideanDistance,
)


class BaseModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    loss_en: torch.FloatTensor = None
    loss_lang: torch.FloatTensor = None
    loss_consistency1: torch.FloatTensor = None
    loss_consistency2: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    penultimate_layer: torch.FloatTensor = None


class Classifier(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_labels,
        dropout=0.1,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        z = torch.relu(x)
        x = self.dropout(z)
        x = self.out_proj(x)
        return x, z


class BaseModel(PreTrainedModel):
    def __init__(self, hparams, num_labels):
        config = AutoConfig.from_pretrained(
            hparams.pretrained_model_name, num_labels=num_labels
        )
        super().__init__(config)
        setattr(
            self,
            self.config.model_type,
            AutoModel.from_pretrained(
                hparams.pretrained_model_name, config=self.config
            ),
        )
        self.classifier = Classifier(
            self.config.hidden_size,
            num_labels,
            dropout=hparams.classifier_dropout_prob,
        )

    def forward(self, input_ids, labels=None, **kwargs):
        model = getattr(self, self.config.model_type)
        model_args_name = set(model.forward.__code__.co_varnames[1:])  # skip self
        valid_kwargs = {
            key: value for (key, value) in kwargs.items() if key in model_args_name
        }
        encoder_outputs = model(input_ids, **valid_kwargs)
        features = encoder_outputs.last_hidden_state[:, 0]  # equiv. to [CLS]
        logits, penultimate_layer = self.classifier(features)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return BaseModelOutput(
            loss=loss, logits=logits, penultimate_layer=penultimate_layer
        )


class ConsistencyModel(PreTrainedModel):
    def __init__(self, hparams, num_labels):
        config = AutoConfig.from_pretrained(
            hparams.pretrained_model_name, num_labels=num_labels
        )
        super().__init__(config)
        setattr(
            self,
            self.config.model_type,
            AutoModel.from_pretrained(
                hparams.pretrained_model_name, config=self.config
            ),
        )
        self.classifier = Classifier(
            self.config.hidden_size,
            num_labels,
            dropout=hparams.classifier_dropout_prob,
        )

    def forward(
        self,
        input_ids,
        input_ids_lang=None,
        labels=None,
        consistency_loss_func1=None,
        consistency_loss_func2=None,
        lambda_ori=0.0,
        lambda_lang=0.0,
        lambda_consistency1=0.0,
        lambda_consistency2=0.0,
        **kwargs,
    ):
        model = getattr(self, self.config.model_type)
        model_args_name = set(model.forward.__code__.co_varnames[1:])  # skip self
        valid_kwargs = {
            key: value for (key, value) in kwargs.items() if key in model_args_name
        }
        encoder_outputs = model(input_ids, **valid_kwargs)
        features = encoder_outputs.last_hidden_state[:, 0]  # equiv. to [CLS]
        logits, penultimate_layer = self.classifier(features)

        # Original Loss
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_en = (
                loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
                * lambda_ori
            )
            loss = loss_en

        loss_lang = None
        loss_consistency1 = None
        loss_consistency2 = None
        if input_ids_lang is not None:
            encoder_outputs_lang = model(input_ids_lang, **valid_kwargs)
            features_lang = encoder_outputs_lang.last_hidden_state[
                :, 0
            ]  # equiv. to [CLS]
            logits_lang, penultimate_layer_lang = self.classifier(features_lang)

            if labels is not None:
                loss_fct_lang = CrossEntropyLoss()
                loss_lang = (
                    loss_fct_lang(
                        logits_lang.view(-1, self.config.num_labels), labels.view(-1)
                    )
                    * lambda_lang
                )

                loss += loss_lang

            if consistency_loss_func1 and consistency_loss_func1 in [
                "KLDiv",
                "KLDiv-reverse",
                "symKLDiv",
                "symCE",
                "JSDiv",
                "JSDis",
            ]:

                if consistency_loss_func1 == "KLDiv":
                    loss_consistency1 = KL(logits, logits_lang, self.config.num_labels)
                elif consistency_loss_func1 == "KLDiv-reverse":
                    loss_consistency1 = KL(logits_lang, logits, self.config.num_labels)
                elif consistency_loss_func1 == "symKLDiv":
                    loss_consistency1 = symmetric_KLDiv(
                        logits, logits_lang, self.config.num_labels
                    )
                elif consistency_loss_func1 == "symCE":
                    loss_consistency1 = symmetric_cross_entropy(
                        logits, logits_lang, self.config.num_labels
                    )
                elif consistency_loss_func1 == "JSDiv":
                    loss_consistency1 = JSDiv(
                        logits, logits_lang, self.config.num_labels
                    )
                elif consistency_loss_func1 == "JSDis":
                    loss_consistency1 = JSDis(
                        logits, logits_lang, self.config.num_labels
                    )

                loss_consistency1 *= lambda_consistency1
                loss += loss_consistency1

            if consistency_loss_func2 and consistency_loss_func2.split("_")[0] in [
                "NegCosSim",
                "EuclideanDis",
            ]:

                if consistency_loss_func2.split("_")[0] == "NegCosSim":
                    if consistency_loss_func2.split("_")[-1] == "features":
                        loss_consistency2 = CosSimilarityLoss(features, features_lang)

                    elif consistency_loss_func2.split("_")[-1] == "penultimate-layer":
                        loss_consistency2 = CosSimilarityLoss(
                            penultimate_layer, penultimate_layer_lang
                        )

                elif consistency_loss_func2.split("_")[0] == "EuclideanDis":
                    if consistency_loss_func2.split("_")[-1] == "features":
                        loss_consistency2 = EuclideanDistance(features, features_lang)

                    elif consistency_loss_func2.split("_")[-1] == "penultimate-layer":
                        loss_consistency2 = EuclideanDistance(
                            penultimate_layer, penultimate_layer_lang
                        )

                loss_consistency2 *= lambda_consistency2
                loss += loss_consistency2

        return BaseModelOutput(
            loss=loss,
            loss_en=loss_en,
            loss_lang=loss_lang,
            loss_consistency1=loss_consistency1,
            loss_consistency2=loss_consistency2,
            logits=logits,
            penultimate_layer=penultimate_layer,
        )
