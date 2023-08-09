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
    kl,
    j,
    js,
    mse,
    cos,
)

REG_FCT = {
    "kl": kl,
    "j": j,
    "js": js,
    "mse": mse,
    "cos": cos,
}


class BaseModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    loss_en: torch.FloatTensor = None
    loss_lang: torch.FloatTensor = None
    reg_consistency1: torch.FloatTensor = None
    reg_consistency2: torch.FloatTensor = None
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
        consistency_reg_func1=None,
        consistency_reg_func2=None,
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

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_en = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            loss = loss_en

        loss_lang = None
        reg_consistency1 = None
        reg_consistency2 = None
        if input_ids_lang is not None:
            encoder_outputs_lang = model(input_ids_lang, **valid_kwargs)
            features_lang = encoder_outputs_lang.last_hidden_state[
                :, 0
            ]  # equiv. to [CLS]
            logits_lang, penultimate_layer_lang = self.classifier(features_lang)

            if labels is not None:
                loss_fct_lang = CrossEntropyLoss()
                loss_lang = loss_fct_lang(
                    logits_lang.view(-1, self.config.num_labels), labels.view(-1)
                )
                loss += loss_lang

            if consistency_reg_func1:
                reg_consistency1 = REG_FCT[consistency_reg_func1](
                    logits, logits_lang, self.config.num_labels
                )
                loss += lambda_consistency1 * reg_consistency1

            if consistency_reg_func2:
                func, layer = consistency_reg_func2.split("-")
                if layer == "feat":
                    layer = features
                    layer_lang = features_lang
                elif layer == "penu":
                    layer = penultimate_layer
                    layer_lang = penultimate_layer_lang
                else:
                    raise KeyError(layer)
                reg_consistency2 = REG_FCT[func](layer, layer_lang)
                loss += lambda_consistency2 * reg_consistency2

        return BaseModelOutput(
            loss=loss,
            loss_en=loss_en,
            loss_lang=loss_lang,
            reg_consistency1=reg_consistency1,
            reg_consistency2=reg_consistency2,
            logits=logits,
            penultimate_layer=penultimate_layer,
        )
