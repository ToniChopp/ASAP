### copy from CXR Bert
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor as T
from transformers import BertConfig, BertForMaskedLM
from transformers.modeling_outputs import ModelOutput
from torch.nn import CrossEntropyLoss

import ipdb


BERTTupleOutput = Tuple[T, T, T, T, T]


class CXRBertConfig(BertConfig):
    """
    Config class for CXR-BERT model.
    :param projection_size: Dimensionality of the joint latent space.
    """

    model_type = "cxr-bert"

    def __init__(
        self, 
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.25,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        projection_size=128,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        gradient_checkpointing=False,
        torch_dtype="float32",
        **kwargs
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.projection_size = projection_size
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.torch_dtype = torch_dtype
        self.return_loss = True


@dataclass
class CXRBertOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    logits: Optional[torch.FloatTensor] = None
    cls_projected_embedding: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertProjectionHead(nn.Module):
    """Projection head to be used with BERT CLS token.

    This is similar to ``BertPredictionHeadTransform`` in HuggingFace.

    :param config: Configuration for BERT.
    """

    def __init__(self, config: CXRBertConfig) -> None:
        super().__init__()
        self.dense_to_hidden = nn.Linear(config.hidden_size, config.projection_size)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(config.projection_size, eps=1e-12)
        self.dense_to_output = nn.Linear(config.projection_size, config.projection_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_to_hidden(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense_to_output(hidden_states)

        return hidden_states


class CXRBertModel(BertForMaskedLM):
    """
    Implements the CXR-BERT model outlined in the manuscript:
    Boecking et al. "Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing", 2022
    https://link.springer.com/chapter/10.1007/978-3-031-20059-5_1

    Extends the HuggingFace BertForMaskedLM model by adding a separate projection head. The projection "[CLS]" token is
    used to align the latent vectors of image and text modalities.
    """

    config_class = CXRBertConfig  # type: ignore

    def __init__(self, config: CXRBertConfig):
        super().__init__(config)

        self.cls_projection_head = BertProjectionHead(config)
        self.return_loss = config.return_loss
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_cls_projected_embedding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[BERTTupleOutput, CXRBertOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bert_for_masked_lm_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = bert_for_masked_lm_output.hidden_states[-1]
        cls_projected_embedding = (
            self.cls_projection_head(last_hidden_state[:, 0, :]) if output_cls_projected_embedding else None
        )


        if return_dict:
            return CXRBertOutput(
                last_hidden_state=last_hidden_state,
                logits=bert_for_masked_lm_output.logits,
                cls_projected_embedding=cls_projected_embedding,
                hidden_states=bert_for_masked_lm_output.hidden_states if output_hidden_states else None,
                attentions=bert_for_masked_lm_output.attentions,
            )
        else:
            return (
                last_hidden_state,
                bert_for_masked_lm_output.logits,
                cls_projected_embedding,
                bert_for_masked_lm_output.hidden_states,
                bert_for_masked_lm_output.attentions,
            )

    def get_projected_text_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize_embeddings: bool = True
    ) -> torch.Tensor:
        """
        Returns l2-normalised projected cls token embeddings for the given input token ids and attention mask.
        The joint latent space is trained using a contrastive objective between image and text data modalities.

        :param input_ids: (batch_size, sequence_length)
        :param attention_mask: (batch_size, sequence_length)
        :param normalize_embeddings: Whether to l2-normalise the embeddings.
        :return: (batch_size, projection_size)
        """

        outputs = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, output_cls_projected_embedding=True, return_dict=True
        )
        assert isinstance(outputs, CXRBertOutput)

        cls_projected_embedding = outputs.cls_projected_embedding
        assert cls_projected_embedding is not None

        if normalize_embeddings:
            return F.normalize(cls_projected_embedding, dim=1)

        return cls_projected_embedding




class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()

        self.model = CXRBertModel(CXRBertConfig())

    def forward(self, batch, temperature=0.07):
        ids_f = batch['ids_f'].cuda()
        labels_f = batch['labels_f'].cuda()
        attention_mask_f = batch['attention_mask_f'].cuda()
        token_type_ids_f = batch['type_ids_f'].cuda()
        ids_i = batch['ids_i'].cuda()
        labels_i = batch['labels_i'].cuda()
        attention_mask_i = batch['attention_mask_i'].cuda()
        token_type_ids_i = batch['type_ids_i'].cuda()


        output_f = self.model(input_ids=ids_f, attention_mask=attention_mask_f, token_type_ids=token_type_ids_f)
        output_i = self.model(input_ids=ids_i, attention_mask=attention_mask_i, token_type_ids=token_type_ids_i)
        # ipdb.set_trace()
        feature_f = output_f.last_hidden_state[:, 0, :]
        feature_i = output_i.last_hidden_state[:, 0, :]
        prediction_scores_f = output_f.logits
        prediction_scores_i = output_i.logits

        masked_lm_loss_f = None
        masked_lm_loss_i = None
        mlm_loss_fct = CrossEntropyLoss(reduction='none')  # -100 index = padding token
        contrastive_loss_fct = nn.CrossEntropyLoss()

        if labels_f is not None:
            masked_lm_loss_f = mlm_loss_fct(prediction_scores_f.view(-1, prediction_scores_f.shape[-1]), labels_f.view(-1))
            masked_lm_loss_f = masked_lm_loss_f.mean()

        if labels_i is not None:
            masked_lm_loss_i = mlm_loss_fct(prediction_scores_i.view(-1, prediction_scores_i.shape[-1]), labels_i.view(-1))
            masked_lm_loss_i = masked_lm_loss_i.mean()
        
        masked_lm_loss = masked_lm_loss_f + masked_lm_loss_i

        feature_f = F.normalize(feature_f, p=2, dim=1)
        feature_i = F.normalize(feature_i, p=2, dim=1)

        logits_f = torch.matmul(feature_f, feature_i.T) / temperature
        logits_i = torch.matmul(feature_i, feature_f.T) / temperature

        labels = torch.arange(logits_f.size(0)).cuda()

        contrastive_loss_fi = contrastive_loss_fct(logits_f, labels)
        contrastive_loss_if = contrastive_loss_fct(logits_i, labels)
        contrastive_loss = contrastive_loss_fi + contrastive_loss_if

        return masked_lm_loss, contrastive_loss