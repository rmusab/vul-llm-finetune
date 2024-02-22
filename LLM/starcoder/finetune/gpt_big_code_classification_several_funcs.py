import collections
import importlib
from argparse import Namespace
from importlib import util
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import GPTBigCodeModel, GPTBigCodePreTrainedModel, GPTBigCodeConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from utils.focal_loss import FocalLoss
import bitsandbytes as bnb

def get_contributing_params(y, top_level=True):
    nf = y.grad_fn.next_functions if top_level else y.next_functions
    for f, _ in nf:
        try:
            yield f.variable
        except AttributeError:
            pass  # node has no tensor
        if f is not None:
            yield from get_contributing_params(f, top_level=False)

class GPTBigCodeConfigClassificationSeveralFunc(GPTBigCodeConfig):
    def __init__(
            self,
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=None,
            activation_function="gelu_pytorch_tanh",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            attention_softmax_in_fp32=True,
            scale_attention_softmax_in_fp32=True,
            multi_query=True,
            **kwargs,
    ):
        super().__init__(vocab_size,n_positions,n_embd,n_layer,n_head,n_inner, activation_function, resid_pdrop,
            embd_pdrop, attn_pdrop, layer_norm_epsilon, initializer_range, scale_attn_weights, use_cache,
            bos_token_id, eos_token_id, attention_softmax_in_fp32, scale_attention_softmax_in_fp32, multi_query,**kwargs)
        self.args = {}

    def set_special_params(self, args):
        self.args = vars(args)

class GPTBigCodeClassificationSeveralFunc(GPTBigCodePreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.args = Namespace(**self.config.args)
        self.num_labels = self.config.num_labels

        if self.num_labels is None:
            print("Please pass the number of labels. The number of labels is set to default")
            self.num_labels = 2

        if self.num_labels == 2:
            self.config.problem_type = "single_label_classification"
        else:
            self.config.problem_type = "multi_label_classification"

        self.transformer = GPTBigCodeModel(config)

        if self.args.top_mlp_layers_num > 0:
            top_layers = []
            for i in range(self.args.top_mlp_layers_num):
                top_layers.append((f"linear_layer_{i}", nn.Linear(self.config.n_embd, self.config.n_embd)))
                top_layers.append((f"relu_{i}", nn.GELU()))

            self.top_layers = torch.nn.Sequential(
                collections.OrderedDict(top_layers)
            )
        else:
            self.top_layers = None

        self.linear_layer = nn.Linear(self.config.n_embd, self.num_labels, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        weights = None
        
        if self.args.use_vul_weights:
            #a workaround because we can't add new parameters to the froward function because of the PEFT wrapping
            weights = inputs_embeds
            inputs_embeds = None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # (num_layers, batch_size, seq_length, hidden_size)

        if not (self.top_layers is None):
            hidden_states = self.top_layers(hidden_states)

        logits = self.linear_layer(hidden_states)
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        assert (
                self.config.pad_token_id is not None and self.args.sep_token_id is not None
        ), "Cannot handle batch sizes > 1 if no padding token is defined."


        if (input_ids is not None) :#and (tokens_pos is not None):
            input_ids_arr = input_ids.view(-1).cpu().detach().numpy()
            logits = logits.view(-1, self.num_labels)
            last_tokens_pos = []
            for i in range(input_ids_arr.shape[0]):
                if (i > 0) and (input_ids_arr[i] == self.args.sep_token_id) and (input_ids_arr[i - 1] != self.args.sep_token_id):
                    last_tokens_pos.append(i - 1)

            last_tokens_pos_tensor = torch.from_numpy(np.array(last_tokens_pos)).to(logits.device)
            pooled_logits = logits.view(-1, self.num_labels)[last_tokens_pos_tensor, :]
        else:
            raise NotImplementedError("")

        if not (weights is None):
            weights = weights.to(logits.device)
            weights_mask = torch.gt(weights, 0.0)
            weights = torch.masked_select(weights, weights_mask).view(-1)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            labels_mask = torch.ne(labels, -100)
            labels = torch.masked_select(labels, labels_mask).view(-1)

            reduction = 'none' \
                if (self.args.use_vul_weights and not(weights is None)) \
                else self.args.loss_reduction

            if self.config.problem_type == "single_label_classification":
                loss_fct = FocalLoss(reduction=reduction, gamma=self.args.focal_loss_gamma)\
                    if self.args.use_focal_loss \
                    else CrossEntropyLoss(reduction=reduction)
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss(reduction=reduction)
                loss = loss_fct(pooled_logits, labels)
            else:
                raise NotImplementedError(self.config.problem_type)

            if self.args.use_vul_weights and not(weights is None):
                assert(labels.shape == weights.shape)
                assert (loss.shape == weights.shape)
                loss = loss * weights

                if self.args.loss_reduction == 'mean':
                    loss = loss.mean()
                elif self.args.loss_reduction == 'sum':
                    loss = loss.sum()
                else:
                    raise NotImplementedError(self.args.loss_reduction)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        pooled_logits_numpy = pooled_logits.cpu().detach().numpy()
        logits_per_batch = [list() for _ in range(batch_size)]
        for i in range(pooled_logits_numpy.shape[0]):
            curr_init_batch_num = last_tokens_pos[i] // sequence_length
            logits_per_batch[curr_init_batch_num].append(pooled_logits_numpy[i, :].ravel())

        for i in range(len(logits_per_batch)):
            padding_arr = np.ones(
                (self.args.max_funcs_in_seq - len(logits_per_batch[i]), self.num_labels),
                                  dtype=pooled_logits_numpy.dtype) * -10000.
            logits_per_batch[i] = np.vstack([np.stack(logits_per_batch[i]), padding_arr])
        logits_per_batch = np.stack(logits_per_batch)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=torch.from_numpy(logits_per_batch).to(logits.device),
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
