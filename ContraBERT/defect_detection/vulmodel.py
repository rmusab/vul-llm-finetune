# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class VulModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(VulModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        if args.max_pool:
            self.linear_max = nn.Linear(config.hidden_size, config.hidden_size)
            self.linear_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, labels=None):
        if self.args.max_pool:
            outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1))[0]
            outputs = self.maxpool(outputs, input_ids.ne(1))
            logits = self.linear_proj(outputs)
        else:
            logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

    def maxpool(self, states, mask=None):
        mask = mask.unsqueeze(-1)
        node_state = states * mask.float()
        embedding_p = self.linear_max(node_state).transpose(-1, -2)
        maxpool_embedding = F.max_pool1d(embedding_p, kernel_size=embedding_p.size(-1)).squeeze(-1)
        return maxpool_embedding
