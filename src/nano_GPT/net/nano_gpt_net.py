# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/14 20:19
import torch
from torch import nn
from torch.nn import functional as F

from src.nano_GPT.config import model_config
from src.nano_GPT.config.model_config import ModelConfig
from src.nano_GPT.net.layers.gpt_block import GPTBlock


class NanoGPTNet(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(NanoGPTNet, self).__init__()
        self._word_embedding_table = nn.Embedding(model_config.vocab_size, model_config.embed_size)
        self._position_embedding_table = nn.Embedding(model_config.seq_len, model_config.embed_size)
        self.gpt_blocks = nn.Sequential(*[GPTBlock(model_config) for _ in range(model_config.n_block)])
        self.ln_final = nn.LayerNorm(model_config.embed_size)
        self.lm_head = nn.Linear(model_config.embed_size, model_config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def forward(self, x, targets=None):
        loss = None
        batch_size, seq_len = x.shape
        word_embed = self._word_embedding_table(x)
        position_embed = self._position_embedding_table(torch.arange(seq_len, device=x.device))
        input_embed = word_embed + position_embed
        attention = self.gpt_blocks(input_embed)
        attention = self.ln_final(attention)
        logits = self.lm_head(attention)
        if targets is not None:
            batch_size, seq_len, vocab_size = logits.size()
            loss = F.cross_entropy(logits.view(batch_size*seq_len, -1), targets.view(batch_size*seq_len))
        return logits, loss

    def generate(self, word_idx: torch.Tensor, max_new_token_len: int=200):
        cur_context = word_idx
        for _ in range(max_new_token_len):
            cur_previous_context = cur_context[:, :self.seq_len]
            logits, _ = self(cur_context)
            cur_word_logits = logits[:, -1, :]
            cur_word_probs = F.softmax(cur_word_logits, dim=-1)
            cur_word_idx = torch.multinomial(cur_word_probs, 1)
            cur_context = torch.cat([cur_word_idx, cur_previous_context], dim=1)
        return cur_context

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        else:
            pass
