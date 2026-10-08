from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SASRecConfig:
    item_num: int
    maxlen: int = 100
    hidden_units: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    dropout_rate: float = 0.2
    norm_first: bool = True


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.dropout1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        return outputs.transpose(-1, -2)


class SASRec(torch.nn.Module):
    """Self-attentive sequential recommender adapted from SASRec.pytorch."""

    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        if config.hidden_units % config.num_heads != 0:
            raise ValueError("hidden_units must be divisible by num_heads")
        self.config = config
        self.item_num = config.item_num
        self.maxlen = config.maxlen
        self.hidden_units = config.hidden_units
        self.norm_first = config.norm_first

        self.item_emb = torch.nn.Embedding(
            config.item_num + 1, config.hidden_units, padding_idx=0
        )
        self.pos_emb = torch.nn.Embedding(
            config.maxlen + 1, config.hidden_units, padding_idx=0
        )
        self.emb_dropout = torch.nn.Dropout(config.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(config.num_blocks):
            self.attention_layernorms.append(
                torch.nn.LayerNorm(config.hidden_units, eps=1e-8)
            )
            self.attention_layers.append(
                torch.nn.MultiheadAttention(
                    config.hidden_units,
                    config.num_heads,
                    config.dropout_rate,
                    batch_first=False,
                )
            )
            self.forward_layernorms.append(
                torch.nn.LayerNorm(config.hidden_units, eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(config.hidden_units, config.dropout_rate)
            )

        self.last_layernorm = torch.nn.LayerNorm(config.hidden_units, eps=1e-8)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_normal_(param)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)
            self.pos_emb.weight[0].fill_(0)

    def log2feats(self, log_seqs: torch.Tensor) -> torch.Tensor:
        device = log_seqs.device
        batch_size, seq_len = log_seqs.shape

        seqs = self.item_emb(log_seqs)
        seqs = seqs * (self.hidden_units**0.5)

        positions = torch.arange(1, seq_len + 1, device=device).unsqueeze(0)
        positions = positions.expand(batch_size, -1) * (log_seqs != 0)
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        attention_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        key_padding_mask = log_seqs == 0

        for layer_idx, attention_layer in enumerate(self.attention_layers):
            seqs_t = seqs.transpose(0, 1)
            if self.norm_first:
                attn_input = self.attention_layernorms[layer_idx](seqs_t)
                attn_output, _ = attention_layer(
                    attn_input,
                    attn_input,
                    attn_input,
                    attn_mask=attention_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                seqs = (seqs_t + attn_output).transpose(0, 1)
                seqs = seqs + self.forward_layers[layer_idx](
                    self.forward_layernorms[layer_idx](seqs)
                )
            else:
                attn_output, _ = attention_layer(
                    seqs_t,
                    seqs_t,
                    seqs_t,
                    attn_mask=attention_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                seqs = self.attention_layernorms[layer_idx](
                    seqs_t + attn_output
                ).transpose(0, 1)
                seqs = self.forward_layernorms[layer_idx](
                    seqs + self.forward_layers[layer_idx](seqs)
                )

        return self.last_layernorm(seqs)

    def encode_last(self, log_seqs: torch.Tensor) -> torch.Tensor:
        return self.log2feats(log_seqs)[:, -1, :]

    def forward(
        self,
        log_seqs: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final_feat = self.encode_last(log_seqs)
        pos_embs = self.item_emb(pos_items)
        neg_embs = self.item_emb(neg_items)

        pos_logits = (final_feat * pos_embs).sum(dim=-1)
        neg_logits = torch.einsum("bd,bnd->bn", final_feat, neg_embs)
        return pos_logits, neg_logits

    def score_items(self, log_seqs: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        final_feat = self.encode_last(log_seqs)
        item_embs = self.item_emb(item_ids)
        return final_feat @ item_embs.T
