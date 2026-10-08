from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DINConfig:
    item_num: int
    maxlen: int = 100
    embedding_dim: int = 64
    hidden_units: tuple[int, ...] = (128, 64)
    attention_units: tuple[int, ...] = (64, 16)
    dropout_rate: float = 0.2
    attention_weight_normalization: bool = True


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: tuple[int, ...],
        output_dim: int | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(torch.nn.Linear(last_dim, hidden_dim))
            layers.append(torch.nn.PReLU())
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        if output_dim is not None:
            layers.append(torch.nn.Linear(last_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalActivationUnit(torch.nn.Module):
    """DeepCTR-style local activation unit used by DIN attention pooling."""

    def __init__(self, embedding_dim: int, hidden_units: tuple[int, ...]) -> None:
        super().__init__()
        self.mlp = MLP(embedding_dim * 4, hidden_units, output_dim=1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        if query.dim() == 2:
            query = query.unsqueeze(1)
        query = query.expand(-1, keys.size(1), -1)
        features = torch.cat([query, keys, query - keys, query * keys], dim=-1)
        return self.mlp(features).squeeze(-1)


class AttentionSequencePooling(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        attention_units: tuple[int, ...],
        weight_normalization: bool = True,
    ) -> None:
        super().__init__()
        self.local_att = LocalActivationUnit(embedding_dim, attention_units)
        self.weight_normalization = weight_normalization

    def forward(
        self,
        query_emb: torch.Tensor,
        hist_emb: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, maxlen, _ = hist_emb.shape
        scores = self.local_att(query_emb, hist_emb)
        time_ids = torch.arange(maxlen, device=lengths.device).unsqueeze(0)
        valid_mask = time_ids < lengths.unsqueeze(1)

        if self.weight_normalization:
            scores = scores.masked_fill(~valid_mask, -torch.inf)
            weights = torch.softmax(scores, dim=1)
            weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        else:
            weights = torch.where(valid_mask, scores, torch.zeros_like(scores))

        return torch.bmm(weights.unsqueeze(1), hist_emb).squeeze(1)


class DIN(torch.nn.Module):
    """DIN adapted for full-catalog item ranking."""

    def __init__(self, config: DINConfig) -> None:
        super().__init__()
        self.config = config
        self.item_num = config.item_num
        self.maxlen = config.maxlen
        self.embedding_dim = config.embedding_dim

        self.item_emb = torch.nn.Embedding(
            config.item_num + 1,
            config.embedding_dim,
            padding_idx=0,
        )
        self.attention = AttentionSequencePooling(
            config.embedding_dim,
            config.attention_units,
            weight_normalization=config.attention_weight_normalization,
        )
        self.dnn = MLP(
            input_dim=config.embedding_dim * 4,
            hidden_units=config.hidden_units,
            output_dim=1,
            dropout_rate=config.dropout_rate,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_normal_(param)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)

    def score_candidates(
        self,
        hist_seqs: torch.Tensor,
        lengths: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> torch.Tensor:
        hist_emb = self.item_emb(hist_seqs)
        candidate_emb = self.item_emb(candidate_items)

        if candidate_emb.dim() == 2:
            hist = self.attention(candidate_emb, hist_emb, lengths)
            dnn_input = torch.cat(
                [candidate_emb, hist, candidate_emb - hist, candidate_emb * hist],
                dim=-1,
            )
            return self.dnn(dnn_input).squeeze(-1)

        batch_size, num_candidates, embedding_dim = candidate_emb.shape
        flat_candidates = candidate_emb.reshape(batch_size * num_candidates, embedding_dim)
        expanded_hist = hist_emb.unsqueeze(1).expand(-1, num_candidates, -1, -1)
        expanded_hist = expanded_hist.reshape(batch_size * num_candidates, hist_emb.size(1), embedding_dim)
        expanded_lengths = lengths.unsqueeze(1).expand(-1, num_candidates).reshape(-1)
        hist = self.attention(flat_candidates, expanded_hist, expanded_lengths)
        dnn_input = torch.cat(
            [flat_candidates, hist, flat_candidates - hist, flat_candidates * hist],
            dim=-1,
        )
        return self.dnn(dnn_input).reshape(batch_size, num_candidates)

    def forward(
        self,
        hist_seqs: torch.Tensor,
        lengths: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_logits = self.score_candidates(hist_seqs, lengths, pos_items)
        neg_logits = self.score_candidates(hist_seqs, lengths, neg_items)
        return pos_logits, neg_logits

