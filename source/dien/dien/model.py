from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class DIENConfig:
    item_num: int
    maxlen: int = 100
    embedding_dim: int = 64
    hidden_units: tuple[int, ...] = (128, 64)
    attention_units: tuple[int, ...] = (64, 16)
    dropout_rate: float = 0.2
    aux_loss_weight: float = 0.1


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: tuple[int, ...],
        output_dim: int | None = None,
        dropout_rate: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(torch.nn.Linear(last_dim, hidden_dim))
            if activation == "prelu":
                layers.append(torch.nn.PReLU())
            else:
                layers.append(torch.nn.ReLU())
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        if output_dim is not None:
            layers.append(torch.nn.Linear(last_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalActivationUnit(torch.nn.Module):
    """DIN/DIEN-style local activation unit for query-aware sequence attention."""

    def __init__(self, embedding_dim: int, hidden_units: tuple[int, ...]) -> None:
        super().__init__()
        self.mlp = MLP(embedding_dim * 4, hidden_units, output_dim=1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        if query.dim() == 2:
            query = query.unsqueeze(1)
        query = query.expand(-1, keys.size(1), -1)
        features = torch.cat([query, keys, query - keys, query * keys], dim=-1)
        return self.mlp(features).squeeze(-1)


class AUGRUCell(torch.nn.Module):
    """AUGRU cell adapted from DeepCTR-Torch's DIEN layer."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.zeros(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight_ih)
        torch.nn.init.xavier_uniform_(self.weight_hh)

    def forward(self, inputs: torch.Tensor, hx: torch.Tensor, att_score: torch.Tensor) -> torch.Tensor:
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        update_gate = att_score.view(-1, 1) * update_gate
        return (1.0 - update_gate) * hx + update_gate * new_state


class InterestExtractor(torch.nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True,
        )
        self.aux_net = MLP(embedding_dim * 2, (100, 50), output_dim=1)

    def forward(
        self,
        hist_emb: torch.Tensor,
        lengths: torch.Tensor,
        aux_neg_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, maxlen, embedding_dim = hist_emb.shape
        interests = hist_emb.new_zeros(batch_size, maxlen, embedding_dim)
        aux_loss = hist_emb.new_zeros(())

        valid_mask = lengths > 0
        if not valid_mask.any():
            return interests, aux_loss

        valid_hist = hist_emb[valid_mask]
        valid_lengths = lengths[valid_mask]
        packed = pack_padded_sequence(
            valid_hist,
            valid_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        valid_interests, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            padding_value=0.0,
            total_length=maxlen,
        )
        interests[valid_mask] = valid_interests

        if aux_neg_emb is not None:
            aux_loss = self.auxiliary_loss(interests, hist_emb, aux_neg_emb, lengths)
        return interests, aux_loss

    def auxiliary_loss(
        self,
        interests: torch.Tensor,
        hist_emb: torch.Tensor,
        aux_neg_emb: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, maxlen, embedding_dim = interests.shape
        if maxlen <= 1:
            return interests.new_zeros(())

        states = interests[:, :-1, :]
        click_seq = hist_emb[:, 1:, :]
        noclick_seq = aux_neg_emb[:, 1:, :]
        valid_steps = torch.clamp(lengths - 1, min=0)

        time_ids = torch.arange(maxlen - 1, device=lengths.device).unsqueeze(0)
        mask = time_ids < valid_steps.unsqueeze(1)
        if not mask.any():
            return interests.new_zeros(())

        click_logits = self.aux_net(torch.cat([states, click_seq], dim=-1)).squeeze(-1)
        noclick_logits = self.aux_net(torch.cat([states, noclick_seq], dim=-1)).squeeze(-1)
        logits = torch.cat([click_logits[mask], noclick_logits[mask]], dim=0)
        labels = torch.cat(
            [torch.ones_like(click_logits[mask]), torch.zeros_like(noclick_logits[mask])],
            dim=0,
        )
        return F.binary_cross_entropy_with_logits(logits, labels)


class InterestEvolution(torch.nn.Module):
    def __init__(self, embedding_dim: int, attention_units: tuple[int, ...]) -> None:
        super().__init__()
        self.attention = LocalActivationUnit(embedding_dim, attention_units)
        self.augru = AUGRUCell(embedding_dim, embedding_dim)

    def forward(
        self,
        query_emb: torch.Tensor,
        interests: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, maxlen, embedding_dim = interests.shape
        scores = self.attention(query_emb, interests)

        time_ids = torch.arange(maxlen, device=lengths.device).unsqueeze(0)
        valid_mask = time_ids < lengths.unsqueeze(1)
        scores = scores.masked_fill(~valid_mask, -torch.inf)
        att_scores = torch.softmax(scores, dim=1)
        att_scores = torch.where(valid_mask, att_scores, torch.zeros_like(att_scores))

        h = interests.new_zeros(batch_size, embedding_dim)
        for step in range(maxlen):
            new_h = self.augru(interests[:, step, :], h, att_scores[:, step])
            h = torch.where(valid_mask[:, step].unsqueeze(1), new_h, h)
        return h


class DIEN(torch.nn.Module):
    """DIEN adapted for item ranking over Amazon review sequences."""

    def __init__(self, config: DIENConfig) -> None:
        super().__init__()
        self.config = config
        self.item_num = config.item_num
        self.maxlen = config.maxlen
        self.embedding_dim = config.embedding_dim
        self.aux_loss_weight = config.aux_loss_weight

        self.item_emb = torch.nn.Embedding(
            config.item_num + 1,
            config.embedding_dim,
            padding_idx=0,
        )
        self.interest_extractor = InterestExtractor(config.embedding_dim)
        self.interest_evolution = InterestEvolution(config.embedding_dim, config.attention_units)
        self.dnn = MLP(
            input_dim=config.embedding_dim * 3,
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

    def encode_interests(
        self,
        hist_seqs: torch.Tensor,
        lengths: torch.Tensor,
        aux_neg_seqs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hist_emb = self.item_emb(hist_seqs)
        aux_neg_emb = self.item_emb(aux_neg_seqs) if aux_neg_seqs is not None else None
        return self.interest_extractor(hist_emb, lengths, aux_neg_emb)

    def score_candidates(
        self,
        hist_seqs: torch.Tensor,
        lengths: torch.Tensor,
        candidate_items: torch.Tensor,
        aux_neg_seqs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interests, aux_loss = self.encode_interests(hist_seqs, lengths, aux_neg_seqs)
        candidate_emb = self.item_emb(candidate_items)

        if candidate_emb.dim() == 2:
            evolved = self.interest_evolution(candidate_emb, interests, lengths)
            dnn_input = torch.cat(
                [evolved, candidate_emb, evolved * candidate_emb],
                dim=-1,
            )
            logits = self.dnn(dnn_input).squeeze(-1)
            return logits, aux_loss

        batch_size, num_candidates, embedding_dim = candidate_emb.shape
        flat_candidates = candidate_emb.reshape(batch_size * num_candidates, embedding_dim)
        expanded_interests = interests.unsqueeze(1).expand(-1, num_candidates, -1, -1)
        expanded_interests = expanded_interests.reshape(
            batch_size * num_candidates,
            interests.size(1),
            embedding_dim,
        )
        expanded_lengths = lengths.unsqueeze(1).expand(-1, num_candidates).reshape(-1)
        evolved = self.interest_evolution(flat_candidates, expanded_interests, expanded_lengths)
        dnn_input = torch.cat([evolved, flat_candidates, evolved * flat_candidates], dim=-1)
        logits = self.dnn(dnn_input).reshape(batch_size, num_candidates)
        return logits, aux_loss

    def forward(
        self,
        hist_seqs: torch.Tensor,
        lengths: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        aux_neg_seqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_logits, aux_loss = self.score_candidates(
            hist_seqs,
            lengths,
            pos_items,
            aux_neg_seqs=aux_neg_seqs,
        )
        neg_logits, _ = self.score_candidates(
            hist_seqs,
            lengths,
            neg_items,
            aux_neg_seqs=None,
        )
        return pos_logits, neg_logits, aux_loss

