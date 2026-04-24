from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .configs import ModelConfig


class ComiRecSA(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_size: int,
        num_interests: int,
        maxlen: int,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.num_interests = num_interests
        self.padding_idx = padding_idx

        self.item_embeddings = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.register_buffer("item_bias", torch.zeros(num_items))
        self.position_embedding = nn.Parameter(torch.empty(1, maxlen, embedding_dim))
        self.attention_hidden = nn.Linear(embedding_dim, hidden_size * 4)
        self.attention_projection = nn.Linear(hidden_size * 4, num_interests)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        nn.init.xavier_uniform_(self.position_embedding)
        nn.init.xavier_uniform_(self.attention_hidden.weight)
        nn.init.zeros_(self.attention_hidden.bias)
        nn.init.xavier_uniform_(self.attention_projection.weight)
        nn.init.zeros_(self.attention_projection.bias)
        with torch.no_grad():
            self.item_embeddings.weight[self.padding_idx].zero_()

    def get_user_interest_embeddings(
        self,
        history_items: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        item_embeddings = self.item_embeddings(history_items)
        attention_input = item_embeddings + self.position_embedding[:, : history_items.size(1), :]
        attention_hidden = torch.tanh(self.attention_hidden(attention_input))
        attention_logits = self.attention_projection(attention_hidden).transpose(1, 2)
        attention_logits = attention_logits.masked_fill(
            (history_mask <= 0).unsqueeze(1),
            torch.finfo(attention_logits.dtype).min,
        )
        attention_weights = torch.softmax(attention_logits, dim=-1)
        return attention_weights @ item_embeddings

    def encode_history(
        self,
        history_items: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.get_user_interest_embeddings(history_items, history_mask)

    def select_matching_interest(
        self,
        interest_embeddings: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        target_embeddings = self.item_embeddings(target_item_ids)
        interest_scores = (interest_embeddings * target_embeddings.unsqueeze(1)).sum(dim=-1)
        selected_interest_idx = torch.argmax(interest_scores, dim=1)
        batch_idx = torch.arange(interest_embeddings.size(0), device=interest_embeddings.device)
        return interest_embeddings[batch_idx, selected_interest_idx]

    def get_training_user_embeddings(
        self,
        history_items: torch.Tensor,
        history_mask: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        interest_embeddings = self.encode_history(history_items, history_mask)
        return self.select_matching_interest(interest_embeddings, target_item_ids)

    def forward(
        self,
        history_items: torch.Tensor,
        history_mask: torch.Tensor,
        target_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        interest_embeddings = self.encode_history(history_items, history_mask)
        if target_item_ids is None:
            return interest_embeddings
        return self.select_matching_interest(interest_embeddings, target_item_ids)

    def score_all_items(self, interest_embeddings: torch.Tensor) -> torch.Tensor:
        return interest_embeddings @ self.item_embeddings.weight.T + self.item_bias.view(1, 1, -1)


class SampledSoftmaxLoss(nn.Module):
    def __init__(self, num_items: int, num_sampled: int = 10) -> None:
        super().__init__()
        if num_sampled <= 0:
            raise ValueError("num_sampled must be positive")
        self.num_items = num_items
        self.num_sampled = num_sampled

        class_ids = torch.arange(num_items, dtype=torch.float32)
        normalizer = torch.log(torch.tensor(float(num_items + 1)))
        sampling_probs = (torch.log(class_ids + 2.0) - torch.log(class_ids + 1.0)) / normalizer
        self.register_buffer("sampling_probs", sampling_probs)
        self.register_buffer("log_range", normalizer)

    def _expected_counts(
        self,
        class_probabilities: torch.Tensor,
        *,
        batch_size: int,
        num_sampled: int,
    ) -> torch.Tensor:
        if num_sampled == batch_size:
            expected = class_probabilities * batch_size
            return expected.clamp_min(torch.finfo(expected.dtype).tiny)
        expected = 1.0 - torch.pow(1.0 - class_probabilities, num_sampled)
        return expected.clamp_min(torch.finfo(expected.dtype).tiny)

    def _sample_unique_candidate_ids(
        self,
        *,
        device: torch.device,
        num_sampled: int,
    ) -> tuple[torch.Tensor, int]:
        if num_sampled > self.num_items:
            raise ValueError(
                f"num_sampled ({num_sampled}) must be <= num_items ({self.num_items}) for unique sampling"
            )

        sampled_ids: list[int] = []
        seen_ids: set[int] = set()
        num_tries = 0
        log_range = self.log_range.to(device)

        while len(sampled_ids) < num_sampled:
            remaining = num_sampled - len(sampled_ids)
            chunk_size = max(remaining * 2, 32)
            uniforms = torch.rand(chunk_size, device=device)
            candidates = torch.floor(torch.exp(uniforms * log_range)).to(torch.long) - 1
            candidates = torch.remainder(candidates, self.num_items)

            for candidate_id in candidates.tolist():
                num_tries += 1
                if candidate_id in seen_ids:
                    continue
                seen_ids.add(candidate_id)
                sampled_ids.append(candidate_id)
                if len(sampled_ids) == num_sampled:
                    break

        return torch.tensor(sampled_ids, dtype=torch.long, device=device), num_tries

    def forward(
        self,
        model: ComiRecSA,
        user_embeddings: torch.Tensor,
        positive_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_sampled = self.num_sampled * user_embeddings.size(0)
        sampling_probs = self.sampling_probs.to(user_embeddings.device)
        sampled_item_ids, num_tries = self._sample_unique_candidate_ids(
            device=user_embeddings.device,
            num_sampled=num_sampled,
        )

        positive_embeddings = model.item_embeddings(positive_item_ids)
        positive_bias = model.item_bias[positive_item_ids]
        positive_logits = (
            (user_embeddings * positive_embeddings).sum(dim=-1, keepdim=True)
            + positive_bias.unsqueeze(1)
        )

        sampled_embeddings = model.item_embeddings(sampled_item_ids)
        sampled_bias = model.item_bias[sampled_item_ids]
        sampled_logits = user_embeddings @ sampled_embeddings.T + sampled_bias.unsqueeze(0)

        accidental_hits = positive_item_ids.unsqueeze(1) == sampled_item_ids.unsqueeze(0)
        sampled_logits = sampled_logits.masked_fill(
            accidental_hits,
            -torch.finfo(sampled_logits.dtype).max,
        )

        positive_expected = self._expected_counts(
            sampling_probs[positive_item_ids],
            batch_size=num_sampled,
            num_sampled=num_tries,
        ).unsqueeze(1)
        sampled_expected = self._expected_counts(
            sampling_probs[sampled_item_ids],
            batch_size=num_sampled,
            num_sampled=num_tries,
        ).unsqueeze(0)

        positive_logits = positive_logits - positive_expected.log()
        sampled_logits = sampled_logits - sampled_expected.log()

        logits = torch.cat([positive_logits, sampled_logits], dim=1)
        labels = torch.zeros(user_embeddings.size(0), dtype=torch.long, device=user_embeddings.device)
        return F.cross_entropy(logits, labels)


def build_model(
    num_items: int,
    maxlen: int,
    model_config: ModelConfig,
) -> ComiRecSA:
    return ComiRecSA(
        num_items=num_items,
        embedding_dim=model_config.embedding_dim,
        hidden_size=model_config.hidden_size,
        num_interests=model_config.num_interests,
        maxlen=maxlen,
        padding_idx=model_config.padding_idx,
    )
