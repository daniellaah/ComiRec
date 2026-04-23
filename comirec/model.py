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


class InBatchSoftmaxLoss(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        model: ComiRecSA,
        user_embeddings: torch.Tensor,
        positive_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        item_embeddings = model.item_embeddings(positive_item_ids)
        item_bias = model.item_bias[positive_item_ids]
        logits = (user_embeddings @ item_embeddings.T + item_bias.unsqueeze(0)) / self.temperature

        duplicate_targets = positive_item_ids.unsqueeze(0) == positive_item_ids.unsqueeze(1)
        diagonal = torch.eye(
            positive_item_ids.size(0),
            dtype=torch.bool,
            device=positive_item_ids.device,
        )
        logits = logits.masked_fill(
            duplicate_targets & ~diagonal,
            torch.finfo(logits.dtype).min,
        )

        labels = torch.arange(user_embeddings.size(0), device=user_embeddings.device)
        return F.cross_entropy(logits, labels)


def build_model(num_items: int, maxlen: int, model_config: ModelConfig) -> ComiRecSA:
    return ComiRecSA(
        num_items=num_items,
        embedding_dim=model_config.embedding_dim,
        hidden_size=model_config.hidden_size,
        num_interests=model_config.num_interests,
        maxlen=maxlen,
        padding_idx=model_config.padding_idx,
    )
