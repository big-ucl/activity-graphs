"""Encoder of the columns of a tensor group, driven by the group's schema."""

from collections.abc import Callable

import torch
from torch import Tensor

from activitygraphs.schema import TensorSchema


def default_emb_dim(cardinality: int) -> int:
    """Return the embedding width of a categorical column with ``cardinality`` categories."""
    return min(16, (cardinality + 1) // 2)


class SchemaEncoder(torch.nn.Module):
    """Encodes the columns of a tensor group, with one learned embedding per categorical column.

    The encoder outputs the numeric columns, unchanged, then the embedding of each categorical column, in column order.

    Attributes:
        embeddings: Embedding of each categorical column, in column order.
        out_dim: Width of the output, the number of numeric columns plus the sum of the embedding widths.
    """

    def __init__(self, schema: TensorSchema, emb_dim: Callable[[int], int] = default_emb_dim):
        """Build one embedding per categorical column of ``schema``.

        Args:
            schema: Schema of the tensor group.
            emb_dim: Returns the embedding width of a categorical column, given its cardinality.
        """
        super().__init__()
        self.embeddings = torch.nn.ModuleList()
        for spec in schema.categorical.columns:
            width = emb_dim(spec.cardinality)
            self.embeddings.append(torch.nn.Embedding(spec.cardinality, width))

        embedded_width = sum(embedding.embedding_dim for embedding in self.embeddings)
        self.out_dim = schema.numeric.width + embedded_width

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        """Return the encoded columns, ``[n_rows, out_dim]``.

        Args:
            x_num: Numeric columns, ``[n_rows, n_numeric]``.
            x_cat: Codes of the categorical columns, int64, ``[n_rows, n_categorical]``.
        """
        encoded = [x_num]
        for column, embedding in enumerate(self.embeddings):
            encoded.append(embedding(x_cat[:, column]))

        return torch.cat(encoded, dim=-1)
