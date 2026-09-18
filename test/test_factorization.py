"""Tests for the home-zone matrix factorisation baseline."""

import pytest
import torch
import torch_geometric as pyg

from activitygraphs.ml.baselines.factorization import HomeZoneMFBaseline
from activitygraphs.ml.losses import dense_bpr_loss

IS_HOME_IDX = 0
MAX_RECALL_K = 5

N_NODES = 12
BLOCK_SIZE = 4
# Only the first two nodes of each block house anyone, so the other rows of the visit matrix stay empty.
HOME_NODES = [0, 1, 4, 5, 8, 9]
RESIDENTS_PER_HOME = 6


def make_graphs(labels: torch.Tensor, homes: torch.Tensor) -> list[pyg.data.Data]:
    """One graph per user, with ``is_home`` in column 0, as the dataset stores it."""
    graphs = []
    for user_labels, home in zip(labels, homes, strict=True):
        x = torch.zeros(len(user_labels), 2)
        x[home, IS_HOME_IDX] = 1.0
        graph = pyg.data.Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long), y=user_labels.float().unsqueeze(-1))
        graph.distances = torch.zeros(len(user_labels), 1)
        graphs.append(graph)
    return graphs


def block_preference(seed: int = 0, visit_home: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Users who visit two nodes of their own home block, the structure a factorisation can recover.

    Args:
        seed: Seed of the visited-node draw.
        visit_home: Whether each user's home node is also marked visited.

    Returns:
        ``(labels, homes)``, ``[n_users, N_NODES]`` and ``[n_users]``.
    """
    generator = torch.Generator().manual_seed(seed)
    labels, homes = [], []

    for home in HOME_NODES:
        block_start = home - home % BLOCK_SIZE
        others = torch.tensor([n for n in range(block_start, block_start + BLOCK_SIZE) if n != home])

        for _ in range(RESIDENTS_PER_HOME):
            user_labels = torch.zeros(N_NODES)
            user_labels[others[torch.randperm(len(others), generator=generator)[:2]]] = 1.0
            user_labels[home] = float(visit_home)

            labels.append(user_labels)
            homes.append(home)

    return torch.stack(labels), torch.tensor(homes)


def block_of(node: int) -> int:
    return node // BLOCK_SIZE


def make_loaders(**kwargs) -> tuple[pyg.loader.DataLoader, pyg.loader.DataLoader]:
    """Train and validation loaders over the same block-preference users."""
    graphs = make_graphs(*block_preference(**kwargs))
    return pyg.loader.DataLoader(graphs, batch_size=6), pyg.loader.DataLoader(graphs, batch_size=6)


@pytest.fixture
def loaders():
    return make_loaders()


def fitted(loaders, **kwargs) -> HomeZoneMFBaseline:
    baseline = HomeZoneMFBaseline(
        num_nodes=N_NODES,
        is_home_idx=IS_HOME_IDX,
        max_recall_k=MAX_RECALL_K,
        n_factors=4,
        n_epochs=200,
        eval_every=20,
        **kwargs,
    )
    return baseline.fit(*loaders)


class TestDenseBPRLoss:
    def test_it_scores_the_sampled_pairs(self):
        scores = torch.tensor([[3.0, 1.0], [0.5, 2.0]])
        pos_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        neg_weight = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        loss = dense_bpr_loss(scores, pos_weight, neg_weight, n_pairs=4)
        expected = -torch.nn.functional.logsigmoid(torch.tensor([2.0, 1.5])).mean()

        assert loss == pytest.approx(float(expected), abs=1e-6)

    def test_a_row_without_a_negative_is_dropped(self):
        scores = torch.tensor([[3.0, 1.0], [0.5, 2.0]])
        pos_weight = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        neg_weight = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        loss = dense_bpr_loss(scores, pos_weight, neg_weight, n_pairs=4)
        expected = -torch.nn.functional.logsigmoid(torch.tensor(1.5))

        assert loss == pytest.approx(float(expected), abs=1e-6)

    def test_no_row_with_both_gives_a_zero_attached_to_the_scores(self):
        scores = torch.zeros(2, 2, requires_grad=True)
        loss = dense_bpr_loss(scores, torch.zeros(2, 2), torch.ones(2, 2), n_pairs=4)

        assert float(loss.detach()) == 0.0
        assert loss.requires_grad

    def test_a_zero_weight_candidate_is_never_drawn(self):
        scores = torch.tensor([[0.0, 0.0, 100.0]])
        pos_weight = torch.tensor([[1.0, 0.0, 0.0]])
        neg_weight = torch.tensor([[0.0, 1.0, 0.0]])

        loss = dense_bpr_loss(scores, pos_weight, neg_weight, n_pairs=64)

        assert loss == pytest.approx(float(-torch.nn.functional.logsigmoid(torch.tensor(0.0))), abs=1e-6)


class TestHomeZoneMFBaseline:
    def test_it_ranks_the_home_block_above_the_others(self, loaders):
        scores = fitted(loaders).scores

        for home in HOME_NODES:
            same_block = torch.tensor([block_of(n) == block_of(home) and n != home for n in range(N_NODES)])
            other_block = torch.tensor([block_of(n) != block_of(home) for n in range(N_NODES)])

            assert scores[home][same_block].min() > scores[home][other_block].max()

    def test_a_home_without_residents_scores_the_bias_alone(self, loaders):
        scores = fitted(loaders).scores
        empty = [n for n in range(N_NODES) if n not in HOME_NODES]

        rows = scores[empty]
        assert torch.allclose(rows, rows[0].expand_as(rows), atol=1e-6)

    def test_visits_to_the_home_node_do_not_change_the_fit(self):
        with_home = fitted(make_loaders(visit_home=True), seed=4).scores
        without_home = fitted(make_loaders(visit_home=False), seed=4).scores

        assert torch.equal(with_home, without_home)

    def test_forward_reads_the_score_of_each_user_home_and_node(self, loaders):
        train_loader, _ = loaders
        baseline = fitted(loaders)
        batch = next(iter(train_loader))

        out = baseline(batch.x, batch.edge_index, None, batch.batch)
        homes = batch.x[:, IS_HOME_IDX].reshape(-1, N_NODES).argmax(dim=1)
        expected = baseline.scores[homes.repeat_interleave(N_NODES), torch.arange(N_NODES).repeat(len(homes))]

        assert out.shape == (len(homes) * N_NODES, 1)
        assert torch.allclose(out.reshape(-1), expected)

    def test_it_reports_the_validation_recall_of_the_epoch_it_keeps(self, loaders):
        baseline = fitted(loaders)

        assert 0.0 < baseline.val_avg_recall <= 1.0
        assert baseline.best_epoch is not None

    def test_the_same_seed_gives_the_same_scores(self, loaders):
        assert torch.equal(fitted(loaders, seed=3).scores, fitted(loaders, seed=3).scores)

    def test_different_seeds_give_different_scores(self, loaders):
        assert not torch.equal(fitted(loaders, seed=1).scores, fitted(loaders, seed=2).scores)
