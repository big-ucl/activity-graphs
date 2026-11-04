"""Tests for the home-zone matrix factorisation baseline."""

import lightning as L
import pytest
import torch
from graphs import edgeless_locations, join_users, loader_of, make_users

from activitygraphs.config import LossConfig
from activitygraphs.datasets import Locations
from activitygraphs.models.baselines.factorization import HomeZoneMFBaseline
from activitygraphs.training.lightning_module import RankingModule
from activitygraphs.training.losses import build_loss

MAX_RECALL_K = 5
N_FACTORS = 4
LEARNING_RATE = 5e-2
N_EPOCHS = 60

N_NODES = 12
BLOCK_SIZE = 4
# Only the first two nodes of each block house anyone, so the other rows of the visit matrix stay empty.
HOME_NODES = [0, 1, 4, 5, 8, 9]
RESIDENTS_PER_HOME = 6


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


def trained(visit_home: bool = True, seed: int = 0) -> tuple[HomeZoneMFBaseline, Locations]:
    """Return a ``HomeZoneMFBaseline`` trained by ``RankingModule`` on the block-preference users, and its locations."""
    torch.manual_seed(seed)
    users = make_users(*block_preference(visit_home=visit_home))
    locations = edgeless_locations(N_NODES)
    scorer = HomeZoneMFBaseline(N_NODES, N_FACTORS)
    loss = build_loss(LossConfig(), MAX_RECALL_K)
    module = RankingModule(
        scorer=scorer,
        locations=locations,
        lr=LEARNING_RATE,
        loss=loss,
        max_recall_k=MAX_RECALL_K,
        recall_ks=[1, MAX_RECALL_K],
    )
    loader = loader_of(users, batch_size=12)
    trainer = L.Trainer(
        max_epochs=N_EPOCHS,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    return scorer, locations


def scores_of(scorer: HomeZoneMFBaseline) -> torch.Tensor:
    """Return the score of every node for a user living at every node, ``[N_NODES, N_NODES]``, indexed by (home, node)."""
    with torch.no_grad():
        return scorer.home_factors @ scorer.node_factors.T + scorer.bias


@pytest.fixture(scope="module")
def fitted() -> HomeZoneMFBaseline:
    scorer, _ = trained()
    return scorer


class TestHomeZoneMFBaseline:
    def test_the_home_factors_start_at_zero(self):
        scorer = HomeZoneMFBaseline(N_NODES, N_FACTORS)

        assert torch.equal(scorer.home_factors, torch.zeros(N_NODES, N_FACTORS))
        assert scorer.node_factors.abs().sum() > 0

    def test_it_ranks_the_home_block_above_the_others(self, fitted):
        scores = scores_of(fitted)

        for home in HOME_NODES:
            same_block = torch.tensor([block_of(n) == block_of(home) and n != home for n in range(N_NODES)])
            other_block = torch.tensor([block_of(n) != block_of(home) for n in range(N_NODES)])

            assert scores[home][same_block].min() > scores[home][other_block].max()

    def test_a_home_without_residents_scores_the_bias_alone(self, fitted):
        empty = [n for n in range(N_NODES) if n not in HOME_NODES]

        assert torch.equal(fitted.home_factors[empty], torch.zeros(len(empty), N_FACTORS))
        torch.testing.assert_close(scores_of(fitted)[empty], fitted.bias.detach().expand(len(empty), -1))

    def test_forward_reads_the_score_of_each_user_home_and_node(self, fitted):
        users = make_users(*block_preference()).subset(list(range(6)))
        locations = edgeless_locations(N_NODES)
        batch = join_users(locations, users)

        with torch.no_grad():
            out = fitted(locations, batch)

        assert out.shape == (6, N_NODES)
        torch.testing.assert_close(out, scores_of(fitted)[users.home_node])
