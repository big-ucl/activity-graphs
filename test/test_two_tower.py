"""Tests for the two-tower scorer and its builder."""

import itertools

import lightning as L
import pytest
import torch
from encoders import continuous_schema
from graphs import chain_edge_index, join_users, loader_of, make_locations, make_users
from omegaconf import OmegaConf

from activitygraphs.config import HomePEConfig, LossConfig, read_config
from activitygraphs.datasets import Locations, Users
from activitygraphs.models.architectures import build_scorer
from activitygraphs.models.features import InputFeatures
from activitygraphs.models.two_tower import LocationBias, NodeBias, TwoTowerScorer
from activitygraphs.schema import DatasetSchema
from activitygraphs.tensors.build import USER_NODE_SCHEMA
from activitygraphs.training.lightning_module import RankingModule
from activitygraphs.training.losses import build_loss
from activitygraphs.utils import get_project_root

NUM_NODES = 6
NUM_USERS = 4
NUM_LOCATION_FEATURES = 3
NUM_EDGE_FEATURES = 2
NUM_RWPE_FEATURES = 4
NUM_ATTRIBUTES = 2
HOME_PE = HomePEConfig(bins=4, max_hops=5)
FLAGS = ["full_info", "use_home_pe", "use_demographics", "use_rwpe"]


@pytest.fixture(scope="module")
def config():
    two_tower = read_config(get_project_root(), verbose=False).models.two_tower
    two_tower.num_gcn_layers = 2
    two_tower.hidden_channels = 16
    two_tower.embedding_dim = 8
    return two_tower


def make_schema() -> DatasetSchema:
    """The schema of a survey with continuous location, edge and user attribute columns."""
    return DatasetSchema(
        locations=continuous_schema("locations", NUM_LOCATION_FEATURES),
        edges=continuous_schema("edges", NUM_EDGE_FEATURES),
        user_nodes=USER_NODE_SCHEMA,
        user_attributes=continuous_schema("user_attributes", NUM_ATTRIBUTES),
        distances=continuous_schema("distances", 1),
        targets=continuous_schema("targets", 1),
    )


def make_survey() -> tuple[Locations, Users]:
    """A chain of ``NUM_NODES`` nodes with random features and ``NUM_USERS`` users with random visits."""
    generator = torch.Generator().manual_seed(0)
    edge_index = chain_edge_index(NUM_NODES)
    x = torch.rand(NUM_NODES, NUM_LOCATION_FEATURES, generator=generator)
    edge_attr = torch.rand(edge_index.shape[1], NUM_EDGE_FEATURES, generator=generator)
    random_walk_pe = torch.rand(NUM_NODES, NUM_RWPE_FEATURES, generator=generator)
    distance_m = torch.rand(NUM_NODES, NUM_NODES, generator=generator) * 1000.0
    locations = make_locations(x, edge_index, edge_attr, random_walk_pe, distance_m)

    visited = (torch.rand(NUM_USERS, NUM_NODES, generator=generator) < 0.5).float()
    home_node = torch.tensor([0, 2, 2, 5])
    attributes = torch.rand(NUM_USERS, NUM_ATTRIBUTES, generator=generator)

    return locations, make_users(visited, home_node, attributes)


def scorer_of(config, features: InputFeatures, locations: Locations) -> TwoTowerScorer:
    scorer = build_scorer(config, features, locations, make_schema(), HOME_PE)
    assert isinstance(scorer, TwoTowerScorer)
    return scorer


ALL_FEATURES = [
    InputFeatures(**dict(zip(FLAGS, active, strict=True))) for active in itertools.product([False, True], repeat=4)
]


class TestTwoTowerScorer:
    @pytest.mark.parametrize("features", ALL_FEATURES, ids=str)
    def test_every_feature_combination_scores_every_node_of_every_user(self, config, features):
        locations, users = make_survey()
        scorer = scorer_of(config, features, locations)

        out = scorer(locations, join_users(locations, users))

        assert out.shape == (NUM_USERS, NUM_NODES)
        assert torch.isfinite(out).all()

    def test_the_scores_of_a_user_do_not_depend_on_the_other_users_of_the_batch(self, config):
        locations, users = make_survey()
        scorer = scorer_of(config, InputFeatures(full_info=True, use_home_pe=True), locations).eval()

        with torch.no_grad():
            together = scorer(locations, join_users(locations, users))
            alone = scorer(locations, join_users(locations, users.subset([1])))

        torch.testing.assert_close(alone[0], together[1])

    def test_without_the_pairwise_block_and_the_bias_the_logits_are_the_dot_product(self, config):
        locations, users = make_survey()
        scorer = scorer_of(config, InputFeatures(), locations).eval()
        last_pair_layer = scorer.pair_network.lins[-1]
        torch.nn.init.zeros_(last_pair_layer.weight)
        torch.nn.init.zeros_(last_pair_layer.bias)
        torch.nn.init.zeros_(scorer.bias.linear.weight)
        torch.nn.init.zeros_(scorer.bias.linear.bias)
        batch = join_users(locations, users)

        with torch.no_grad():
            out = scorer(locations, batch)
            node_embedding = scorer.node_tower(locations)
            user_embedding = scorer.user_tower(batch, node_embedding[batch.home_node])

        torch.testing.assert_close(out, user_embedding @ node_embedding.T)

    def test_the_input_widths_follow_the_features(self, config):
        locations, _ = make_survey()
        embedding_dim = config.embedding_dim
        bare = scorer_of(config, InputFeatures(use_demographics=False, use_rwpe=False), locations)
        full = scorer_of(config, InputFeatures(full_info=True, use_home_pe=True), locations)

        assert bare.node_tower.network.pre_lin.lins[0].in_features == NUM_LOCATION_FEATURES
        assert full.node_tower.network.pre_lin.lins[0].in_features == NUM_LOCATION_FEATURES + NUM_RWPE_FEATURES
        assert bare.user_tower.network.lins[0].in_features == embedding_dim
        assert full.user_tower.network.lins[0].in_features == embedding_dim + NUM_ATTRIBUTES
        assert bare.pair_encoder is None
        assert bare.pair_network.lins[0].in_features == 2 * embedding_dim
        assert full.pair_network.lins[0].in_features == 2 * embedding_dim + HOME_PE.bins + 1

    def test_it_trains_through_the_ranking_module(self, config):
        locations, users = make_survey()
        scorer = scorer_of(config, InputFeatures(full_info=True, use_home_pe=True), locations)
        module = RankingModule(
            scorer=scorer,
            locations=locations,
            lr=1e-2,
            loss=build_loss(LossConfig(n_pairs=8), max_recall_k=3),
            max_recall_k=3,
            recall_ks=[1, 3],
        )
        before = [parameter.detach().clone() for parameter in scorer.parameters()]
        loader = loader_of(users, batch_size=2)
        trainer = L.Trainer(
            max_epochs=2,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

        after = list(scorer.parameters())
        assert any(not torch.equal(old, new.detach()) for old, new in zip(before, after, strict=True))


class TestInCityDiagnostics:
    def test_the_default_bias_is_the_location_bias(self, config):
        locations, _ = make_survey()

        assert isinstance(scorer_of(config, InputFeatures(), locations).bias, LocationBias)

    def test_the_node_bias_is_a_free_parameter_per_node_starting_at_zero(self, config):
        locations, users = make_survey()
        node_bias = OmegaConf.merge(config, {"bias": "node"})
        scorer = scorer_of(node_bias, InputFeatures(), locations)

        assert isinstance(scorer.bias, NodeBias)
        assert torch.equal(scorer.bias(torch.ones(NUM_NODES, config.embedding_dim)), torch.zeros(NUM_NODES))
        assert scorer(locations, join_users(locations, users)).shape == (NUM_USERS, NUM_NODES)

    def test_an_unknown_bias_raises(self, config):
        locations, _ = make_survey()
        unknown = OmegaConf.merge(config, {"bias": "nope"})

        with pytest.raises(ValueError, match="nope"):
            scorer_of(unknown, InputFeatures(), locations)

    def test_the_node_embedding_widens_the_node_tower_input(self, config):
        locations, users = make_survey()
        features = InputFeatures(use_rwpe=False)
        plain = scorer_of(config, features, locations)
        embedded = scorer_of(OmegaConf.merge(config, {"node_embedding": True}), features, locations)

        assert plain.node_tower.node_embedding is None
        assert embedded.node_tower.node_embedding is not None
        assert embedded.node_tower.node_embedding.weight.shape == (NUM_NODES, config.embedding_dim)
        width = embedded.node_tower.network.pre_lin.lins[0].in_features
        assert width == NUM_LOCATION_FEATURES + config.embedding_dim
        assert embedded(locations, join_users(locations, users)).shape == (NUM_USERS, NUM_NODES)

    @pytest.mark.parametrize(
        ("overrides", "name"),
        [
            ({}, "TwoTower-2"),
            ({"bias": "node"}, "TwoTower-2-nodebias"),
            ({"node_embedding": True}, "TwoTower-2-nodeemb"),
            ({"bias": "node", "node_embedding": True}, "TwoTower-2-nodebias-nodeemb"),
        ],
    )
    def test_the_name_records_the_diagnostics(self, config, overrides, name):
        assert OmegaConf.merge(config, overrides).name == name
