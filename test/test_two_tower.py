"""Tests for the two-tower scorer and its builder."""

import dataclasses
import itertools

import lightning as L
import pytest
import torch
from encoders import continuous_schema
from graphs import chain_edge_index, join_users, loader_of, make_locations, make_users
from omegaconf import OmegaConf

from activitygraphs.config import HomePEConfig, LossConfig, read_config
from activitygraphs.datasets import Locations, Users
from activitygraphs.experiments.variants import Variant
from activitygraphs.models.architectures import build_scorer
from activitygraphs.models.features import InputFeatures
from activitygraphs.models.two_tower import LocationBias, MLPBias, NodeBias, TwoTowerScorer
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
NUM_ACCESS_FEATURES = 2
NUM_ATTRIBUTES = 2
NUM_TRAVEL_TIME_FEATURES = 2
HOME_PE = HomePEConfig(bins=4, max_hops=5)
FLAGS = ["use_dist", "use_home_pe", "use_demographics", "use_rwpe", "use_access", "use_network_hops", "use_tt"]


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
        access=continuous_schema("access", NUM_ACCESS_FEATURES),
        contiguity_edges=continuous_schema("contiguity_edges", NUM_EDGE_FEATURES),
        rail_edges=continuous_schema("rail_edges", 0),
        user_nodes=USER_NODE_SCHEMA,
        user_attributes=continuous_schema("user_attributes", NUM_ATTRIBUTES),
        distances=continuous_schema("distances", 1),
        travel_times=continuous_schema("travel_times", NUM_TRAVEL_TIME_FEATURES),
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

    visited = (torch.rand(NUM_USERS, NUM_NODES, generator=generator) < 0.5).float()
    home_node = torch.tensor([0, 2, 2, 5])
    attributes = torch.rand(NUM_USERS, NUM_ATTRIBUTES, generator=generator)
    access = torch.rand(NUM_NODES, NUM_ACCESS_FEATURES, generator=generator)
    travel_times = torch.rand(NUM_NODES, NUM_NODES, NUM_TRAVEL_TIME_FEATURES, generator=generator)
    locations = make_locations(
        x, edge_index, edge_attr, random_walk_pe, distance_m, access, travel_times_min=travel_times
    )

    return locations, make_users(visited, home_node, attributes)


def scorer_of(config, features: InputFeatures, locations: Locations) -> TwoTowerScorer:
    scorer = build_scorer(config, features, locations, make_schema(), HOME_PE)
    assert isinstance(scorer, TwoTowerScorer)
    return scorer


ALL_FEATURES = [
    InputFeatures(**dict(zip(FLAGS, active, strict=True)))
    for active in itertools.product([False, True], repeat=len(FLAGS))
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
        scorer = scorer_of(config, InputFeatures(use_dist=True, use_home_pe=True), locations).eval()

        with torch.no_grad():
            together = scorer(locations, join_users(locations, users))
            alone = scorer(locations, join_users(locations, users.subset([1])))

        torch.testing.assert_close(alone[0], together[1])

    def test_without_the_pairwise_block_and_the_bias_the_logits_are_the_dot_product(self, config):
        locations, users = make_survey()
        scorer = scorer_of(config, InputFeatures(use_home_pe=False, use_dist=False), locations).eval()
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
        schema = make_schema()
        bare_features = InputFeatures(
            use_home_pe=False, use_dist=False, use_demographics=False, use_rwpe=False, use_access=False, use_tt=False
        )
        bare = scorer_of(config, bare_features, locations)
        full = scorer_of(config, InputFeatures(use_dist=True, use_home_pe=True), locations)
        bare_node_width = bare.node_tower.node_encoder.out_dim
        full_node_width = full.node_tower.node_encoder.out_dim

        assert bare_node_width == schema.locations.width
        assert full_node_width - bare_node_width == schema.access.width + locations.num_rwpe_features
        assert bare.pair_encoder is None
        assert full.pair_encoder is not None
        assert full.pair_encoder.out_dim == HOME_PE.bins + schema.distances.width + schema.travel_times.width

    @pytest.mark.parametrize("use_demographics", [False, True])
    def test_the_user_attributes_change_the_scores_only_with_the_demographics(self, config, use_demographics):
        locations, users = make_survey()
        scorer = scorer_of(config, InputFeatures(use_demographics=use_demographics), locations).eval()
        shifted_users = dataclasses.replace(users, attributes=users.attributes + 1.0)

        with torch.no_grad():
            scores = scorer(locations, join_users(locations, users))
            shifted_scores = scorer(locations, join_users(locations, shifted_users))

        assert torch.equal(scores, shifted_scores) is not use_demographics

    def test_it_trains_through_the_ranking_module(self, config):
        locations, users = make_survey()
        scorer = scorer_of(config, InputFeatures(use_dist=True, use_home_pe=True), locations)
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
        assert torch.equal(scorer.bias(locations, torch.ones(NUM_NODES, config.embedding_dim)), torch.zeros(NUM_NODES))
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
        added_width = embedded.node_tower.node_encoder.out_dim - plain.node_tower.node_encoder.out_dim

        assert added_width == config.embedding_dim
        assert embedded(locations, join_users(locations, users)).shape == (NUM_USERS, NUM_NODES)

    @pytest.mark.parametrize(
        ("overrides", "name"),
        [
            ({}, "TwoTower-4"),
            ({"bias": "node"}, "TwoTower-4-nodebias"),
            ({"bias": "embedding_mlp"}, "TwoTower-4-mlpbias"),
            ({"bias": "feature_mlp"}, "TwoTower-4-featbias"),
            ({"pair_location_features": True}, "TwoTower-4-pairloc"),
            ({"node_embedding": True}, "TwoTower-4-nodeemb"),
            ({"bias": "node", "node_embedding": True}, "TwoTower-4-nodebias-nodeemb"),
        ],
    )
    def test_the_name_records_the_diagnostics(self, overrides, name):
        two_tower = read_config(get_project_root(), verbose=False).models.two_tower

        assert Variant(OmegaConf.merge(two_tower, overrides)).name == name


class TestMLPBias:
    @pytest.mark.parametrize("bias", ["embedding_mlp", "feature_mlp"])
    def test_scores_every_node_of_every_user(self, config, bias):
        locations, users = make_survey()
        scorer = scorer_of(OmegaConf.merge(config, {"bias": bias}), InputFeatures(use_dist=True), locations)

        assert isinstance(scorer.bias, MLPBias)
        assert scorer(locations, join_users(locations, users)).shape == (NUM_USERS, NUM_NODES)

    def test_the_embedding_head_reads_the_node_embedding(self, config):
        locations, _ = make_survey()
        scorer = scorer_of(OmegaConf.merge(config, {"bias": "embedding_mlp"}), InputFeatures(), locations).eval()
        node_embedding = torch.ones(NUM_NODES, config.embedding_dim)
        num_linear_layers = sum(isinstance(module, torch.nn.Linear) for module in scorer.bias.modules())

        with torch.no_grad():
            bias = scorer.bias(locations, node_embedding)
            zero_embedding_bias = scorer.bias(locations, torch.zeros_like(node_embedding))

        assert bias.shape == (NUM_NODES,)
        assert not torch.equal(bias, zero_embedding_bias)
        assert num_linear_layers == config.num_bias_layers

    def test_the_feature_head_projects_the_location_features_alone(self, config):
        locations, _ = make_survey()
        scorer = scorer_of(OmegaConf.merge(config, {"bias": "feature_mlp"}), InputFeatures(), locations).eval()
        node_embedding = torch.ones(NUM_NODES, config.embedding_dim)
        shifted_locations = dataclasses.replace(locations, x=locations.x + 1.0)

        with torch.no_grad():
            bias = scorer.bias(locations, node_embedding)
            zero_embedding_bias = scorer.bias(locations, torch.zeros_like(node_embedding))
            shifted_bias = scorer.bias(shifted_locations, node_embedding)

        torch.testing.assert_close(bias, zero_embedding_bias)
        assert not torch.equal(bias, shifted_bias)


class TestPairLocationFeatures:
    def test_widen_the_pairwise_block_by_the_embedding_width(self, config):
        locations, users = make_survey()
        features = InputFeatures(use_dist=True, use_home_pe=True)
        plain = scorer_of(config, features, locations)
        widened = scorer_of(OmegaConf.merge(config, {"pair_location_features": True}), features, locations)

        assert plain.pair_encoder is not None and widened.pair_encoder is not None
        assert widened.pair_encoder.out_dim - plain.pair_encoder.out_dim == config.embedding_dim
        assert widened(locations, join_users(locations, users)).shape == (NUM_USERS, NUM_NODES)

    def test_project_the_location_features_through_a_layer_of_their_own(self, config):
        locations, users = make_survey()
        features = InputFeatures(use_home_pe=False, use_dist=False, use_tt=False)
        scorer = scorer_of(OmegaConf.merge(config, {"pair_location_features": True}), features, locations)
        batch = join_users(locations, users)

        shifted_locations = dataclasses.replace(locations, x=locations.x + 1.0)

        assert scorer.pair_encoder is not None
        with torch.no_grad():
            projected = scorer.pair_encoder(locations, batch)
            shifted = scorer.pair_encoder(shifted_locations, batch)

        assert projected.shape == (NUM_USERS, NUM_NODES, config.embedding_dim)
        for user in range(1, NUM_USERS):
            torch.testing.assert_close(projected[user], projected[0])
        assert not torch.equal(projected, shifted)
