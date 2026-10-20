"""Tests for rl_training.reward module."""

import unittest

from rl_training.reward import (
    RewardConfig,
    RewardFunction,
    compute_dataset_stats,
)


class TestRewardConfig(unittest.TestCase):

    def test_default_values(self):
        cfg = RewardConfig()
        self.assertEqual(cfg.sync_cost, 1.0)
        self.assertEqual(cfg.late_penalty, 2.0)
        self.assertEqual(cfg.mode, "event_rate")

    def test_from_dict(self):
        cfg = RewardConfig.from_dict({
            "sync_cost": 0.5,
            "late_penalty": 3.0,
            "mode": "time_to_event",
        })
        self.assertEqual(cfg.sync_cost, 0.5)
        self.assertEqual(cfg.late_penalty, 3.0)
        self.assertEqual(cfg.mode, "time_to_event")

    def test_from_dict_ignores_unknown_keys(self):
        cfg = RewardConfig.from_dict({"sync_cost": 0.5, "unknown_key": 99})
        self.assertEqual(cfg.sync_cost, 0.5)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            RewardConfig(mode="invalid")

    def test_to_dict_roundtrip(self):
        cfg = RewardConfig(sync_cost=2.0, late_penalty=5.0)
        d = cfg.to_dict()
        cfg2 = RewardConfig.from_dict(d)
        self.assertEqual(cfg.sync_cost, cfg2.sync_cost)
        self.assertEqual(cfg.late_penalty, cfg2.late_penalty)


class TestRewardFunctionEventRate(unittest.TestCase):

    def setUp(self):
        self.cfg = RewardConfig(
            sync_cost=1.0,
            late_penalty=2.0,
            successful_sync_reward=0.5,
            empty_sync_penalty=1.0,
            aoi_penalty_per_minute=0.01,
            aoi_threshold_minutes=5.0,
            mode="event_rate",
        )
        self.reward_fn = RewardFunction(self.cfg)

    def test_sync_with_events(self):
        """Sync when events exist → good sync reward."""
        r = self.reward_fn.compute(action=1, actual_value=3.0, current_aoi=0.0)
        expected = -1.0 + 0.5  # -sync_cost + successful_sync_reward
        self.assertAlmostEqual(r, expected)

    def test_sync_without_events(self):
        """Sync when no events → empty sync penalty."""
        r = self.reward_fn.compute(action=1, actual_value=0.0, current_aoi=0.0)
        expected = -1.0  # -empty_sync_penalty
        self.assertAlmostEqual(r, expected)

    def test_no_sync_with_events(self):
        """Don't sync when events exist → late penalty."""
        r = self.reward_fn.compute(action=0, actual_value=3.0, current_aoi=0.0)
        expected = -2.0  # -late_penalty
        self.assertAlmostEqual(r, expected)

    def test_no_sync_no_events(self):
        """Don't sync when calm → no penalty (best outcome)."""
        r = self.reward_fn.compute(action=0, actual_value=0.0, current_aoi=0.0)
        self.assertAlmostEqual(r, 0.0)

    def test_aoi_penalty_below_threshold(self):
        """AoI below threshold → no AoI penalty."""
        r = self.reward_fn.compute(action=0, actual_value=0.0, current_aoi=3.0)
        self.assertAlmostEqual(r, 0.0)

    def test_aoi_penalty_above_threshold(self):
        """AoI above threshold → continuous penalty."""
        r = self.reward_fn.compute(action=0, actual_value=0.0, current_aoi=10.0)
        # AoI excess = 10 - 5 = 5, penalty = -0.01 * 5 = -0.05
        self.assertAlmostEqual(r, -0.05)

    def test_combined_discrete_and_aoi(self):
        """Discrete penalty + AoI penalty stack."""
        r = self.reward_fn.compute(action=0, actual_value=3.0, current_aoi=15.0)
        # late_penalty = -2.0, aoi_penalty = -0.01 * 10 = -0.10
        self.assertAlmostEqual(r, -2.10)


class TestRewardFunctionTimeToEvent(unittest.TestCase):

    def setUp(self):
        self.cfg = RewardConfig(
            sync_cost=1.0,
            late_penalty=2.0,
            successful_sync_reward=0.5,
            empty_sync_penalty=1.0,
            aoi_penalty_per_minute=0.01,
            aoi_threshold_minutes=5.0,
            mode="time_to_event",
        )
        self.reward_fn = RewardFunction(self.cfg)

    def test_sync_when_event_imminent(self):
        """TTE < threshold → urgent, sync is good."""
        r = self.reward_fn.compute(action=1, actual_value=2.0, current_aoi=0.0)
        expected = -1.0 + 0.5
        self.assertAlmostEqual(r, expected)

    def test_sync_when_event_far(self):
        """TTE > threshold → calm, sync is wasted."""
        r = self.reward_fn.compute(action=1, actual_value=20.0, current_aoi=0.0)
        expected = -1.0
        self.assertAlmostEqual(r, expected)

    def test_no_sync_when_event_imminent(self):
        """TTE < threshold → missed event."""
        r = self.reward_fn.compute(action=0, actual_value=2.0, current_aoi=0.0)
        expected = -2.0
        self.assertAlmostEqual(r, expected)

    def test_no_sync_when_event_far(self):
        """TTE > threshold → correct wait."""
        r = self.reward_fn.compute(action=0, actual_value=20.0, current_aoi=0.0)
        self.assertAlmostEqual(r, 0.0)


class TestRewardComponents(unittest.TestCase):

    def test_compute_components_keys(self):
        cfg = RewardConfig()
        reward_fn = RewardFunction(cfg)
        components = reward_fn.compute_components(
            action=1, actual_value=1.0, current_aoi=10.0
        )
        expected_keys = {
            "discrete_reward", "aoi_penalty", "item_bonus", "total_reward",
            "is_urgent", "did_sync", "aoi_excess",
        }
        self.assertEqual(set(components.keys()), expected_keys)

    def test_components_sum_to_total(self):
        cfg = RewardConfig()
        reward_fn = RewardFunction(cfg)
        components = reward_fn.compute_components(
            action=1, actual_value=1.0, current_aoi=10.0
        )
        total = components["discrete_reward"] + components["aoi_penalty"] + components["item_bonus"]
        self.assertAlmostEqual(total, components["total_reward"])

    def test_components_match_compute(self):
        cfg = RewardConfig()
        reward_fn = RewardFunction(cfg)
        r = reward_fn.compute(action=0, actual_value=5.0, current_aoi=8.0)
        components = reward_fn.compute_components(
            action=0, actual_value=5.0, current_aoi=8.0
        )
        self.assertAlmostEqual(r, components["total_reward"])


class TestRewardModeAoiOnly(unittest.TestCase):
    """Tests for reward_mode='aoi_only' — no ground-truth peek."""

    def setUp(self):
        self.cfg = RewardConfig(
            sync_cost=1.0,
            aoi_penalty_per_minute=0.01,
            aoi_threshold_minutes=5.0,
            mode="event_rate",
            reward_mode="aoi_only",
        )
        self.reward_fn = RewardFunction(self.cfg)

    def test_sync_always_costs_sync_cost(self):
        """In aoi_only mode, sync reward is always -sync_cost regardless of events."""
        r_events = self.reward_fn.compute(action=1, actual_value=10.0, current_aoi=0.0)
        r_empty = self.reward_fn.compute(action=1, actual_value=0.0, current_aoi=0.0)
        self.assertAlmostEqual(r_events, -1.0)
        self.assertAlmostEqual(r_empty, -1.0)

    def test_skip_always_zero_discrete(self):
        """In aoi_only mode, skip has no late_penalty even when events exist."""
        r = self.reward_fn.compute(action=0, actual_value=10.0, current_aoi=0.0)
        self.assertAlmostEqual(r, 0.0)

    def test_aoi_penalty_still_applies(self):
        """Continuous AoI penalty still works in aoi_only mode."""
        r = self.reward_fn.compute(action=0, actual_value=0.0, current_aoi=15.0)
        expected = -0.01 * (15.0 - 5.0)
        self.assertAlmostEqual(r, expected)

    def test_components_is_urgent_always_false(self):
        components = self.reward_fn.compute_components(
            action=0, actual_value=10.0, current_aoi=0.0
        )
        self.assertAlmostEqual(components["is_urgent"], 0.0)


class TestRewardConfigRewardMode(unittest.TestCase):

    def test_invalid_reward_mode_raises(self):
        with self.assertRaises(ValueError):
            RewardConfig(reward_mode="invalid")

    def test_reward_mode_in_to_dict(self):
        cfg = RewardConfig(reward_mode="aoi_only")
        d = cfg.to_dict()
        self.assertEqual(d["reward_mode"], "aoi_only")

    def test_reward_mode_from_dict(self):
        cfg = RewardConfig.from_dict({"reward_mode": "aoi_only"})
        self.assertEqual(cfg.reward_mode, "aoi_only")

    def test_default_reward_mode_is_full(self):
        cfg = RewardConfig()
        self.assertEqual(cfg.reward_mode, "full")


class TestRewardPresets(unittest.TestCase):

    def test_default_preset(self):
        cfg = RewardConfig.from_preset("default")
        self.assertAlmostEqual(cfg.sync_cost, 1.0)
        self.assertAlmostEqual(cfg.late_penalty, 2.0)

    def test_balanced_preset(self):
        cfg = RewardConfig.from_preset("balanced")
        self.assertAlmostEqual(cfg.sync_cost, 0.5)
        self.assertAlmostEqual(cfg.successful_sync_reward, 1.5)

    def test_sparse_friendly_preset(self):
        cfg = RewardConfig.from_preset("sparse_friendly")
        self.assertAlmostEqual(cfg.sync_cost, 0.3)
        self.assertAlmostEqual(cfg.late_penalty, 3.0)
        self.assertAlmostEqual(cfg.aoi_threshold_minutes, 10.0)

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            RewardConfig.from_preset("nonexistent")

    def test_preset_with_overrides(self):
        cfg = RewardConfig.from_preset("balanced", sync_cost=0.1)
        self.assertAlmostEqual(cfg.sync_cost, 0.1)
        self.assertAlmostEqual(cfg.successful_sync_reward, 1.5)

    def test_from_dict_with_preset_key(self):
        cfg = RewardConfig.from_dict({
            "preset": "sparse_friendly",
            "mode": "time_to_event",
        })
        self.assertAlmostEqual(cfg.sync_cost, 0.3)
        self.assertEqual(cfg.mode, "time_to_event")

    def test_from_dict_without_preset(self):
        cfg = RewardConfig.from_dict({"sync_cost": 5.0})
        self.assertAlmostEqual(cfg.sync_cost, 5.0)

    def test_balanced_sync_reward_exceeds_cost(self):
        """Balanced preset: successful sync yields net positive discrete reward."""
        cfg = RewardConfig.from_preset("balanced")
        reward_fn = RewardFunction(cfg)
        r = reward_fn.compute(action=1, actual_value=3.0, current_aoi=0.0)
        self.assertGreater(r, 0.0)


class TestComputeDatasetStats(unittest.TestCase):

    def test_dense_event_rate(self):
        """Events every 5 minutes → median IET ≈ 5."""
        values = [0.0] * 100
        for i in range(0, 100, 5):
            values[i] = 3.0
        stats = compute_dataset_stats(values, "event_rate")
        self.assertAlmostEqual(stats["median_inter_event_time"], 5.0)
        self.assertAlmostEqual(stats["event_rate"], 20 / 100)
        self.assertEqual(stats["total_minutes"], 100)

    def test_sparse_event_rate(self):
        """Events every 50 minutes → median IET ≈ 50."""
        values = [0.0] * 500
        for i in range(0, 500, 50):
            values[i] = 1.0
        stats = compute_dataset_stats(values, "event_rate")
        self.assertAlmostEqual(stats["median_inter_event_time"], 50.0)

    def test_no_events(self):
        """No events → IET = total length."""
        values = [0.0] * 200
        stats = compute_dataset_stats(values, "event_rate")
        self.assertEqual(stats["median_inter_event_time"], 200.0)
        self.assertEqual(stats["total_events"], 0.0)

    def test_single_event(self):
        values = [0.0] * 100
        values[50] = 1.0
        stats = compute_dataset_stats(values, "event_rate")
        self.assertEqual(stats["total_events"], 1.0)
        self.assertEqual(stats["median_inter_event_time"], 100.0)


class TestFromDataset(unittest.TestCase):

    def test_dense_data_gets_smaller_threshold(self):
        """Dense events → threshold near minimum (5.0)."""
        values = [0.0] * 200
        for i in range(0, 200, 5):
            values[i] = 3.0
        cfg = RewardConfig.from_dataset(values, "event_rate")
        self.assertAlmostEqual(cfg.aoi_threshold_minutes, 5.0)
        self.assertGreater(cfg.aoi_penalty_per_minute, 0.01)

    def test_sparse_data_gets_larger_threshold(self):
        """Sparse events (every 100 min) → threshold > 5.0."""
        values = [0.0] * 2000
        for i in range(0, 2000, 100):
            values[i] = 1.0
        cfg = RewardConfig.from_dataset(values, "event_rate")
        self.assertGreater(cfg.aoi_threshold_minutes, 5.0)
        self.assertLess(cfg.aoi_penalty_per_minute, 0.01)

    def test_very_sparse_data(self):
        """Events every 700 min → large threshold, tiny per-step penalty."""
        values = [0.0] * 5000
        for i in range(0, 5000, 700):
            values[i] = 1.0
        cfg = RewardConfig.from_dataset(values, "event_rate")
        self.assertGreater(cfg.aoi_threshold_minutes, 100.0)
        self.assertLess(cfg.aoi_penalty_per_minute, 0.001)

    def test_gap_cost_invariant(self):
        """Total AoI cost over one median gap ≈ gap_cost_ratio * empty_sync_penalty."""
        values = [0.0] * 2000
        for i in range(0, 2000, 100):
            values[i] = 1.0
        gap_cost_ratio = 3.0
        cfg = RewardConfig.from_dataset(values, "event_rate", gap_cost_ratio=gap_cost_ratio)

        median_iet = 100.0
        effective_gap = median_iet - cfg.aoi_threshold_minutes
        total_aoi_cost = cfg.aoi_penalty_per_minute * (effective_gap ** 2) / 2
        expected_cost = gap_cost_ratio * cfg.empty_sync_penalty
        self.assertAlmostEqual(total_aoi_cost, expected_cost, places=2)

    def test_overrides_applied(self):
        values = [0.0] * 200
        for i in range(0, 200, 10):
            values[i] = 1.0
        cfg = RewardConfig.from_dataset(values, "event_rate", sync_cost=99.0)
        self.assertAlmostEqual(cfg.sync_cost, 99.0)

    def test_mode_set_correctly(self):
        values = [10.0] * 100
        cfg = RewardConfig.from_dataset(values, "time_to_event")
        self.assertEqual(cfg.mode, "time_to_event")

    def test_auto_preset_via_from_preset_raises(self):
        with self.assertRaises(ValueError) as ctx:
            RewardConfig.from_preset("auto")
        self.assertIn("from_dataset", str(ctx.exception))

    def test_successful_sync_is_net_positive_for_sparse(self):
        """For sparse data, successful sync should yield net positive reward."""
        values = [0.0] * 5000
        for i in range(0, 5000, 500):
            values[i] = 1.0
        cfg = RewardConfig.from_dataset(values, "event_rate")
        reward_fn = RewardFunction(cfg)
        r = reward_fn.compute(action=1, actual_value=1.0, current_aoi=0.0)
        self.assertGreater(r, 0.0)


class TestAoiModeQuadratic(unittest.TestCase):

    def setUp(self):
        self.cfg = RewardConfig(
            aoi_penalty_per_minute=0.01,
            aoi_mode="quadratic",
            mode="event_rate",
        )
        self.reward_fn = RewardFunction(self.cfg)

    def test_zero_aoi_no_penalty(self):
        r = self.reward_fn.compute(action=0, actual_value=0.0, current_aoi=0.0)
        self.assertAlmostEqual(r, 0.0)

    def test_quadratic_penalty_grows(self):
        """Penalty at aoi=10 is 100x penalty at aoi=1."""
        r1 = self.reward_fn.compute(action=0, actual_value=0.0, current_aoi=1.0)
        r10 = self.reward_fn.compute(action=0, actual_value=0.0, current_aoi=10.0)
        self.assertAlmostEqual(r1, -0.01)
        self.assertAlmostEqual(r10, -1.0)
        self.assertAlmostEqual(r10 / r1, 100.0)

    def test_no_threshold_effect(self):
        """Even aoi=2 (below default threshold=5) incurs penalty."""
        r = self.reward_fn.compute(action=0, actual_value=0.0, current_aoi=2.0)
        self.assertAlmostEqual(r, -0.04)


class TestAoiModeLinearNoThreshold(unittest.TestCase):

    def setUp(self):
        self.cfg = RewardConfig(
            aoi_penalty_per_minute=0.01,
            aoi_threshold_minutes=5.0,
            aoi_mode="linear_no_threshold",
            mode="event_rate",
        )
        self.reward_fn = RewardFunction(self.cfg)

    def test_penalty_from_zero(self):
        """Penalty starts immediately, not after threshold."""
        r = self.reward_fn.compute(action=0, actual_value=0.0, current_aoi=3.0)
        self.assertAlmostEqual(r, -0.03)

    def test_differs_from_linear(self):
        """At aoi=10, linear gives -0.05, linear_no_threshold gives -0.10."""
        r = self.reward_fn.compute(action=0, actual_value=0.0, current_aoi=10.0)
        self.assertAlmostEqual(r, -0.10)


class TestItemRewardPerUnit(unittest.TestCase):

    def setUp(self):
        self.cfg = RewardConfig(
            sync_cost=1.0,
            successful_sync_reward=0.5,
            empty_sync_penalty=1.0,
            item_reward_per_unit=0.1,
            aoi_penalty_per_minute=0.0,
            mode="event_rate",
        )
        self.reward_fn = RewardFunction(self.cfg)

    def test_item_bonus_on_sync(self):
        """Sync with 5 items: -1.0 + 0.5 + 0.1*5 = 0.0."""
        r = self.reward_fn.compute(
            action=1, actual_value=5.0, current_aoi=0.0, items_found=5.0,
        )
        self.assertAlmostEqual(r, 0.0)

    def test_item_bonus_scales_with_count(self):
        r10 = self.reward_fn.compute(
            action=1, actual_value=10.0, current_aoi=0.0, items_found=10.0,
        )
        r2 = self.reward_fn.compute(
            action=1, actual_value=2.0, current_aoi=0.0, items_found=2.0,
        )
        self.assertAlmostEqual(r10 - r2, 0.8)

    def test_no_item_bonus_on_skip(self):
        r = self.reward_fn.compute(
            action=0, actual_value=5.0, current_aoi=0.0, items_found=5.0,
        )
        self.assertAlmostEqual(r, -2.0)

    def test_no_item_bonus_when_zero(self):
        r = self.reward_fn.compute(
            action=1, actual_value=0.0, current_aoi=0.0, items_found=0.0,
        )
        self.assertAlmostEqual(r, -1.0)

    def test_default_items_found_preserves_behavior(self):
        """Default items_found=0.0 means no bonus (backward compat)."""
        cfg = RewardConfig(
            sync_cost=1.0, successful_sync_reward=0.5,
            item_reward_per_unit=0.1, mode="event_rate",
        )
        rf = RewardFunction(cfg)
        r = rf.compute(action=1, actual_value=3.0, current_aoi=0.0)
        self.assertAlmostEqual(r, -0.5)


class TestAoiModeValidation(unittest.TestCase):

    def test_invalid_aoi_mode_raises(self):
        with self.assertRaises(ValueError):
            RewardConfig(aoi_mode="exponential")

    def test_valid_aoi_modes(self):
        for mode in ("linear", "quadratic", "linear_no_threshold"):
            cfg = RewardConfig(aoi_mode=mode)
            self.assertEqual(cfg.aoi_mode, mode)

    def test_aoi_mode_in_to_dict(self):
        cfg = RewardConfig(aoi_mode="quadratic")
        d = cfg.to_dict()
        self.assertEqual(d["aoi_mode"], "quadratic")

    def test_aoi_mode_roundtrip(self):
        cfg = RewardConfig(aoi_mode="quadratic", item_reward_per_unit=0.5)
        d = cfg.to_dict()
        cfg2 = RewardConfig.from_dict(d)
        self.assertEqual(cfg2.aoi_mode, "quadratic")
        self.assertAlmostEqual(cfg2.item_reward_per_unit, 0.5)


if __name__ == "__main__":
    unittest.main()
