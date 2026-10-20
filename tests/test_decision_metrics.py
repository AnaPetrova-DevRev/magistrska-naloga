"""Tests for decision-quality metrics (AoII and SPO Regret)."""

import unittest

import numpy as np
import pandas as pd

from sync_policy_evaluation.decision_metrics import (
    compute_aoii_series,
    compute_aoii_summary,
    compute_oracle_schedule,
    compute_spo_regret,
    _detect_events_array,
    _compute_true_aoi_series,
    _count_empty_syncs,
)


class TestDetectEventsArray(unittest.TestCase):
    """Test event detection from actual_values."""

    def test_event_rate_positive_is_event(self):
        actual = np.array([0, 0, 3, 0, 1, 0])
        events = _detect_events_array(actual, "event_rate")
        np.testing.assert_array_equal(events, [False, False, True, False, True, False])

    def test_event_rate_zero_is_no_event(self):
        actual = np.zeros(5)
        events = _detect_events_array(actual, "event_rate")
        self.assertTrue(np.all(~events))

    def test_tte_upward_jump_is_event(self):
        # TTE counts down, then jumps up on event (countdown resets)
        actual = np.array([10, 9, 8, 7, 15, 14, 13])
        events = _detect_events_array(actual, "time_to_event")
        expected = [False, False, False, False, True, False, False]
        np.testing.assert_array_equal(events, expected)

    def test_tte_small_jump_ignored(self):
        actual = np.array([10, 9, 9.5, 8, 7])
        events = _detect_events_array(actual, "time_to_event", tte_min_jump=1.0)
        self.assertTrue(np.all(~events))


class TestAoIISeries(unittest.TestCase):
    """Test AoII computation on synthetic traces."""

    def test_calm_period_zero_aoii(self):
        """No events → AoII always 0."""
        actual = np.zeros(10)
        actions = np.zeros(10, dtype=np.int64)
        aoii = compute_aoii_series(actual, actions, "event_rate")
        np.testing.assert_array_equal(aoii, np.zeros(10))

    def test_event_then_immediate_sync(self):
        """Event at t=2, sync at t=2 → AoII stays 0 (sync captures it)."""
        actual = np.array([0, 0, 5, 0, 0])
        actions = np.array([0, 0, 1, 0, 0], dtype=np.int64)
        aoii = compute_aoii_series(actual, actions, "event_rate")
        np.testing.assert_array_equal(aoii, [0, 0, 0, 0, 0])

    def test_event_missed_grows_linearly(self):
        """Event at t=1 with value 2, no syncs → AoII = time * pending (linear in time)."""
        actual = np.array([0, 2, 0, 0, 0])
        actions = np.zeros(5, dtype=np.int64)
        aoii = compute_aoii_series(actual, actions, "event_rate")
        # t=0: pending=0 → 0
        # t=1: event arrives, pending=2, time_since_correct=1 → 1*2=2
        # t=2: pending=2, time_since_correct=2 → 2*2=4
        # t=3: pending=2, time_since_correct=3 → 3*2=6
        # t=4: pending=2, time_since_correct=4 → 4*2=8
        np.testing.assert_array_equal(aoii, [0, 2, 4, 6, 8])

    def test_multiple_events_accumulate(self):
        """Events at t=1 (value 1) and t=3 (value 2), no syncs."""
        actual = np.array([0, 1, 0, 2, 0])
        actions = np.zeros(5, dtype=np.int64)
        aoii = compute_aoii_series(actual, actions, "event_rate")
        # t=0: 0
        # t=1: pending=1, time=1 → 1
        # t=2: pending=1, time=2 → 2
        # t=3: pending=1+2=3, time=3 → 9
        # t=4: pending=3, time=4 → 12
        np.testing.assert_array_equal(aoii, [0, 1, 2, 9, 12])

    def test_sync_resets_aoii(self):
        """Event at t=1, sync at t=3 → AoII resets."""
        actual = np.array([0, 1, 0, 0, 0, 0])
        actions = np.array([0, 0, 0, 1, 0, 0], dtype=np.int64)
        aoii = compute_aoii_series(actual, actions, "event_rate")
        # t=0: 0
        # t=1: pending=1, time=1 → 1
        # t=2: pending=1, time=2 → 2
        # t=3: sync → pending=0, time=0 → 0
        # t=4: no events, pending=0 → 0
        # t=5: no events, pending=0 → 0
        np.testing.assert_array_equal(aoii, [0, 1, 2, 0, 0, 0])

    def test_tte_task_type(self):
        """TTE: event detected by upward jump."""
        actual = np.array([5, 4, 3, 10, 9, 8])
        actions = np.zeros(6, dtype=np.int64)
        aoii = compute_aoii_series(actual, actions, "time_to_event")
        # Event at t=3 (jump from 3 to 10)
        # t=0-2: no events, aoii=0
        # t=3: pending=1, time=1 → 1
        # t=4: pending=1, time=2 → 2
        # t=5: pending=1, time=3 → 3
        np.testing.assert_array_equal(aoii, [0, 0, 0, 1, 2, 3])


class TestAoIISummary(unittest.TestCase):
    """Test AoII summary aggregation."""

    def test_empty_array(self):
        result = compute_aoii_summary(np.array([]))
        self.assertEqual(result["avg_aoii"], 0.0)

    def test_known_values(self):
        aoii = np.array([0, 1, 2, 0, 0, 10])
        result = compute_aoii_summary(aoii)
        self.assertAlmostEqual(result["avg_aoii"], np.mean(aoii))
        self.assertEqual(result["max_aoii"], 10.0)
        self.assertAlmostEqual(result["median_aoii"], float(np.median(aoii)))


class TestOracleSchedule(unittest.TestCase):
    """Test oracle optimal schedule computation."""

    def test_no_events_no_syncs(self):
        """Oracle never syncs when there are no events."""
        actual = np.zeros(10)
        oracle = compute_oracle_schedule(actual, "event_rate")
        np.testing.assert_array_equal(oracle, np.zeros(10))

    def test_syncs_on_event_minutes(self):
        """Oracle syncs immediately when events arrive."""
        actual = np.array([0, 0, 3, 0, 0, 1, 0])
        oracle = compute_oracle_schedule(actual, "event_rate")
        expected = np.array([0, 0, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(oracle, expected)

    def test_consecutive_events(self):
        """Oracle syncs each minute during a burst."""
        actual = np.array([0, 2, 3, 1, 0, 0])
        oracle = compute_oracle_schedule(actual, "event_rate")
        expected = np.array([0, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(oracle, expected)

    def test_oracle_has_zero_true_aoi(self):
        """Oracle achieves zero true AoI (syncs at every event)."""
        actual = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        oracle = compute_oracle_schedule(actual, "event_rate")
        true_aoi = _compute_true_aoi_series(actual, oracle, "event_rate")
        np.testing.assert_array_equal(true_aoi, np.zeros(10))

    def test_oracle_has_zero_empty_syncs(self):
        """Oracle never syncs when no events are pending."""
        actual = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        oracle = compute_oracle_schedule(actual, "event_rate")
        empty = _count_empty_syncs(oracle, actual, "event_rate")
        self.assertEqual(empty, 0)


class TestSPORegret(unittest.TestCase):
    """Test SPO Regret computation."""

    def test_oracle_policy_zero_regret(self):
        """When the policy IS the oracle, regret = 0."""
        actual = np.array([0, 0, 1, 0, 0, 1, 0])
        oracle = compute_oracle_schedule(actual, "event_rate")
        df = pd.DataFrame({
            "actual_value": actual,
            "action": oracle,
        })
        result = compute_spo_regret(df, "event_rate")
        self.assertAlmostEqual(result["spo_regret_total"], 0.0, places=10)

    def test_never_sync_positive_regret(self):
        """Never syncing when events exist → positive regret (staleness)."""
        actual = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        actions = np.zeros(10, dtype=np.int64)
        df = pd.DataFrame({"actual_value": actual, "action": actions})
        result = compute_spo_regret(df, "event_rate")
        self.assertGreater(result["spo_regret_total"], 0.0)

    def test_always_sync_positive_regret(self):
        """Always syncing (many empty syncs) → positive regret (waste)."""
        actual = np.zeros(10)
        actual[5] = 1
        actions = np.ones(10, dtype=np.int64)
        df = pd.DataFrame({"actual_value": actual, "action": actions})
        result = compute_spo_regret(df, "event_rate")
        self.assertGreater(result["spo_regret_total"], 0.0)

    def test_empty_dataframe(self):
        """Empty DataFrame returns zero regret."""
        df = pd.DataFrame({"actual_value": [], "action": []})
        result = compute_spo_regret(df, "event_rate")
        self.assertEqual(result["spo_regret_total"], 0.0)

    def test_result_keys(self):
        """Verify all expected keys are present."""
        actual = np.array([0, 1, 0, 0, 1])
        actions = np.array([0, 0, 1, 0, 1], dtype=np.int64)
        df = pd.DataFrame({"actual_value": actual, "action": actions})
        result = compute_spo_regret(df, "event_rate")
        expected_keys = {
            "spo_regret_total", "spo_regret_per_step",
            "oracle_sync_count", "oracle_avg_aoi",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_regret_components(self):
        """Verify regret = aoi_cost + empty_sync_cost."""
        # 1 event at t=2, never sync → true_aoi: [0,0,0,1,2,3,4], sum=10
        actual = np.array([0, 0, 1, 0, 0, 0, 0])
        actions = np.zeros(7, dtype=np.int64)
        df = pd.DataFrame({"actual_value": actual, "action": actions})
        result = compute_spo_regret(df, "event_rate")
        # aoi_cost = 10 * 0.02 = 0.2, empty_syncs = 0
        self.assertAlmostEqual(result["spo_regret_total"], 0.2, places=10)


class TestTrueAoISeries(unittest.TestCase):
    """Test internal true AoI computation."""

    def test_no_events_zero_aoi(self):
        actual = np.zeros(5)
        actions = np.zeros(5, dtype=np.int64)
        aoi = _compute_true_aoi_series(actual, actions, "event_rate")
        np.testing.assert_array_equal(aoi, np.zeros(5))

    def test_event_grows_until_sync(self):
        actual = np.array([0, 1, 0, 0, 0, 0])
        actions = np.array([0, 0, 0, 0, 1, 0], dtype=np.int64)
        aoi = _compute_true_aoi_series(actual, actions, "event_rate")
        # Event at t=1, AoI grows from t=1 onwards, sync at t=4 clears it
        expected = [0, 0, 1, 2, 0, 0]
        np.testing.assert_array_equal(aoi, expected)


if __name__ == "__main__":
    unittest.main()
