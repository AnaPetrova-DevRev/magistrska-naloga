"""UTC date filtering for synthetic policy evaluation metadata."""

from copy import deepcopy
from datetime import date
import unittest
from unittest.mock import patch

import sync_policy_dashboard as dashboard


class TestSidebarDateFilter(unittest.TestCase):
    def test_mixed_timezones_and_naive_metadata_use_complete_utc_days(self):
        def evaluation(name, start, end):
            return {"name": name, "eval_start": start, "eval_end": end}

        evaluations = {"synthetic": [
            evaluation("before", "2025-01-01T12:00Z", "2025-01-01T23:59:59Z"),
            evaluation("touches_start", "2025-01-01T23:00Z", "2025-01-02T00:00Z"),
            evaluation("late_on_end_day", "2025-01-02T22:00Z", "2025-01-02T23:00Z"),
            evaluation("next_day", "2025-01-03T00:00Z", "2025-01-03T01:00Z"),
            evaluation("offset", "2025-01-03T01:30+02:00", "2025-01-03T01:45+02:00"),
            evaluation("naive", "2025-01-02T12:00", "2025-01-02T13:00"),
            evaluation("missing", None, None),
        ]}
        before = deepcopy(evaluations)
        with patch.object(dashboard.st, "sidebar") as sidebar:
            sidebar.date_input.return_value = (date(2025, 1, 2), date(2025, 1, 2))
            filtered = dashboard.sidebar_date_filter(evaluations)
            self.assertEqual(sidebar.date_input.call_args.kwargs["min_value"], date(2025, 1, 1))
            self.assertEqual(sidebar.date_input.call_args.kwargs["max_value"], date(2025, 1, 3))
        self.assertEqual(
            [evaluation["name"] for evaluation in filtered["synthetic"]],
            ["touches_start", "late_on_end_day", "offset", "naive", "missing"],
        )
        self.assertEqual(evaluations, before)

    def test_naive_evaluation_late_on_selected_end_day_is_included(self):
        evaluations = {"synthetic": [{
            "eval_start": "2025-01-02T18:00", "eval_end": "2025-01-02T19:00",
        }]}
        with patch.object(dashboard.st, "sidebar") as sidebar:
            sidebar.date_input.return_value = (date(2025, 1, 2), date(2025, 1, 2))
            self.assertEqual(dashboard.sidebar_date_filter(evaluations), evaluations)

    def test_incomplete_date_selection_preserves_evaluations(self):
        evaluations = {"synthetic": [{
            "eval_start": "2025-01-02T18:00Z", "eval_end": "2025-01-02T19:00Z",
        }]}
        with patch.object(dashboard.st, "sidebar") as sidebar:
            sidebar.date_input.return_value = (date(2025, 1, 2),)
            self.assertIs(dashboard.sidebar_date_filter(evaluations), evaluations)

    def test_missing_start_metadata_does_not_require_date_input(self):
        evaluations = {"synthetic": [{"eval_start": None, "eval_end": None}]}
        with patch.object(dashboard.st, "sidebar") as sidebar:
            self.assertIs(dashboard.sidebar_date_filter(evaluations), evaluations)
            sidebar.date_input.assert_not_called()


if __name__ == "__main__":
    unittest.main()
