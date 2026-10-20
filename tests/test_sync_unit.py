"""Registry behavior using synthetic entries and mocked local storage."""

import json
import unittest
from unittest.mock import patch

import sync_unit


class TestSyncUnitRegistry(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"organization": "org_a", "external_system": "system_a", "project": "project_1"},
            {"organization": "org_a", "external_system": "system_b", "project": "project_2"},
            {"organization": "org_b", "external_system": "system_a", "project": "project_3"},
        ]
        storage_patch = patch("sync_unit.storage_access.StorageAccess")
        self.storage = storage_patch.start().return_value
        self.addCleanup(storage_patch.stop)
        self.storage.read_from_storage.return_value = json.dumps(self.records)

    def test_get_sync_units_deserializes_registry(self):
        units = sync_unit.get_sync_units()
        self.assertEqual(units, [sync_unit.SyncUnit("org_a", "system_a", "project_1"),
                                 sync_unit.SyncUnit("org_a", "system_b", "project_2"),
                                 sync_unit.SyncUnit("org_b", "system_a", "project_3")])
        self.storage.read_from_storage.assert_called_once_with("sync_units_register.json")

    def test_organization_filter_excludes_other_organizations(self):
        units = sync_unit.get_sync_units_for_organization("org_a")
        self.assertEqual([unit.project_id for unit in units], ["project_1", "project_2"])
        self.assertEqual(sync_unit.get_sync_units_for_organization("unknown"), [])

    def test_organization_and_system_filter_uses_both_fields(self):
        units = sync_unit.get_sync_units_for_organization_and_external_system("org_a", "system_a")
        self.assertEqual(units, [sync_unit.SyncUnit("org_a", "system_a", "project_1")])

    def test_validation_requires_registered_triplet(self):
        self.assertTrue(sync_unit.validate_sync_unit(sync_unit.SyncUnit("org_a", "system_a", "project_1")))
        self.assertFalse(sync_unit.validate_sync_unit(sync_unit.SyncUnit("org_a", "system_a", "unknown")))

    def test_empty_registry_has_no_units(self):
        self.storage.read_from_storage.return_value = "[]"
        self.assertEqual(sync_unit.get_sync_units(), [])


if __name__ == "__main__":
    unittest.main()
