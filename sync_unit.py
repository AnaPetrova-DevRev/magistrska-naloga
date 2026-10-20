



from dataclasses import dataclass
import storage_access
import json


@dataclass
class SyncUnit:
    """Represents a sync unit: Organization -> External System -> Project"""
    organization_id: str
    external_system_id: str
    project_id: str

    def get_name(self) -> str:
        """Get the unique name for this sync unit"""
        return f"{self.organization_id}_{self.external_system_id}_{self.project_id}"

    def get_path_for_sync_unit(self) -> str:
        """Get the file path for this sync unit"""
        return f"{self.organization_id}/{self.external_system_id}/{self.project_id}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "organization": self.organization_id,
            "external_system": self.external_system_id,
            "project": self.project_id
        }

def get_sync_units() -> list[SyncUnit]:
    """Get all registered sync units from storage"""
    sa = storage_access.StorageAccess()
    file = sa.read_from_storage("sync_units_register.json")

    sync_units_data = json.loads(file)

    sync_units: list[SyncUnit] = []
    for unit_data in sync_units_data:
        sync_unit = SyncUnit(
            organization_id=unit_data["organization"],
            external_system_id=unit_data["external_system"],
            project_id=unit_data["project"]
        )
        sync_units.append(sync_unit)

    return sync_units


def get_sync_units_for_organization(organization_id: str) -> list[SyncUnit]:
    """Get all sync units for a specific organization"""
    sync_units = get_sync_units()
    return [unit for unit in sync_units if unit.organization_id == organization_id]


def get_sync_units_for_organization_and_external_system(
    organization_id: str,
    external_system_id: str
) -> list[SyncUnit]:
    """Get all sync units for a specific organization and external system"""
    sync_units = get_sync_units()
    return [
        unit for unit in sync_units
        if unit.organization_id == organization_id and unit.external_system_id == external_system_id
    ]


def validate_sync_unit(sync_unit: SyncUnit) -> bool:
    """Check if a sync unit exists in the registry"""
    sync_units = get_sync_units()
    return sync_unit in sync_units
