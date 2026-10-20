from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrchestratorMessage:
    """Message sent to orchestrator to trigger an action on a sync unit"""
    sync_unit_obj: Any  # SyncUnit object
    action: str
    parameters: dict = field(default_factory=dict)

    def to_string(self) -> str:
        return f"Sync unit: {self.sync_unit_obj.get_name()}, Action: {self.action}"

    def get_sync_unit(self) -> Any:
        return self.sync_unit_obj

    def to_dict(self) -> dict:
        return {
            "sync_unit": self.sync_unit_obj.to_dict(),
            "action": self.action,
            "parameters": self.parameters
        }


@dataclass
class OrchestratorResponse:
    """Response from orchestrator after processing a message"""
    success: bool
    message: str
    is_done: bool = False
    logs: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def add_log(self, log: str) -> None:
        self.logs.append(log)

    def to_string(self) -> str:
        return f"Success: {self.success}, Message: {self.message}, Logs: {self.logs}"

    def to_dict(self) -> dict:
        result = {
            "success": self.success,
            "message": self.message,
            "is_done": self.is_done,
            "logs": self.logs
        }
        if self.data:
            result["data"] = self.data
        return result