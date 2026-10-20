# each app has its own state

# last processed file, are we done with the work, when was the last time this app was called etc

# for each app - when it is called, for example for the brightpath salesforce import unit:
# lets say we called the data cleanup app
# it will save the "data_cleanup_state" in that folder and this state is ONLY applicaple for that sync unit.

from dataclasses import dataclass
from typing import Optional


@dataclass
class State:
    """Tracks the processing state for a specific function"""
    last_processed_chunk_number: int = 0
    is_done: bool = False
    last_called_time: Optional[str] = None
    total_processed: int = 0
    function_name: str = ""

    @staticmethod
    def from_dict(data: dict) -> 'State':
        """Create a State object from a dictionary"""
        return State(
            last_processed_chunk_number=data.get("last_processed_chunk_number", 0),
            is_done=data.get("is_done", False),
            last_called_time=data.get("last_called_time", None),
            total_processed=data.get("total_processed", 0),
            function_name=data.get("function_name", "")
        )