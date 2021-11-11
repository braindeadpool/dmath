from __future__ import annotations
from abc import ABC, abstractmethod


class BatchedUnit(ABC):

    @abstractmethod
    def update(self) -> None:
        """
        Update the intermediate values tracked by current batch.
        """
        return

    @abstractmethod
    def merge(self, batched_unit: BatchedUnit) -> None:
        """
        Merge the current batch's intermediate values with the intermediate values computed in another batch.
        """
        return
    
    @abstractmethod
    def compute(self) -> Any:
        """
        Compute the final value from the intermediate values.
        """
        return