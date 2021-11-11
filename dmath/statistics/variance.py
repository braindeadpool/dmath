from __future__ import annotations
import numpy as np
import numpy.typing as npt
from ..batched_unit import BatchedUnit


class Variance1D(BatchedUnit):
    def __init__(self, values: npt.ArrayLike, is_sample_variance=True) -> None:
        super().__init__()
        self.is_sample_variance = is_sample_variance
        values = np.array(values)
        values = values.squeeze()

        if values.ndim > 1:
            raise Exception("Should be 1D array like")
        
        self.num_samples = len(values)
        self.mean = np.mean(values)
        # Sum of squares of differences from the current mean
        self.ssd = np.sum((values - self.mean)**2)
    
    def update(self, values: npt.ArrayLike) -> None:
        values = np.array(values)
        values = values.squeeze()

        if values.ndim > 1:
            raise Exception("Should be 1D array like")

        self.merge(Variance1D(values))
    
    def merge(self, batched_unit: Variance1D) -> None:
        new_num_samples = self.num_samples + batched_unit.num_samples
        delta = batched_unit.mean - self.mean
        self.ssd += batched_unit.ssd + delta**2 * (self.num_samples * batched_unit.num_samples)/new_num_samples
        self.mean += delta*batched_unit.num_samples/new_num_samples
        self.num_samples = new_num_samples
    
    def compute(self) -> float:
        if self.is_sample_variance:
            return self.ssd/(self.num_samples-1)
        return self.ssd/self.num_samples