from typing import Protocol
import numpy as np

class PivotingStrategy(Protocol):
    def pick_entering_index(self, reduced_costs: np.ndarray) -> int:
        # Used as model
        pass

class SmallestSubscriptRule:
    @staticmethod
    def pick_entering_index(reduced_costs: np.ndarray) -> int:
        for i, cost in enumerate(reduced_costs):
            if (cost < 0):
                return i
        raise RuntimeError("Pivoting strategy assumes some reduced cost is negative.")