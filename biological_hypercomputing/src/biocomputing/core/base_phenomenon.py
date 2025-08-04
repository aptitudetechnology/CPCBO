"""Base class for all biological computing phenomena."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

class BasePhenomenon(ABC):
    """Abstract base class for biological computing phenomena."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = {}
        self.metrics = {}
        self.synergies = []
    @abstractmethod
    def initialize(self) -> None:
        pass
    @abstractmethod
    def step(self, dt: float) -> None:
        pass
    @abstractmethod
    def compute(self, input_data: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def get_emergent_properties(self) -> Dict[str, Any]:
        pass
