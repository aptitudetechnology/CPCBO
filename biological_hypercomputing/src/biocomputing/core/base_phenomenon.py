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
        """Initialize the phenomenon."""
        pass
    
    @abstractmethod
    def step(self, dt: float) -> None:
        """Execute one simulation step."""
        pass
    
    @abstractmethod
    def compute(self, input_data: np.ndarray) -> np.ndarray:
        """Perform computation using this phenomenon."""
        pass
    
    @abstractmethod
    def get_emergent_properties(self) -> Dict[str, Any]:
        """Return emergent properties of this phenomenon."""
        pass
    
    def add_synergy(self, other_phenomenon: 'BasePhenomenon') -> None:
        """Add synergistic interaction with another phenomenon."""
        self.synergies.append(other_phenomenon)
    
    def measure_performance(self) -> Dict[str, float]:
        """Measure performance metrics."""
        return self.metrics.copy()
