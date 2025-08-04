"""Unit tests for base phenomenon class."""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from biocomputing.core.base_phenomenon import BasePhenomenon

class MockPhenomenon(BasePhenomenon):
    """Mock implementation for testing."""
    
    def initialize(self):
        self.state = {'initialized': True}
    
    def step(self, dt: float):
        self.state['time'] = self.state.get('time', 0) + dt
    
    def compute(self, input_data: np.ndarray) -> np.ndarray:
        return input_data + 0.1
    
    def get_emergent_properties(self):
        return {'complexity': 0.5}

class TestBasePhenomenon(unittest.TestCase):
    """Test cases for BasePhenomenon class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {'test_param': 1.0}
        self.phenomenon = MockPhenomenon(self.config)
    
    def test_initialization(self):
        """Test phenomenon initialization."""
        self.assertEqual(self.phenomenon.config, self.config)
        self.assertEqual(self.phenomenon.state, {})
        self.assertEqual(self.phenomenon.metrics, {})
        self.assertEqual(self.phenomenon.synergies, [])
    
    def test_initialize_method(self):
        """Test initialize method."""
        self.phenomenon.initialize()
        self.assertTrue(self.phenomenon.state['initialized'])
    
    def test_step_method(self):
        """Test step method."""
        self.phenomenon.initialize()
        self.phenomenon.step(0.1)
        self.assertEqual(self.phenomenon.state['time'], 0.1)
        
        self.phenomenon.step(0.1)
        self.assertEqual(self.phenomenon.state['time'], 0.2)
    
    def test_compute_method(self):
        """Test compute method."""
        input_data = np.array([1.0, 2.0, 3.0])
        result = self.phenomenon.compute(input_data)
        expected = np.array([1.1, 2.1, 3.1])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_get_emergent_properties(self):
        """Test get_emergent_properties method."""
        properties = self.phenomenon.get_emergent_properties()
        self.assertEqual(properties['complexity'], 0.5)
    
    def test_add_synergy(self):
        """Test add_synergy method."""
        other_phenomenon = MockPhenomenon({'other': True})
        self.phenomenon.add_synergy(other_phenomenon)
        self.assertEqual(len(self.phenomenon.synergies), 1)
        self.assertEqual(self.phenomenon.synergies[0], other_phenomenon)
    
    def test_measure_performance(self):
        """Test measure_performance method."""
        self.phenomenon.metrics = {'accuracy': 0.9, 'speed': 100}
        performance = self.phenomenon.measure_performance()
        self.assertEqual(performance['accuracy'], 0.9)
        self.assertEqual(performance['speed'], 100)
        
        # Ensure it returns a copy
        performance['accuracy'] = 0.8
        self.assertEqual(self.phenomenon.metrics['accuracy'], 0.9)

if __name__ == '__main__':
    unittest.main()
