"""System tests for biological hypercomputing integration."""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from biocomputing.synergies.full_integration.biological_hypercomputing import BiologicalHypercomputing

class TestBiologicalHypercomputing(unittest.TestCase):
    """Test complete biological hypercomputing system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'cellular_networks_params': {'network_size': 50},
            'molecular_noise_params': {'amplitude': 0.1},
            'genetic_circuits_params': {'complexity': 3},
            'cellular_metabolism_params': {'energy_capacity': 100},
            'multiscale_processes_params': {'scales': 3},
            'self_organization_params': {'organization_strength': 0.5},
            'swarm_intelligence_params': {'swarm_size': 30},
            'evolutionary_adaptation_params': {'mutation_rate': 0.01},
            'quantum_biology_params': {'coherence_time': 1.0},
            'resource_constraints_params': {'total_resources': 1000}
        }
    
    def test_system_initialization(self):
        """Test complete system initialization."""
        hypercomputing = BiologicalHypercomputing(self.config)
        hypercomputing.initialize()
        
        self.assertTrue(hypercomputing.hypercomputing_state['initialized'])
        self.assertEqual(len(hypercomputing.phenomena), 10)  # All 10 phenomena
        
        # Check that all phenomena are initialized
        for phenomenon in hypercomputing.phenomena.values():
            self.assertIsNotNone(phenomenon.state)
    
    def test_transcend_digital_limits(self):
        """Test transcendence of digital computing limits."""
        hypercomputing = BiologicalHypercomputing(self.config)
        
        problem = {
            'name': 'optimization_problem',
            'size': 100,
            'complexity': 'high'
        }
        
        result = hypercomputing.transcend_digital_limits(problem)
        
        self.assertIn('solution', result)
        self.assertIn('emergent_properties', result)
        self.assertIn('transcendence_metrics', result)
        self.assertIn('phenomena_contributions', result)
        self.assertIn('biological_advantages', result)
        
        # Check transcendence metrics
        metrics = result['transcendence_metrics']
        self.assertIn('parallel_efficiency', metrics)
        self.assertIn('noise_benefit_ratio', metrics)
        self.assertIn('resource_efficiency', metrics)
        self.assertIn('architectural_adaptability', metrics)
        self.assertIn('quantum_speedup', metrics)
        self.assertIn('emergent_intelligence_gain', metrics)
    
    def test_continuous_evolution(self):
        """Test continuous system evolution."""
        hypercomputing = BiologicalHypercomputing(self.config)
        hypercomputing.initialize()
        
        evolution_result = hypercomputing.continuous_evolution(generations=5)
        
        self.assertIn('evolution_history', evolution_result)
        self.assertIn('final_performance', evolution_result)
        self.assertIn('performance_improvement', evolution_result)
        self.assertIn('evolved_capabilities', evolution_result)
        
        # Check evolution history
        history = evolution_result['evolution_history']
        self.assertEqual(len(history), 5)
        
        for generation_data in history:
            self.assertIn('generation', generation_data)
            self.assertIn('performance', generation_data)
            self.assertIn('synergies', generation_data)
    
    def test_biological_advantages_identification(self):
        """Test identification of biological advantages."""
        hypercomputing = BiologicalHypercomputing(self.config)
        
        test_metrics = {
            'parallel_efficiency': 0.95,
            'noise_benefit_ratio': 1.3,
            'resource_efficiency': 0.85,
            'architectural_adaptability': 0.9,
            'quantum_speedup': 2.5,
            'emergent_intelligence_gain': 0.4
        }
        
        advantages = hypercomputing._identify_biological_advantages(test_metrics)
        
        expected_advantages = [
            "Noise improves performance instead of degrading it",
            "Near-perfect parallel scaling",
            "Automatic resource optimization",
            "Self-modifying architecture during runtime",
            "Quantum acceleration in warm, noisy environments",
            "Genuine emergence of new computational capabilities"
        ]
        
        for expected in expected_advantages:
            self.assertIn(expected, advantages)
    
    def test_phenomena_integration(self):
        """Test integration between all phenomena."""
        hypercomputing = BiologicalHypercomputing(self.config)
        hypercomputing.initialize()
        
        # Test that synergies are detected between phenomena
        synergies = hypercomputing.synergy_manager.detect_synergies()
        self.assertGreater(len(synergies), 0)
        
        # Test that scale coordination works
        test_state = {'test': 'data'}
        coordinated = hypercomputing.scale_coordinator.coordinate_scales(test_state)
        self.assertIsInstance(coordinated, dict)
        
        # Test emergence detection
        system_state = {'complexity': 1.0}
        component_states = [{'local_complexity': 0.1} for _ in range(5)]
        emergence = hypercomputing.emergence_detector.detect_emergence(system_state, component_states)
        self.assertIsInstance(emergence, dict)

if __name__ == '__main__':
    unittest.main()
