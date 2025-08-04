"""Integration tests for synergistic combinations."""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from biocomputing.synergies.dual_phenomenon.programmable_stochastic_mesh import ProgrammableStochasticMesh
from biocomputing.synergies.dual_phenomenon.resource_temporal_load_balancing import ResourceTemporalLoadBalancing

class TestSynergisticCombinations(unittest.TestCase):
    """Test synergistic combinations of phenomena."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'cellular_networks_params': {'network_size': 100},
            'molecular_noise_params': {'amplitude': 0.1},
            'genetic_circuits_params': {'complexity': 5}
        }
    
    def test_programmable_stochastic_mesh_initialization(self):
        """Test programmable stochastic mesh initialization."""
        mesh = ProgrammableStochasticMesh(self.config)
        mesh.initialize()
        
        self.assertTrue(mesh.mesh_state['initialized'])
        self.assertFalse(mesh.mesh_state['programmed'])
    
    def test_programmable_stochastic_mesh_programming(self):
        """Test mesh programming functionality."""
        mesh = ProgrammableStochasticMesh(self.config)
        mesh.initialize()
        
        mesh.self_program("matrix_multiplication")
        
        self.assertTrue(mesh.mesh_state['programmed'])
        self.assertEqual(mesh.mesh_state['target_function'], "matrix_multiplication")
    
    def test_noise_optimization_computation(self):
        """Test computation with noise optimization."""
        mesh = ProgrammableStochasticMesh(self.config)
        mesh.initialize()
        mesh.self_program("optimization_task")
        
        input_data = np.random.randn(10)
        result = mesh.compute_with_noise_optimization(input_data)
        
        self.assertEqual(result.shape, input_data.shape)
        self.assertIsInstance(result, np.ndarray)
    
    def test_noise_benefit_demonstration(self):
        """Test demonstration of noise benefits."""
        mesh = ProgrammableStochasticMesh(self.config)
        mesh.initialize()
        mesh.self_program("test_function")
        
        input_data = np.random.randn(5)
        demonstration = mesh.demonstrate_noise_benefit(input_data)
        
        self.assertIn('no_noise_performance', demonstration)
        self.assertIn('with_noise_performance', demonstration)
        self.assertIn('noise_benefit', demonstration)
        self.assertIn('improvement_ratio', demonstration)
    
    def test_resource_temporal_load_balancing(self):
        """Test resource temporal load balancing."""
        balancer = ResourceTemporalLoadBalancing(self.config)
        balancer.initialize()
        
        tasks = [
            {'id': 'task1', 'resources': {'energy': 10, 'materials': 5}},
            {'id': 'task2', 'resources': {'energy': 15, 'materials': 8}},
            {'id': 'task3', 'resources': {'energy': 20, 'materials': 12}}
        ]
        
        schedule = balancer.auto_schedule_computation(tasks)
        
        self.assertEqual(len(schedule), len(tasks))
        for task in tasks:
            self.assertIn(task['id'], schedule)
            self.assertIn('start_time', schedule[task['id']])
            self.assertIn('resource_allocation', schedule[task['id']])
    
    def test_energy_efficiency_demonstration(self):
        """Test energy efficiency demonstration."""
        balancer = ResourceTemporalLoadBalancing(self.config)
        balancer.initialize()
        
        tasks = [
            {'id': 'task1', 'resources': {'energy': 30}},
            {'id': 'task2
