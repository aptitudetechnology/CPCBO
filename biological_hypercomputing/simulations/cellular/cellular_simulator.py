"""Cellular-scale simulation framework."""

import numpy as np
from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from biocomputing.phenomena.cellular_networks.cellular_networks_core import CellularNetworksCore
from biocomputing.phenomena.cellular_metabolism.cellular_metabolism_core import CellularMetabolismCore

class CellularSimulator:
    """Simulate cellular-scale biological computing processes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cellular_networks = CellularNetworksCore(config)
        self.cellular_metabolism = CellularMetabolismCore(config)
        self.cells = []
        self.network_connections = []
    
    def initialize_cellular_population(self, num_cells: int) -> None:
        """Initialize population of cells."""
        self.cells = []
        for i in range(num_cells):
            cell = {
                'id': i,
                'position': np.random.randn(2),
                'energy': np.random.uniform(50, 100),
                'state': 'healthy',
                'connections': [],
                'metabolic_rate': np.random.uniform(0.8, 1.2)
            }
            self.cells.append(cell)
        
        # Establish network connections
        self._establish_cellular_network()
    
    def simulate_cellular_dynamics(self, steps: int, dt: float = 0.1) -> Dict[str, Any]:
        """Simulate cellular dynamics and network interactions."""
        population_history = []
        
        for step in range(steps):
            # Update cellular metabolism
            self._update_cellular_metabolism(dt)
            
            # Process cellular network communications
            self._process_network_communications()
            
            # Handle cell division and death
            self._handle_cell_lifecycle()
            
            # Record population state
            if step % 10 == 0:
                population_history.append(self._capture_population_state())
        
        return {
            'population_history': population_history,
            'final_population': len(self.cells),
            'simulation_steps': steps
        }
    
    def _establish_cellular_network(self) -> None:
        """Establish network connections between cells."""
        self.network_connections = []
        
        for i, cell1 in enumerate(self.cells):
            for j, cell2 in enumerate(self.cells[i+1:], i+1):
                distance = np.linalg.norm(cell1['position'] - cell2['position'])
                
                if distance < 2.0:  # Connection threshold
                    connection = {
                        'cell1_id': cell1['id'],
                        'cell2_id': cell2['id'],
                        'strength': 1.0 / (1.0 + distance),
                        'active': True
                    }
                    self.network_connections.append(connection)
                    cell1['connections'].append(cell2['id'])
                    cell2['connections'].append(cell1['id'])
    
    def _update_cellular_metabolism(self, dt: float) -> None:
        """Update metabolic state of all cells."""
        for cell in self.cells:
            # Energy consumption based on metabolic rate
            energy_consumption = cell['metabolic_rate'] * dt
            cell['energy'] -= energy_consumption
            
            # Energy regeneration (simplified)
            if cell['energy'] < 80:
                cell['energy'] += 0.5 * dt
            
            # Update cell state based on energy
            if cell['energy'] < 20:
                cell['state'] = 'stressed'
            elif cell['energy'] < 10:
                cell['state'] = 'dying'
            else:
                cell['state'] = 'healthy'
    
    def _process_network_communications(self) -> None:
        """Process communications through cellular network."""
        for connection in self.network_connections:
            if not connection['active']:
                continue
            
            cell1 = self.cells[connection['cell1_id']]
            cell2 = self.cells[connection['cell2_id']]
            
            # Simple communication: share energy if needed
            if cell1['energy'] > 70 and cell2['energy'] < 30:
                energy_transfer = min(10, cell1['energy'] - 50)
                cell1['energy'] -= energy_transfer
                cell2['energy'] += energy_transfer
            elif cell2['energy'] > 70 and cell1['energy'] < 30:
                energy_transfer = min(10, cell2['energy'] - 50)
                cell2['energy'] -= energy_transfer
                cell1['energy'] += energy_transfer
    
    def _handle_cell_lifecycle(self) -> None:
        """Handle cell division and death."""
        cells_to_remove = []
        cells_to_add = []
        
        for i, cell in enumerate(self.cells):
            # Cell death
            if cell['energy'] <= 0:
                cells_to_remove.append(i)
            
            # Cell division
            elif cell['energy'] > 120 and cell['state'] == 'healthy':
                # Create daughter cell
                daughter_cell = {
                    'id': len(self.cells) + len(cells_to_add),
                    'position': cell['position'] + np.random.randn(2) * 0.1,
                    'energy': cell['energy'] / 2,
                    'state': 'healthy',
                    'connections': [],
                    'metabolic_rate': cell['metabolic_rate'] * (1 + np.random.normal(0, 0.1))
                }
                cells_to_add.append(daughter_cell)
                cell['energy'] /= 2  # Parent cell loses half energy
        
        # Remove dead cells
        for i in reversed(cells_to_remove):
            del self.cells[i]
        
        # Add new cells
        self.cells.extend(cells_to_add)
        
        # Update network if population changed
        if cells_to_remove or cells_to_add:
            self._establish_cellular_network()
    
    def _capture_population_state(self) -> Dict[str, Any]:
        """Capture current population state."""
        total_energy = sum(cell['energy'] for cell in self.cells)
        healthy_cells = sum(1 for cell in self.cells if cell['state'] == 'healthy')
        stressed_cells = sum(1 for cell in self.cells if cell['state'] == 'stressed')
        
        return {
            'population_size': len(self.cells),
            'total_energy': total_energy,
            'average_energy': total_energy / len(self.cells) if self.cells else 0,
            'healthy_cells': healthy_cells,
            'stressed_cells': stressed_cells,
            'network_connections': len(self.network_connections)
        }

if __name__ == "__main__":
    config = {'cellular_networks_params': {'connectivity': 0.1}}
    simulator = CellularSimulator(config)
    simulator.initialize_cellular_population(50)
    results = simulator.simulate_cellular_dynamics(100)
    print(f"Cellular simulation completed: {results['final_population']} final cells")
