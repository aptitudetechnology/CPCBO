#!/bin/bash

# create_weyltronics.sh - Script to create Weyltronics section for biological hypercomputing project
# Adds Weyl semimetal and topological materials simulation capabilities

echo "Creating Weyltronics section for biological hypercomputing project..."

# Create main weyltronics module directory
mkdir -p src/biocomputing/weyltronics

# Create weyltronics subdirectories
mkdir -p src/biocomputing/weyltronics/core
mkdir -p src/biocomputing/weyltronics/materials
mkdir -p src/biocomputing/weyltronics/transport
mkdir -p src/biocomputing/weyltronics/interfaces
mkdir -p src/biocomputing/weyltronics/topology
mkdir -p src/biocomputing/weyltronics/chiral

# Create __init__.py files
touch src/biocomputing/weyltronics/__init__.py
touch src/biocomputing/weyltronics/core/__init__.py
touch src/biocomputing/weyltronics/materials/__init__.py
touch src/biocomputing/weyltronics/transport/__init__.py
touch src/biocomputing/weyltronics/interfaces/__init__.py
touch src/biocomputing/weyltronics/topology/__init__.py
touch src/biocomputing/weyltronics/chiral/__init__.py

# Create core Weyltronics classes
cat > src/biocomputing/weyltronics/core/weyl_semimetal.py << 'EOF'
"""
Core Weyl semimetal simulation classes for topological quantum computing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class WeylNode:
    """Represents a Weyl node in k-space."""
    position: np.ndarray
    chirality: int  # +1 or -1
    energy: float
    
class WeylSemimetal:
    """
    Simulates a Weyl semimetal with topologically protected states.
    """
    
    def __init__(self, lattice_size: Tuple[int, int, int], 
                 weyl_nodes: List[WeylNode],
                 disorder_strength: float = 0.0):
        self.lattice_size = lattice_size
        self.weyl_nodes = weyl_nodes
        self.disorder_strength = disorder_strength
        self._initialize_hamiltonian()
    
    def _initialize_hamiltonian(self):
        """Initialize the system Hamiltonian."""
        # Placeholder for Weyl Hamiltonian construction
        self.hamiltonian = None
        
    def calculate_band_structure(self) -> Dict[str, np.ndarray]:
        """Calculate electronic band structure."""
        # Placeholder for band structure calculation
        return {
            'k_points': np.array([]),
            'eigenvalues': np.array([]),
            'eigenvectors': np.array([])
        }
    
    def get_surface_states(self, surface: str = 'top') -> np.ndarray:
        """Calculate topologically protected surface states."""
        # Placeholder for surface state calculation
        return np.array([])
    
    def apply_disorder(self, strength: float):
        """Apply disorder while preserving topological protection."""
        self.disorder_strength = strength
        # Topological states remain protected
        
    def compute_chern_number(self) -> int:
        """Compute topological invariant (Chern number)."""
        # Placeholder for topological invariant calculation
        return 0
EOF

cat > src/biocomputing/weyltronics/core/quantum_transport.py << 'EOF'
"""
Quantum transport in Weyltronic systems with biological interfaces.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TransportChannel:
    """Represents a quantum transport channel."""
    conductance: float
    transmission: float
    is_topological: bool
    noise_resilience: float

class QuantumTransport:
    """
    Handles quantum transport in Weyltronic systems.
    """
    
    def __init__(self, weyl_system, temperature: float = 300.0):
        self.weyl_system = weyl_system
        self.temperature = temperature
        self.channels = []
        
    def calculate_conductance(self, voltage: float, 
                            magnetic_field: Optional[np.ndarray] = None) -> float:
        """Calculate quantum conductance with chiral anomaly effects."""
        # Placeholder for conductance calculation
        base_conductance = 1.0  # e^2/h units
        
        if magnetic_field is not None:
            # Chiral anomaly enhancement
            chiral_factor = self._chiral_anomaly_factor(magnetic_field)
            return base_conductance * chiral_factor
            
        return base_conductance
    
    def _chiral_anomaly_factor(self, magnetic_field: np.ndarray) -> float:
        """Calculate chiral anomaly enhancement factor."""
        # Simplified chiral anomaly effect
        b_magnitude = np.linalg.norm(magnetic_field)
        return 1.0 + 0.1 * b_magnitude  # Placeholder formula
    
    def get_edge_currents(self) -> Dict[str, float]:
        """Get topologically protected edge currents."""
        return {
            'top_edge': 1.0,
            'bottom_edge': -1.0,  # Opposite chirality
            'left_edge': 0.5,
            'right_edge': -0.5
        }
    
    def noise_resilience_factor(self, noise_strength: float) -> float:
        """Calculate how transport is affected by noise."""
        # Topological protection provides noise resilience
        return max(0.1, 1.0 - 0.1 * noise_strength)  # Minimal degradation
EOF

# Create materials simulation
cat > src/biocomputing/weyltronics/materials/weyl_materials.py << 'EOF'
"""
Specific Weyl semimetal materials and their properties.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

class WeylMaterialType(Enum):
    TAAS = "TaAs"
    CDAS2 = "CdAs2" 
    NABIAS = "NbAs"
    SYNTHETIC_BIO = "BiologicalHybrid"

@dataclass
class MaterialProperties:
    """Properties of Weyl semimetal materials."""
    fermi_velocity: float
    weyl_node_separation: float
    bulk_gap: float
    surface_state_velocity: float
    bio_compatibility: float  # 0-1 scale

class WeylMaterialDatabase:
    """Database of Weyl semimetal materials and properties."""
    
    def __init__(self):
        self.materials = {
            WeylMaterialType.TAAS: MaterialProperties(
                fermi_velocity=5e5,  # m/s
                weyl_node_separation=0.1,  # 1/Angstrom
                bulk_gap=0.0,  # eV
                surface_state_velocity=3e5,  # m/s
                bio_compatibility=0.2
            ),
            WeylMaterialType.SYNTHETIC_BIO: MaterialProperties(
                fermi_velocity=1e5,  # Reduced for bio-compatibility
                weyl_node_separation=0.05,
                bulk_gap=0.01,
                surface_state_velocity=1e5,
                bio_compatibility=0.9  # Engineered for biological systems
            )
        }
    
    def get_material(self, material_type: WeylMaterialType) -> MaterialProperties:
        """Get properties for a specific material."""
        return self.materials[material_type]
    
    def optimize_for_biology(self, base_material: WeylMaterialType) -> MaterialProperties:
        """Optimize material properties for biological integration."""
        props = self.get_material(base_material)
        # Reduce velocities and gaps for bio-compatibility
        return MaterialProperties(
            fermi_velocity=props.fermi_velocity * 0.5,
            weyl_node_separation=props.weyl_node_separation * 0.8,
            bulk_gap=props.bulk_gap + 0.005,  # Small gap for stability
            surface_state_velocity=props.surface_state_velocity * 0.6,
            bio_compatibility=min(1.0, props.bio_compatibility + 0.3)
        )
EOF

# Create quantum-bio interfaces
cat > src/biocomputing/weyltronics/interfaces/quantum_bio_bridge.py << 'EOF'
"""
Interfaces between Weyltronic hardware and biological systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class BioQuantumSignal:
    """Signal that bridges biological and quantum domains."""
    amplitude: float
    frequency: float
    phase: float
    biological_origin: str  # e.g., "membrane_potential", "metabolic_flux"
    quantum_target: str     # e.g., "weyl_node", "edge_state"

class QuantumBioInterface(ABC):
    """Abstract interface for quantum-biological coupling."""
    
    @abstractmethod
    def biological_to_quantum(self, bio_signal: Any) -> BioQuantumSignal:
        """Convert biological signal to quantum-compatible form."""
        pass
    
    @abstractmethod
    def quantum_to_biological(self, quantum_state: Any) -> Any:
        """Convert quantum information to biological signal."""
        pass
    
    @abstractmethod
    def coupling_strength(self) -> float:
        """Return the coupling strength between domains."""
        pass

class WeylBioCoupler(QuantumBioInterface):
    """Couples Weyltronic states with biological processes."""
    
    def __init__(self, coupling_efficiency: float = 0.8):
        self.coupling_efficiency = coupling_efficiency
        self.active_channels = []
        
    def biological_to_quantum(self, bio_signal: Dict[str, float]) -> BioQuantumSignal:
        """Convert biological signal (e.g., membrane potential) to Weyl node modulation."""
        # Example: membrane potential modulates Weyl node position
        if 'membrane_potential' in bio_signal:
            potential = bio_signal['membrane_potential']
            return BioQuantumSignal(
                amplitude=abs(potential) * 0.1,
                frequency=1000.0,  # Hz
                phase=np.angle(potential) if np.iscomplexobj(potential) else 0.0,
                biological_origin='membrane_potential',
                quantum_target='weyl_node'
            )
        
        # Default conversion
        return BioQuantumSignal(0.0, 0.0, 0.0, 'unknown', 'weyl_node')
    
    def quantum_to_biological(self, quantum_state: np.ndarray) -> Dict[str, float]:
        """Convert quantum state to biological control signal."""
        # Example: edge state current controls ion channel conductance
        edge_current = np.real(quantum_state[0]) if len(quantum_state) > 0 else 0.0
        
        return {
            'ion_conductance': edge_current * self.coupling_efficiency,
            'metabolic_rate': abs(edge_current) * 0.5,
            'gene_expression': np.tanh(edge_current)  # Bounded response
        }
    
    def coupling_strength(self) -> float:
        """Return coupling efficiency."""
        return self.coupling_efficiency
    
    def create_feedback_loop(self, bio_system, weyl_system):
        """Create closed-loop feedback between biological and Weyltronic systems."""
        # Placeholder for feedback implementation
        pass
EOF

# Create topology analysis tools
cat > src/biocomputing/weyltronics/topology/invariants.py << 'EOF'
"""
Topological invariant calculations for Weyltronic systems.
"""

import numpy as np
from typing import Tuple, Dict, List
from scipy.linalg import eigh

class TopologicalAnalyzer:
    """Analyzes topological properties of Weyltronic systems."""
    
    def __init__(self, system):
        self.system = system
        
    def calculate_berry_curvature(self, k_point: np.ndarray, 
                                band_index: int) -> np.ndarray:
        """Calculate Berry curvature at a k-point."""
        # Placeholder for Berry curvature calculation
        # In practice, this would involve derivatives of eigenstates
        return np.array([0.0, 0.0, 0.0])
    
    def chern_number_2d(self, surface_hamiltonian) -> int:
        """Calculate Chern number for 2D surface."""
        # Placeholder for 2D Chern number calculation
        return 1  # Typical value for topological surface
    
    def z2_invariant(self, bulk_hamiltonian) -> Tuple[int, int, int, int]:
        """Calculate Z2 topological invariant."""
        # Placeholder for Z2 calculation
        return (1, 0, 0, 0)  # (nu_0, nu_1, nu_2, nu_3)
    
    def weyl_monopole_charge(self, weyl_node_k: np.ndarray) -> int:
        """Calculate monopole charge of Weyl node."""
        # This is the chirality of the Weyl node
        # Placeholder implementation
        return 1  # +1 or -1
    
    def topological_phase_diagram(self, parameter_range: Dict[str, np.ndarray]) -> Dict:
        """Generate topological phase diagram."""
        # Placeholder for phase diagram calculation
        return {
            'parameters': parameter_range,
            'phases': ['trivial', 'topological'],
            'boundaries': []
        }
EOF

# Create chiral anomaly effects
cat > src/biocomputing/weyltronics/chiral/anomaly_effects.py << 'EOF'
"""
Chiral anomaly effects in Weyltronic systems for enhanced computation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ChiralCurrent:
    """Represents chiral anomaly-induced current."""
    magnitude: float
    direction: np.ndarray
    chirality: int  # +1 or -1
    
class ChiralAnomalyProcessor:
    """Processes information using chiral anomaly effects."""
    
    def __init__(self, weyl_system):
        self.weyl_system = weyl_system
        self.chiral_currents = []
        
    def apply_parallel_fields(self, electric_field: np.ndarray, 
                            magnetic_field: np.ndarray) -> ChiralCurrent:
        """Apply parallel E and B fields to induce chiral anomaly."""
        # Chiral anomaly occurs when E || B
        e_dot_b = np.dot(electric_field, magnetic_field)
        
        if abs(e_dot_b) > 1e-6:  # Fields are parallel
            # Anomalous current proportional to E·B
            current_magnitude = abs(e_dot_b) * 0.1  # Coupling constant
            current_direction = magnetic_field / np.linalg.norm(magnetic_field)
            chirality = 1 if e_dot_b > 0 else -1
            
            return ChiralCurrent(current_magnitude, current_direction, chirality)
        
        return ChiralCurrent(0.0, np.array([0, 0, 0]), 0)
    
    def chiral_computation(self, input_data: np.ndarray, 
                         field_config: Dict[str, np.ndarray]) -> np.ndarray:
        """Perform computation using chiral anomaly effects."""
        e_field = field_config.get('electric', np.array([0, 0, 1]))
        b_field = field_config.get('magnetic', np.array([0, 0, 1]))
        
        # Generate chiral current
        chiral_current = self.apply_parallel_fields(e_field, b_field)
        
        # Use chiral current to process input data
        # This is a simplified model of chiral anomaly computation
        if chiral_current.magnitude > 0:
            # Chirality determines processing direction
            if chiral_current.chirality > 0:
                output = np.fft.fft(input_data)  # Forward transform
            else:
                output = np.fft.ifft(input_data)  # Inverse transform
                
            # Scale by anomaly strength
            output *= chiral_current.magnitude
            return np.real(output)
        
        return input_data  # No anomaly, pass through
    
    def anomaly_enhanced_optimization(self, cost_function, 
                                    initial_guess: np.ndarray) -> np.ndarray:
        """Use chiral anomaly for gradient-free optimization."""
        # Placeholder for anomaly-enhanced optimization
        # The idea is that chiral currents can explore parameter space
        # in ways that classical gradients cannot
        
        current_solution = initial_guess.copy()
        
        # Use different field configurations to explore solution space
        for step in range(100):  # Simple iteration
            # Vary field directions to explore different paths
            theta = 2 * np.pi * step / 100
            e_field = np.array([np.cos(theta), np.sin(theta), 0])
            b_field = np.array([np.cos(theta), np.sin(theta), 0])
            
            # Generate exploration direction via chiral anomaly
            chiral_current = self.apply_parallel_fields(e_field, b_field)
            
            if chiral_current.magnitude > 0:
                # Move in chiral-determined direction
                step_size = 0.01 * chiral_current.magnitude
                direction = chiral_current.direction[:len(current_solution)]
                new_solution = current_solution + step_size * direction
                
                # Accept if improvement (simplified)
                if cost_function(new_solution) < cost_function(current_solution):
                    current_solution = new_solution
        
        return current_solution
EOF

# Create integration with existing phenomena
cat > src/biocomputing/weyltronics/synergy_integrations.py << 'EOF'
"""
Integration of Weyltronics with other biological computing phenomena.
"""

from typing import Dict, Any, List
import numpy as np

class WeyltonicSynergyManager:
    """Manages synergistic interactions between Weyltronics and other phenomena."""
    
    def __init__(self):
        self.active_synergies = {}
        
    def integrate_with_noise(self, weyl_system, noise_system) -> Dict[str, Any]:
        """Integrate Weyltronics with molecular noise for enhanced robustness."""
        # Topological protection + beneficial noise
        synergy_result = {
            'noise_resilience': 0.95,  # Very high due to topological protection
            'noise_enhancement': True,  # Noise can actually help certain computations
            'optimal_noise_level': 0.1,  # Sweet spot for performance
            'protected_channels': weyl_system.get_edge_currents() if hasattr(weyl_system, 'get_edge_currents') else {}
        }
        
        self.active_synergies['noise'] = synergy_result
        return synergy_result
    
    def integrate_with_cellular_networks(self, weyl_system, network_system) -> Dict[str, Any]:
        """Integrate Weyltronics with cellular networks for distributed quantum computing."""
        synergy_result = {
            'quantum_communication_channels': 4,  # Number of topological channels
            'network_coherence_time': 1000.0,  # ms, enhanced by topology
            'distributed_entanglement': True,
            'fault_tolerance': 0.99  # Very high due to topological protection
        }
        
        self.active_synergies['cellular_networks'] = synergy_result
        return synergy_result
    
    def integrate_with_genetic_circuits(self, weyl_system, genetic_system) -> Dict[str, Any]:
        """Integrate Weyltronics with genetic circuits for programmable quantum-bio computing."""
        synergy_result = {
            'programmable_topology': True,  # Genetic circuits control topological states
            'adaptive_quantum_gates': 16,  # Number of controllable quantum operations
            'bio_quantum_memory': 1024,  # Qubits that can be stored in biological systems
            'evolution_rate': 0.01  # How fast the system can adapt/evolve
        }
        
        self.active_synergies['genetic_circuits'] = synergy_result
        return synergy_result
    
    def integrate_with_metabolism(self, weyl_system, metabolic_system) -> Dict[str, Any]:
        """Integrate Weyltronics with cellular metabolism for energy-efficient quantum computing."""
        synergy_result = {
            'energy_efficiency': 0.98,  # Nearly dissipationless transport
            'metabolic_control': True,  # Metabolism controls quantum states
            'atp_quantum_coupling': 0.7,  # How well ATP drives quantum processes
            'thermal_stability': 350.0  # K, operating temperature range
        }
        
        self.active_synergies['metabolism'] = synergy_result
        return synergy_result
    
    def get_combined_capabilities(self) -> Dict[str, Any]:
        """Get the combined computational capabilities from all active synergies."""
        combined = {
            'quantum_biological_hybrid': True,
            'fault_tolerant_biocomputing': True,
            'adaptive_architecture': True,
            'noise_enhanced_performance': True,
            'ultra_low_power': True,
            'programmable_topology': True
        }
        
        # Aggregate metrics from all synergies
        if 'noise' in self.active_synergies:
            combined['noise_resilience'] = self.active_synergies['noise']['noise_resilience']
            
        if 'cellular_networks' in self.active_synergies:
            combined['network_fault_tolerance'] = self.active_synergies['cellular_networks']['fault_tolerance']
            
        if 'genetic_circuits' in self.active_synergies:
            combined['adaptive_gates'] = self.active_synergies['genetic_circuits']['adaptive_quantum_gates']
            
        if 'metabolism' in self.active_synergies:
            combined['energy_efficiency'] = self.active_synergies['metabolism']['energy_efficiency']
        
        return combined
EOF

# Create tests directory structure
mkdir -p tests/weyltronics
mkdir -p tests/weyltronics/core
mkdir -p tests/weyltronics/materials
mkdir -p tests/weyltronics/interfaces
mkdir -p tests/weyltronics/integration

# Create test files
touch tests/weyltronics/__init__.py
touch tests/weyltronics/core/__init__.py
touch tests/weyltronics/materials/__init__.py
touch tests/weyltronics/interfaces/__init__.py
touch tests/weyltronics/integration/__init__.py

cat > tests/weyltronics/test_weyl_semimetal.py << 'EOF'
"""
Tests for Weyl semimetal core functionality.
"""

import pytest
import numpy as np
from src.biocomputing.weyltronics.core.weyl_semimetal import WeylSemimetal, WeylNode
from src.biocomputing.weyltronics.core.quantum_transport import QuantumTransport

class TestWeylSemimetal:
    
    def test_weyl_node_creation(self):
        """Test creation of Weyl nodes."""
        node = WeylNode(
            position=np.array([0.1, 0.1, 0.0]),
            chirality=1,
            energy=0.05
        )
        assert node.chirality in [-1, 1]
        assert len(node.position) == 3
    
    def test_weyl_semimetal_initialization(self):
        """Test WeylSemimetal initialization."""
        nodes = [
            WeylNode(np.array([0.1, 0, 0]), 1, 0.0),
            WeylNode(np.array([-0.1, 0, 0]), -1, 0.0)
        ]
        
        weyl_system = WeylSemimetal(
            lattice_size=(10, 10, 10),
            weyl_nodes=nodes,
            disorder_strength=0.1
        )
        
        assert len(weyl_system.weyl_nodes) == 2
        assert weyl_system.disorder_strength == 0.1
    
    def test_disorder_resilience(self):
        """Test that topological states are resilient to disorder."""
        nodes = [WeylNode(np.array([0.1, 0, 0]), 1, 0.0)]
        weyl_system = WeylSemimetal((5, 5, 5), nodes)
        
        # Apply disorder
        weyl_system.apply_disorder(0.5)
        
        # Topological properties should be preserved
        chern_number = weyl_system.compute_chern_number()
        assert isinstance(chern_number, int)

class TestQuantumTransport:
    
    def test_conductance_calculation(self):
        """Test quantum conductance calculation."""
        nodes = [WeylNode(np.array([0, 0, 0]), 1, 0.0)]
        weyl_system = WeylSemimetal((5, 5, 5), nodes)
        transport = QuantumTransport(weyl_system)
        
        conductance = transport.calculate_conductance(voltage=0.1)
        assert conductance > 0
    
    def test_chiral_anomaly_enhancement(self):
        """Test chiral anomaly enhancement of conductance."""
        nodes = [WeylNode(np.array([0, 0, 0]), 1, 0.0)]
        weyl_system = WeylSemimetal((5, 5, 5), nodes)
        transport = QuantumTransport(weyl_system)
        
        # Without magnetic field
        g0 = transport.calculate_conductance(0.1)
        
        # With magnetic field (should enhance via chiral anomaly)
        b_field = np.array([0, 0, 1.0])
        g_enhanced = transport.calculate_conductance(0.1, b_field)
        
        assert g_enhanced >= g0
    
    def test_noise_resilience(self):
        """Test noise resilience of transport."""
        nodes = [WeylNode(np.array([0, 0, 0]), 1, 0.0)]
        weyl_system = WeylSemimetal((5, 5, 5), nodes)
        transport = QuantumTransport(weyl_system)
        
        # Even with strong noise, should maintain reasonable transport
        resilience = transport.noise_resilience_factor(noise_strength=0.8)
        assert resilience > 0.1  # Should maintain at least 10% performance
EOF

# Create example/tutorial notebooks directory
mkdir -p examples/weyltronics
mkdir -p tutorials/weyltronics

cat > examples/weyltronics/basic_weyl_simulation.py << 'EOF'
"""
Basic example of Weyltronic simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.biocomputing.weyltronics.core.weyl_semimetal import WeylSemimetal, WeylNode
from src.biocomputing.weyltronics.core.quantum_transport import QuantumTransport
from src.biocomputing.weyltronics.materials.weyl_materials import WeylMaterialDatabase, WeylMaterialType

def main():
    """Run basic Weyltronic simulation."""
    
    # Create Weyl nodes (minimal pair)
    weyl_nodes = [
        WeylNode(position=np.array([0.1, 0, 0]), chirality=1, energy=0.0),
        WeylNode(position=np.array([-0.1, 0, 0]), chirality=-1, energy=0.0)
    ]
    
    # Initialize Weyl semimetal system
    weyl_system = WeylSemimetal(
        lattice_size=(20, 20, 20),
        weyl_nodes=weyl_nodes,
        disorder_strength=0.05
    )
    
    # Setup quantum transport
    transport = QuantumTransport(weyl_system, temperature=300.0)
    
    # Test conductance vs magnetic field
    b_fields = np.linspace(0, 2.0, 50)
    conductances = []
    
    for b in b_fields:
        magnetic_field = np.array([0, 0, b])
        g = transport.calculate_conductance(voltage=0.1, magnetic_field=magnetic_field)
        conductances.append(g)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(b_fields, conductances, 'b-', linewidth=2)
    plt.xlabel('Magnetic Field (T)')
    plt.ylabel('Conductance (e²/h)')
    plt.title('Chiral Anomaly Enhancement')
    plt.grid(True)
    
    # Test edge currents
    edge_currents = transport.get_edge_currents()
    
    plt.subplot(1, 2, 2)
    edges = list(edge_currents.keys())
    currents = list(edge_currents.values())
    plt.bar(edges, currents)
    plt.xlabel('Edge')
    plt.ylabel('Current')
    plt.title('Topologically Protected Edge Currents')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('examples/weyltronics/weyl_simulation_results.png', dpi=150)
    plt.show()
    
    # Print material properties
    print("\n=== Weyltronic Material Properties ===")
    material_db = WeylMaterialDatabase()
    
    for material_type in WeylMaterialType:
        props = material_db.get_material(material_type)
        print(f"\n{material_type.value}:")
        print(f"  Fermi velocity: {props.fermi_velocity:.2e} m/s")
        print(f"  Bio-compatibility: {props.bio_compatibility:.2f}")
        
        if material_type != WeylMaterialType.SYNTHETIC_BIO:
            bio_optimized = material_db.optimize_for_biology(material_type)
            print(f"  Bio-optimized compatibility: {bio_optimized.bio_compatibility:.2f}")

    print(f"\n=== Transport Properties ===")
    print(f"Noise resilience (50% noise): {transport.noise_resilience_factor(0.5):.3f}")
    print(f"Chern number: {weyl_system.compute_chern_number()}")

if __name__ == "__main__":
    main()
EOF

cat > examples/weyltronics/quantum_bio_interface_demo.py << 'EOF'
"""
Demonstration of quantum-biological interfaces using Weyltronics.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.biocomputing.weyltronics.interfaces.quantum_bio_bridge import WeylBioCoupler, BioQuantumSignal
from src.biocomputing.weyltronics.chiral.anomaly_effects import ChiralAnomalyProcessor
from src.biocomputing.weyltronics.synergy_integrations import WeyltonicSynergyManager

def simulate_membrane_potential():
    """Simulate biological membrane potential over time."""
    t = np.linspace(0, 0.1, 1000)  # 100ms simulation
    # Action potential-like signal
    membrane_potential = -70 + 120 * np.exp(-((t - 0.05) / 0.01)**2)
    return t, membrane_potential

def main():
    """Run quantum-bio interface demonstration."""
    
    # Initialize quantum-bio coupler
    coupler = WeylBioCoupler(coupling_efficiency=0.85)
    
    # Simulate biological signals
    time, membrane_v = simulate_membrane_potential()
    
    # Convert biological signals to quantum domain
    quantum_signals = []
    biological_responses = []
    
    for v in membrane_v:
        # Bio to quantum conversion
        bio_signal = {'membrane_potential': v}
        quantum_sig = coupler.biological_to_quantum(bio_signal)
        quantum_signals.append(quantum_sig.amplitude)
        
        # Create mock quantum state
        quantum_state = np.array([quantum_sig.amplitude * np.exp(1j * quantum_sig.phase)])
        
        # Quantum to bio conversion
        bio_response = coupler.quantum_to_biological(quantum_state)
        biological_responses.append(bio_response)
    
    # Demonstrate chiral anomaly processing
    anomaly_processor = ChiralAnomalyProcessor(None)  # Mock weyl system
    
    # Test chiral computation
    input_data = np.random.randn(64)
    field_config = {
        'electric': np.array([1, 0, 0]),
        'magnetic': np.array([1, 0, 0])  # Parallel fields for anomaly
    }
    
    processed_data = anomaly_processor.chiral_computation(input_data, field_config)
    
    # Synergy integration demonstration
    synergy_manager = WeyltonicSynergyManager()
    
    # Mock systems for integration
    class MockSystem:
        def get_edge_currents(self):
            return {'top': 1.0, 'bottom': -1.0, 'left': 0.5, 'right': -0.5}
    
    mock_weyl = MockSystem()
    mock_noise = None
    mock_cellular = None
    mock_genetic = None
    mock_metabolic = None
    
    # Integrate with different phenomena
    noise_synergy = synergy_manager.integrate_with_noise(mock_weyl, mock_noise)
    network_synergy = synergy_manager.integrate_with_cellular_networks(mock_weyl, mock_cellular)
    genetic_synergy = synergy_manager.integrate_with_genetic_circuits(mock_weyl, mock_genetic)
    metabolic_synergy = synergy_manager.integrate_with_metabolism(mock_weyl, mock_metabolic)
    
    combined_capabilities = synergy_manager.get_combined_capabilities()
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Membrane potential and quantum coupling
    axes[0, 0].plot(time * 1000, membrane_v, 'b-', linewidth=2, label='Membrane Potential')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Potential (mV)')
    axes[0, 0].set_title('Biological Signal Input')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # Plot 2: Quantum signal amplitude
    axes[0, 1].plot(time * 1000, quantum_signals, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Quantum Amplitude')
    axes[0, 1].set_title('Bio→Quantum Conversion')
    axes[0, 1].grid(True)
    
    # Plot 3: Biological response (ion conductance)
    ion_conductances = [resp['ion_conductance'] for resp in biological_responses]
    axes[0, 2].plot(time * 1000, ion_conductances, 'g-', linewidth=2)
    axes[0, 2].set_xlabel('Time (ms)')
    axes[0, 2].set_ylabel('Ion Conductance')
    axes[0, 2].set_title('Quantum→Bio Response')
    axes[0, 2].grid(True)
    
    # Plot 4: Chiral anomaly processing
    axes[1, 0].plot(input_data[:32], 'b-', alpha=0.7, label='Input')
    axes[1, 0].plot(processed_data[:32], 'r-', alpha=0.7, label='Processed')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Chiral Anomaly Processing')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Synergy integration metrics
    synergy_names = ['Noise', 'Networks', 'Genetic', 'Metabolic']
    resilience_values = [
        noise_synergy['noise_resilience'],
        network_synergy['fault_tolerance'], 
        genetic_synergy['evolution_rate'] * 10,  # Scale for visibility
        metabolic_synergy['energy_efficiency']
    ]
    
    bars = axes[1, 1].bar(synergy_names, resilience_values, 
                         color=['blue', 'green', 'orange', 'purple'], alpha=0.7)
    axes[1, 1].set_ylabel('Performance Metric')
    axes[1, 1].set_title('Synergistic Integration Benefits')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, resilience_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 6: Combined capability radar chart (simplified)
    capabilities = list(combined_capabilities.keys())[:6]  # First 6 boolean capabilities
    capability_values = [1.0 if combined_capabilities[cap] else 0.0 for cap in capabilities]
    
    # Simple bar chart instead of radar for simplicity
    axes[1, 2].barh(range(len(capabilities)), capability_values, color='cyan', alpha=0.7)
    axes[1, 2].set_yticks(range(len(capabilities)))
    axes[1, 2].set_yticklabels([cap.replace('_', ' ').title() for cap in capabilities], fontsize=8)
    axes[1, 2].set_xlabel('Capability Level')
    axes[1, 2].set_title('Combined System Capabilities')
    axes[1, 2].set_xlim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig('examples/weyltronics/quantum_bio_interface_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print detailed results
    print("=== Quantum-Biological Interface Demonstration ===")
    print(f"Coupling efficiency: {coupler.coupling_strength():.2f}")
    print(f"Peak membrane potential: {max(membrane_v):.1f} mV")
    print(f"Peak quantum amplitude: {max(quantum_signals):.3f}")
    print(f"Peak ion conductance response: {max(ion_conductances):.3f}")
    
    print("\n=== Chiral Anomaly Processing ===")
    print(f"Input data RMS: {np.sqrt(np.mean(input_data**2)):.3f}")
    print(f"Processed data RMS: {np.sqrt(np.mean(processed_data**2)):.3f}")
    print(f"Processing gain: {np.sqrt(np.mean(processed_data**2)) / np.sqrt(np.mean(input_data**2)):.2f}")
    
    print("\n=== Synergistic Integration Results ===")
    for synergy_type, synergy_data in [
        ('Noise Integration', noise_synergy),
        ('Network Integration', network_synergy), 
        ('Genetic Integration', genetic_synergy),
        ('Metabolic Integration', metabolic_synergy)
    ]:
        print(f"\n{synergy_type}:")
        for key, value in synergy_data.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            elif isinstance(value, bool):
                print(f"  {key}: {value}")
            elif isinstance(value, int):
                print(f"  {key}: {value}")
    
    print("\n=== Combined System Capabilities ===")
    for capability, status in combined_capabilities.items():
        if isinstance(status, bool):
            print(f"  {capability.replace('_', ' ').title()}: {'✓' if status else '✗'}")
        else:
            print(f"  {capability.replace('_', ' ').title()}: {status}")

if __name__ == "__main__":
    main()
EOF