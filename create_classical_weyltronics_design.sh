#!/bin/bash

# create_classical_weyltronics_design.sh - Classical computing framework for Weyltronics design
# Uses conventional computational methods to design, simulate, and optimize Weyltronic systems

echo "Creating Classical Weyltronics Design Framework..."

# Create classical design framework directories
mkdir -p src/biocomputing/weyltronics/classical_design
mkdir -p src/biocomputing/weyltronics/classical_design/simulation
mkdir -p src/biocomputing/weyltronics/classical_design/optimization
mkdir -p src/biocomputing/weyltronics/classical_design/fabrication
mkdir -p src/biocomputing/weyltronics/classical_design/verification
mkdir -p src/biocomputing/weyltronics/classical_design/cad_tools

# Create classical simulation engine
cat > src/biocomputing/weyltronics/classical_design/simulation/classical_weyl_simulator.py << 'EOF'
"""
Classical simulation engine for Weyltronic system design.

Uses conventional numerical methods to simulate Weyl semimetal physics,
enabling practical design and optimization of Weyltronic computing systems.
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

@dataclass
class ClassicalWeylParams:
    """Parameters for classical Weyl semimetal simulation."""
    lattice_constant: float = 1.0
    hopping_strength: float = 1.0
    weyl_separation: float = 0.1
    disorder_strength: float = 0.0
    temperature: float = 300.0
    chemical_potential: float = 0.0
    
@dataclass
class SimulationGrid:
    """Computational grid for classical simulation."""
    nx: int = 100
    ny: int = 100
    nz: int = 100
    k_resolution: int = 50
    energy_resolution: int = 200

class ClassicalWeylSimulator:
    """
    Classical computational engine for Weyl semimetal simulation.
    
    Uses standard numerical linear algebra and finite element methods
    to simulate topological properties and quantum transport.
    """
    
    def __init__(self, params: ClassicalWeylParams, grid: SimulationGrid):
        self.params = params
        self.grid = grid
        self.hamiltonian_cache = {}
        self.eigensystem_cache = {}
        
    def build_tight_binding_hamiltonian(self, k_point: np.ndarray) -> np.ndarray:
        """Build tight-binding Hamiltonian using classical methods."""
        kx, ky, kz = k_point
        
        # Weyl Hamiltonian in Pauli matrix basis
        # H = vf * (sigma_x * kx + sigma_y * ky + sigma_z * kz) + b * sigma_z
        vf = self.params.hopping_strength
        b = self.params.weyl_separation
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        H = vf * (kx * sigma_x + ky * sigma_y + kz * sigma_z) + b * sigma_z
        
        # Add disorder if specified
        if self.params.disorder_strength > 0:
            disorder = np.random.normal(0, self.params.disorder_strength, (2, 2))
            disorder = (disorder + disorder.conj().T) / 2  # Make Hermitian
            H += disorder
            
        return H
    
    def solve_eigensystem(self, k_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve eigenvalue problem classically."""
        H = self.build_tight_binding_hamiltonian(k_point)
        eigenvalues, eigenvectors = la.eigh(H)
        return eigenvalues, eigenvectors
    
    def calculate_band_structure(self, k_path: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate electronic band structure along k-path."""
        print(f"Calculating band structure for {len(k_path)} k-points...")
        
        eigenvalues_list = []
        eigenvectors_list = []
        
        # Parallel computation for efficiency
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results = list(executor.map(self.solve_eigensystem, k_path))
        
        for eigenvals, eigenvecs in results:
            eigenvalues_list.append(eigenvals)
            eigenvectors_list.append(eigenvecs)
        
        return {
            'k_points': k_path,
            'eigenvalues': np.array(eigenvalues_list),
            'eigenvectors': np.array(eigenvectors_list)
        }
    
    def calculate_berry_curvature_classical(self, k_point: np.ndarray, 
                                          band_index: int, dk: float = 1e-4) -> np.ndarray:
        """Calculate Berry curvature using finite differences."""
        # Finite difference approach for Berry curvature
        kx, ky, kz = k_point
        
        # Get wavefunctions at neighboring points
        _, psi_0 = self.solve_eigensystem(k_point)
        _, psi_x = self.solve_eigensystem(np.array([kx + dk, ky, kz]))
        _, psi_y = self.solve_eigensystem(np.array([kx, ky + dk, kz]))
        _, psi_z = self.solve_eigensystem(np.array([kx, ky, kz + dk]))
        
        # Extract relevant band
        psi_0_n = psi_0[:, band_index]
        psi_x_n = psi_x[:, band_index]
        psi_y_n = psi_y[:, band_index]
        psi_z_n = psi_z[:, band_index]
        
        # Berry curvature components (simplified)
        omega_x = np.imag(np.conj(psi_0_n) @ (psi_y_n - psi_z_n)) / dk
        omega_y = np.imag(np.conj(psi_0_n) @ (psi_z_n - psi_x_n)) / dk
        omega_z = np.imag(np.conj(psi_0_n) @ (psi_x_n - psi_y_n)) / dk
        
        return np.array([omega_x, omega_y, omega_z])
    
    def calculate_chern_number_classical(self, surface_normal: str = 'z') -> int:
        """Calculate Chern number using classical numerical integration."""
        print(f"Calculating Chern number for {surface_normal}-surface...")
        
        # Create 2D k-space grid for surface
        if surface_normal == 'z':
            kx_range = np.linspace(-np.pi, np.pi, self.grid.k_resolution)
            ky_range = np.linspace(-np.pi, np.pi, self.grid.k_resolution)
            KX, KY = np.meshgrid(kx_range, ky_range)
            k_surface_points = np.stack([KX.flatten(), KY.flatten(), 
                                       np.zeros_like(KX.flatten())], axis=1)
        
        # Calculate Berry curvature over surface
        berry_curvature_sum = 0.0
        
        for k_point in k_surface_points:
            omega = self.calculate_berry_curvature_classical(k_point, band_index=0)
            if surface_normal == 'z':
                berry_curvature_sum += omega[2]
        
        # Integrate over Brillouin zone
        dk = 2 * np.pi / self.grid.k_resolution
        chern_number = berry_curvature_sum * dk**2 / (2 * np.pi)
        
        return int(np.round(chern_number))
    
    def simulate_quantum_transport_classical(self, voltage: float, 
                                           magnetic_field: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Simulate quantum transport using classical Green's function methods."""
        print("Simulating quantum transport...")
        
        # Simplified transport calculation
        # In practice, this would use sophisticated Green's function techniques
        
        base_conductance = 2.0  # 2e^2/h (quantum of conductance)
        
        # Temperature broadening
        kT = 8.617e-5 * self.params.temperature  # eV
        fermi_factor = 1.0 / (1.0 + np.exp(voltage / kT))
        
        conductance = base_conductance * fermi_factor
        
        # Magnetic field effects (chiral anomaly)
        if magnetic_field is not None:
            B_magnitude = np.linalg.norm(magnetic_field)
            # Simplified chiral anomaly enhancement
            anomaly_factor = 1.0 + 0.1 * B_magnitude * np.exp(-B_magnitude)
            conductance *= anomaly_factor
        
        # Disorder effects
        disorder_factor = np.exp(-self.params.disorder_strength)
        conductance *= disorder_factor
        
        return {
            'conductance': conductance,
            'resistance': 1.0 / conductance if conductance > 0 else float('inf'),
            'current': conductance * voltage
        }
    
    def optimize_for_target_properties(self, target_properties: Dict[str, float]) -> ClassicalWeylParams:
        """Use classical optimization to find parameters for target properties."""
        print("Optimizing Weyl parameters for target properties...")
        
        def objective_function(params_array):
            # Unpack parameters
            params = ClassicalWeylParams(
                lattice_constant=params_array[0],
                hopping_strength=params_array[1],
                weyl_separation=params_array[2],
                disorder_strength=params_array[3]
            )
            
            # Create temporary simulator
            temp_sim = ClassicalWeylSimulator(params, self.grid)
            
            # Calculate properties
            try:
                # Simplified property calculation for optimization
                transport_result = temp_sim.simulate_quantum_transport_classical(0.1)
                conductance = transport_result['conductance']
                
                # Define cost function
                cost = 0.0
                if 'conductance' in target_properties:
                    cost += (conductance - target_properties['conductance'])**2
                
                # Add constraints
                if params.disorder_strength < 0:
                    cost += 1000  # Penalty for negative disorder
                if params.hopping_strength <= 0:
                    cost += 1000  # Penalty for non-positive hopping
                    
                return cost
                
            except Exception as e:
                print(f"Optimization error: {e}")
                return 1000  # High cost for failed evaluations
        
        # Initial guess
        x0 = [self.params.lattice_constant, self.params.hopping_strength, 
              self.params.weyl_separation, self.params.disorder_strength]
        
        # Bounds for parameters
        bounds = [(0.5, 2.0),    # lattice_constant
                  (0.1, 5.0),    # hopping_strength  
                  (0.01, 0.5),   # weyl_separation
                  (0.0, 0.2)]    # disorder_strength
        
        # Optimize
        result = minimize(objective_function, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimal_params = ClassicalWeylParams(
                lattice_constant=result.x[0],
                hopping_strength=result.x[1],
                weyl_separation=result.x[2],
                disorder_strength=result.x[3]
            )
            print(f"Optimization successful! Final cost: {result.fun:.6f}")
            return optimal_params
        else:
            print(f"Optimization failed: {result.message}")
            return self.params

class ClassicalWeylDesignSuite:
    """Complete classical design suite for Weyltronic systems."""
    
    def __init__(self):
        self.simulators = {}
        self.design_database = {}
        
    def create_design_configuration(self, config_name: str, 
                                   target_specs: Dict[str, float]) -> Dict[str, any]:
        """Create a complete design configuration using classical methods."""
        print(f"Creating design configuration: {config_name}")
        
        # Set up simulation parameters
        params = ClassicalWeylParams(
            lattice_constant=1.0,
            hopping_strength=2.0,
            weyl_separation=0.1,
            temperature=300.0
        )
        
        grid = SimulationGrid(nx=50, ny=50, nz=50, k_resolution=30)
        
        # Create simulator
        simulator = ClassicalWeylSimulator(params, grid)
        
        # Optimize for target specifications
        optimized_params = simulator.optimize_for_target_properties(target_specs)
        
        # Generate complete design
        design_config = {
            'name': config_name,
            'parameters': optimized_params,
            'target_specs': target_specs,
            'simulation_grid': grid,
            'estimated_performance': self._estimate_performance(optimized_params),
            'fabrication_requirements': self._generate_fabrication_requirements(optimized_params),
            'verification_tests': self._generate_verification_tests(optimized_params)
        }
        
        self.design_database[config_name] = design_config
        return design_config
    
    def _estimate_performance(self, params: ClassicalWeylParams) -> Dict[str, float]:
        """Estimate performance characteristics."""
        return {
            'max_operating_frequency': 1e12 * params.hopping_strength,  # Hz
            'energy_efficiency': 0.95 * np.exp(-params.disorder_strength),
            'fault_tolerance': 0.99 * (1.0 - params.disorder_strength),
            'thermal_stability': 400.0 - 100 * params.disorder_strength  # K
        }
    
    def _generate_fabrication_requirements(self, params: ClassicalWeylParams) -> Dict[str, any]:
        """Generate fabrication requirements."""
        return {
            'material_purity': f">{99.9 - params.disorder_strength * 100:.1f}%",
            'lattice_precision': f"±{params.lattice_constant * 0.01:.3f} Å",
            'processing_temperature': f"{200 + params.hopping_strength * 50:.0f} K",
            'vacuum_level': f"<{1e-8 * (1 + params.disorder_strength):.2e} Torr"
        }
    
    def _generate_verification_tests(self, params: ClassicalWeylParams) -> List[str]:
        """Generate verification test procedures."""
        return [
            "Measure Hall conductivity vs magnetic field",
            "Verify topological surface states with ARPES",
            "Test transport at various temperatures",
            f"Confirm Weyl node separation of {params.weyl_separation:.3f} 1/Å",
            "Validate chiral anomaly signature",
            "Check disorder tolerance limits"
        ]
    
    def generate_design_report(self, config_name: str) -> str:
        """Generate comprehensive design report."""
        if config_name not in self.design_database:
            return f"Design configuration '{config_name}' not found."
        
        config = self.design_database[config_name]
        params = config['parameters']
        performance = config['estimated_performance']
        
        report = f"""
CLASSICAL WEYLTRONICS DESIGN REPORT
===================================
Configuration: {config['name']}

DESIGN PARAMETERS:
- Lattice constant: {params.lattice_constant:.3f} Å
- Hopping strength: {params.hopping_strength:.3f} eV
- Weyl separation: {params.weyl_separation:.3f} 1/Å
- Disorder strength: {params.disorder_strength:.3f}
- Operating temperature: {params.temperature:.0f} K

PERFORMANCE ESTIMATES:
- Max frequency: {performance['max_operating_frequency']:.2e} Hz
- Energy efficiency: {performance['energy_efficiency']:.1%}
- Fault tolerance: {performance['fault_tolerance']:.1%}
- Thermal stability: {performance['thermal_stability']:.0f} K

FABRICATION REQUIREMENTS:
- Material purity: {config['fabrication_requirements']['material_purity']}
- Lattice precision: {config['fabrication_requirements']['lattice_precision']}
- Processing temp: {config['fabrication_requirements']['processing_temperature']}
- Vacuum level: {config['fabrication_requirements']['vacuum_level']}

VERIFICATION TESTS:
"""
        for i, test in enumerate(config['verification_tests'], 1):
            report += f"{i}. {test}\n"
        
        return report
EOF

# Create classical optimization engine
cat > src/biocomputing/weyltronics/classical_design/optimization/design_optimizer.py << 'EOF'
"""
Classical optimization engine for Weyltronic system design.

Uses conventional optimization algorithms to find optimal design parameters
for target performance specifications.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import scipy.optimize as opt
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import json

@dataclass
class DesignObjective:
    """Design objective for optimization."""
    name: str
    target_value: float
    weight: float = 1.0
    tolerance: float = 0.1
    constraint_type: str = 'equality'  # 'equality', 'inequality', 'range'

@dataclass
class DesignVariable:
    """Design variable with bounds."""
    name: str
    lower_bound: float
    upper_bound: float
    initial_value: float
    units: str = ""

class ClassicalDesignOptimizer:
    """
    Classical optimization engine for Weyltronic designs.
    
    Supports multiple optimization algorithms and multi-objective optimization
    for finding optimal design parameters.
    """
    
    def __init__(self):
        self.objectives = []
        self.variables = []
        self.constraints = []
        self.optimization_history = []
        
    def add_objective(self, objective: DesignObjective):
        """Add design objective."""
        self.objectives.append(objective)
        
    def add_variable(self, variable: DesignVariable):
        """Add design variable.""" 
        self.variables.append(variable)
        
    def add_constraint(self, constraint_func: Callable, constraint_type: str = 'ineq'):
        """Add design constraint."""
        self.constraints.append({'type': constraint_type, 'fun': constraint_func})
    
    def objective_function(self, x: np.ndarray, simulator_factory: Callable) -> float:
        """
        Multi-objective function for optimization.
        
        Args:
            x: Design variable values
            simulator_factory: Function that creates simulator with given parameters
        """
        try:
            # Create parameter dictionary
            params = {}
            for i, var in enumerate(self.variables):
                params[var.name] = x[i]
            
            # Create simulator with these parameters
            simulator = simulator_factory(params)
            
            # Calculate objectives
            total_cost = 0.0
            
            for obj in self.objectives:
                # Get property value from simulator
                property_value = self._evaluate_property(simulator, obj.name)
                
                # Calculate cost based on objective type
                if obj.constraint_type == 'equality':
                    cost = obj.weight * ((property_value - obj.target_value) / obj.target_value)**2
                elif obj.constraint_type == 'inequality':
                    if property_value < obj.target_value:
                        cost = obj.weight * ((obj.target_value - property_value) / obj.target_value)**2
                    else:
                        cost = 0.0
                elif obj.constraint_type == 'range':
                    if abs(property_value - obj.target_value) > obj.tolerance * obj.target_value:
                        cost = obj.weight * ((abs(property_value - obj.target_value) - 
                                            obj.tolerance * obj.target_value) / obj.target_value)**2
                    else:
                        cost = 0.0
                
                total_cost += cost
            
            # Store evaluation in history
            self.optimization_history.append({
                'parameters': params.copy(),
                'cost': total_cost,
                'objectives': {obj.name: self._evaluate_property(simulator, obj.name) 
                             for obj in self.objectives}
            })
            
            return total_cost
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 1e6  # High penalty for failed evaluations
    
    def _evaluate_property(self, simulator, property_name: str) -> float:
        """Evaluate a specific property using the simulator."""
        if property_name == 'conductance':
            result = simulator.simulate_quantum_transport_classical(0.1)
            return result['conductance']
        elif property_name == 'chern_number':
            return abs(simulator.calculate_chern_number_classical())
        elif property_name == 'energy_gap':
            # Simplified energy gap calculation
            k_point = np.array([0, 0, 0])
            eigenvals, _ = simulator.solve_eigensystem(k_point)
            return abs(eigenvals[1] - eigenvals[0])
        else:
            # Default property evaluation
            return 1.0
    
    def optimize_genetic_algorithm(self, simulator_factory: Callable, 
                                 generations: int = 100, population_size: int = 50) -> Dict:
        """Optimize using genetic algorithm (differential evolution)."""
        print(f"Running genetic algorithm optimization...")
        print(f"Generations: {generations}, Population: {population_size}")
        
        # Set up bounds
        bounds = [(var.lower_bound, var.upper_bound) for var in self.variables]
        
        # Objective function wrapper
        def obj_func(x):
            return self.objective_function(x, simulator_factory)
        
        # Run differential evolution
        result = differential_evolution(
            obj_func, 
            bounds, 
            maxiter=generations,
            popsize=population_size,
            seed=42,
            workers=1,  # Single worker to avoid pickling issues
            updating='immediate'
        )
        
        return self._format_optimization_result(result, 'genetic_algorithm')
    
    def optimize_gradient_based(self, simulator_factory: Callable, 
                               method: str = 'L-BFGS-B') -> Dict:
        """Optimize using gradient-based methods."""
        print(f"Running gradient-based optimization ({method})...")
        
        # Initial guess
        x0 = [var.initial_value for var in self.variables]
        bounds = [(var.lower_bound, var.upper_bound) for var in self.variables]
        
        # Objective function wrapper
        def obj_func(x):
            return self.objective_function(x, simulator_factory)
        
        # Run optimization
        result = minimize(
            obj_func,
            x0,
            method=method,
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        return self._format_optimization_result(result, method)
    
    def optimize_simulated_annealing(self, simulator_factory: Callable,
                                   n_iter: int = 1000) -> Dict:
        """Optimize using simulated annealing (basin hopping)."""
        print(f"Running simulated annealing optimization...")
        
        # Initial guess
        x0 = [var.initial_value for var in self.variables]
        bounds = [(var.lower_bound, var.upper_bound) for var in self.variables]
        
        # Objective function wrapper
        def obj_func(x):
            return self.objective_function(x, simulator_factory)
        
        # Minimizer for each basin hop
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        
        # Run basin hopping
        result = basinhopping(
            obj_func,
            x0,
            niter=n_iter,
            minimizer_kwargs=minimizer_kwargs,
            seed=42
        )
        
        return self._format_optimization_result(result, 'simulated_annealing')
    
    def multi_algorithm_optimization(self, simulator_factory: Callable) -> Dict:
        """Run multiple optimization algorithms and compare results."""
        print("Running multi-algorithm optimization comparison...")
        
        results = {}
        
        # Genetic Algorithm
        try:
            results['genetic'] = self.optimize_genetic_algorithm(
                simulator_factory, generations=50, population_size=20
            )
        except Exception as e:
            print(f"Genetic algorithm failed: {e}")
            results['genetic'] = None
        
        # Gradient-based
        try:
            results['gradient'] = self.optimize_gradient_based(simulator_factory)
        except Exception as e:
            print(f"Gradient-based optimization failed: {e}")
            results['gradient'] = None
        
        # Simulated Annealing
        try:
            results['annealing'] = self.optimize_simulated_annealing(
                simulator_factory, n_iter=100
            )
        except Exception as e:
            print(f"Simulated annealing failed: {e}")
            results['annealing'] = None
        
        # Find best result
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_algorithm = min(valid_results.keys(), 
                               key=lambda k: valid_results[k]['final_cost'])
            results['best'] = {
                'algorithm': best_algorithm,
                'result': valid_results[best_algorithm]
            }
        
        return results
    
    def _format_optimization_result(self, result, algorithm_name: str) -> Dict:
        """Format optimization result for consistent output."""
        if hasattr(result, 'x'):
            optimal_params = {}
            for i, var in enumerate(self.variables):
                optimal_params[var.name] = result.x[i]
            
            return {
                'algorithm': algorithm_name,
                'success': result.success if hasattr(result, 'success') else True,
                'final_cost': result.fun if hasattr(result, 'fun') else float('inf'),
                'optimal_parameters': optimal_params,
                'iterations': result.nit if hasattr(result, 'nit') else 0,
                'message': result.message if hasattr(result, 'message') else 'Completed'
            }
        else:
            return {
                'algorithm': algorithm_name,
                'success': False,
                'final_cost': float('inf'),
                'optimal_parameters': {},
                'iterations': 0,
                'message': 'Optimization failed'
            }
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if not self.optimization_history:
            print("No optimization history to plot.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cost evolution
        costs = [entry['cost'] for entry in self.optimization_history]
        axes[0, 0].plot(costs, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Evaluation')
        axes[0, 0].set_ylabel('Cost')
        axes[0, 0].set_title('Optimization Cost Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Parameter evolution
        if self.variables:
            param_name = self.variables[0].name
            param_values = [entry['parameters'][param_name] for entry in self.optimization_history]
            axes[0, 1].plot(param_values, 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Evaluation')
            axes[0, 1].set_ylabel(param_name)
            axes[0, 1].set_title(f'Parameter Evolution: {param_name}')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Objective values
        if self.objectives:
            obj_name = self.objectives[0].name
            obj_values = [entry['objectives'][obj_name] for entry in self.optimization_history]
            axes[1, 0].plot(obj_values, 'g-', linewidth=2)
            axes[1, 0].axhline(y=self.objectives[0].target_value, color='r', 
                              linestyle='--', label='Target')
            axes[1, 0].set_xlabel('Evaluation')
            axes[1, 0].set_ylabel(obj_name)
            axes[1, 0].set_title(f'Objective Evolution: {obj_name}')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Cost histogram
        axes[1, 1].hist(costs, bins=30, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Cost')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Cost Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def save_optimization_results(self, results: Dict, filename: str):
        """Save optimization results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Optimization results saved to {filename}")

def create_simulator_factory(params: Dict) -> Callable:
    """Factory function to create simulator with given parameters."""
    def factory():
        return ClassicalWeylSimulator(
            ClassicalWeylParams(**params),
            SimulationGrid(nx=50, ny=50, nz=50, k_resolution=30)
        )
    return factory  
    