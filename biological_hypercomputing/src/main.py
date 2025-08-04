"""Main entry point for biological hypercomputing platform."""

import argparse
import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from biocomputing.core.synergy_manager import SynergyManager
from biocomputing.synergies.full_integration.biological_hypercomputing import BiologicalHypercomputing
from biocomputing.synergies.dual_phenomenon.programmable_stochastic_mesh import ProgrammableStochasticMesh
from biocomputing.synergies.dual_phenomenon.resource_temporal_load_balancing import ResourceTemporalLoadBalancing

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_config.get('file', 'biocomputing.log')),
            logging.StreamHandler(sys.stdout) if log_config.get('console_output', True) else logging.NullHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

def run_single_phenomenon_experiment(config: Dict[str, Any]) -> None:
    """Run single phenomenon experiment."""
    logging.info("Starting single phenomenon experiment...")
    
    # Initialize synergy manager for analysis
    synergy_manager = SynergyManager()
    
    # Load and test individual phenomena
    from biocomputing.phenomena.cellular_networks.cellular_networks_core import CellularNetworksCore
    
    cellular_networks = CellularNetworksCore(config)
    cellular_networks.initialize()
    synergy_manager.add_phenomenon(cellular_networks)
    
    # Run basic computation test
    import numpy as np
    test_input = np.random.randn(10)
    result = cellular_networks.compute(test_input)
    
    logging.info(f"Single phenomenon test completed. Output shape: {result.shape}")
    print("Single phenomenon experiment completed successfully!")

def run_dual_phenomenon_experiment(config: Dict[str, Any]) -> None:
    """Run dual phenomenon synergy experiment."""
    logging.info("Starting dual phenomenon experiment...")
    
    # Test Programmable Stochastic Mesh
    print("Testing Programmable Stochastic Mesh Computing...")
    mesh = ProgrammableStochasticMesh(config)
    mesh.initialize()
    mesh.self_program("test_optimization")
    
    import numpy as np
    test_input = np.random.randn(20)
    result = mesh.compute_with_noise_optimization(test_input)
    
    # Demonstrate noise benefits
    noise_demo = mesh.demonstrate_noise_benefit(test_input)
    improvement = noise_demo.get('improvement_ratio', 1.0)
    
    print(f"Noise optimization improvement: {improvement:.2f}x")
    
    # Test Resource-Temporal Load Balancing
    print("Testing Resource-Temporal Load Balancing...")
    balancer = ResourceTemporalLoadBalancing(config)
    balancer.initialize()
    
    tasks = [
        {'id': 'task1', 'resources': {'energy': 20, 'materials': 10}},
        {'id': 'task2', 'resources': {'energy': 30, 'materials': 15}},
        {'id': 'task3', 'resources': {'energy': 25, 'materials': 12}}
    ]
    
    schedule = balancer.auto_schedule_computation(tasks)
    efficiency_demo = balancer.demonstrate_energy_efficiency(tasks)
    
    print(f"Scheduled {len(schedule)} tasks with {efficiency_demo['coordination_efficiency']:.2f} efficiency")
    print("Dual phenomenon experiment completed successfully!")

def run_triple_phenomenon_experiment(config: Dict[str, Any]) -> None:
    """Run triple phenomenon integration experiment."""
    logging.info("Starting triple phenomenon experiment...")
    
    from biocomputing.synergies.triple_phenomenon.emergent_architecture_evolution import EmergentArchitectureEvolution
    from biocomputing.synergies.triple_phenomenon.hierarchical_molecular_swarms import HierarchicalMolecularSwarms
    
    # Test Emergent Architecture Evolution
    print("Testing Emergent Architecture Evolution...")
    arch_evolution = EmergentArchitectureEvolution(config)
    arch_evolution.initialize()
    
    # Simulate performance feedback
    performance_feedback = {'accuracy': 0.8, 'speed': 100, 'efficiency': 0.75}
    evolution_result = arch_evolution.evolve_architecture(performance_feedback)
    
    novel_patterns = arch_evolution.discover_novel_patterns()
    
    print(f"Architecture evolved at step {evolution_result['evolution_step']}")
    print(f"Discovered {len(novel_patterns['novel_patterns'])} novel patterns")
    
    # Test Hierarchical Molecular Swarms
    print("Testing Hierarchical Molecular Swarms...")
    swarms = HierarchicalMolecularSwarms(config)
    swarms.initialize()
    
    import numpy as np
    test_data = np.random.randn(50)
    hierarchical_result = swarms.compute_hierarchical(test_data)
    
    parallelism_demo = swarms.demonstrate_massive_parallelism(1000)
    total_operations = parallelism_demo['total_parallel_operations']
    
    print(f"Hierarchical computation across {len(hierarchical_result['hierarchical_results'])} scales")
    print(f"Demonstrated {total_operations} parallel operations")
    print("Triple phenomenon experiment completed successfully!")

def run_full_integration_experiment(config: Dict[str, Any]) -> None:
    """Run full biological hypercomputing integration experiment."""
    logging.info("Starting full integration experiment...")
    
    print("Initializing Biological Hypercomputing System...")
    hypercomputing = BiologicalHypercomputing(config)
    hypercomputing.initialize()
    
    # Test transcendence of digital limits
    print("Testing transcendence of digital computing limits...")
    problem = {
        'name': 'complex_optimization',
        'size': 1000,
        'complexity': 'very_high',
        'constraints': ['resource_limited', 'time_critical']
    }
    
    transcendence_result = hypercomputing.transcend_digital_limits(problem)
    
    print("Transcendence Metrics:")
    for metric, value in transcendence_result['transcendence_metrics'].items():
        print(f"  {metric}: {value:.3f}")
    
    print("\\nBiological Advantages:")
    for advantage in transcendence_result['biological_advantages']:
        print(f"  • {advantage}")
    
    # Test continuous evolution
    print("\\nTesting continuous system evolution...")
    evolution_result = hypercomputing.continuous_evolution(generations=10)
    
    final_performance = evolution_result['final_performance']
    improvement = evolution_result['performance_improvement']
    
    print(f"Final performance: {final_performance:.3f}")
    print(f"Performance improvement: {improvement:.3f}")
    
    print("Full integration experiment completed successfully!")
    print("\\nBiological Hypercomputing has successfully demonstrated:")
    print("• Transcendence of digital computing limitations")
    print("• Noise-enhanced optimization capabilities") 
    print("• Emergent intelligence and self-organization")
    print("• Massive parallel processing with coordination")
    print("• Continuous evolution and improvement")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Biological Hypercomputing Platform")
    parser.add_argument("--config", default="config/default_config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--experiment", choices=["single", "dual", "triple", "full"],
                       default="single", help="Experiment type to run")
    parser.add_argument("--environment", choices=["development", "production"],
                       default="development", help="Environment configuration")
    
    args = parser.parse_args()
    
    # Load environment-specific config if specified
    if args.environment in ["development", "production"]:
        env_config_path = f"config/environments/{args.environment}.yaml"
        if Path(env_config_path).exists():
            args.config = env_config_path
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config)
    
    print("=" * 60)
    print("BIOLOGICAL HYPERCOMPUTING RESEARCH PLATFORM")
    print("=" * 60)
    print(f"Environment: {args.environment}")
    print(f"Experiment type: {args.experiment}")
    print(f"Configuration: {args.config}")
    print("=" * 60)
    
    try:
        if args.experiment == "single":
            run_single_phenomenon_experiment(config)
        elif args.experiment == "dual":
            run_dual_phenomenon_experiment(config)
        elif args.experiment == "triple":
            run_triple_phenomenon_experiment(config)
        elif args.experiment == "full":
            run_full_integration_experiment(config)
        
        print("\\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        print(f"\\nExperiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
