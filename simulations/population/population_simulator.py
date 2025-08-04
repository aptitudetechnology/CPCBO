"""Population-scale simulation framework."""

import numpy as np
from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from biocomputing.phenomena.swarm_intelligence.swarm_intelligence_core import SwarmIntelligenceCore
from biocomputing.phenomena.evolutionary_adaptation.evolutionary_adaptation_core import EvolutionaryAdaptationCore

class PopulationSimulator:
    """Simulate population-scale biological computing processes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.swarm_intelligence = SwarmIntelligenceCore(config)
        self.evolutionary_adaptation = EvolutionaryAdaptationCore(config)
        self.population = []
        self.environment = {}
    
    def initialize_population(self, population_size: int) -> None:
        """Initialize population of computational agents."""
        self.population = []
        for i in range(population_size):
            agent = {
                'id': i,
                'genome': np.random.randn(10),  # Random genome
                'fitness': 0.0,
                'age': 0,
                'position': np.random.randn(2),
                'behavior_state': 'exploring',
                'energy': 100.0,
                'reproduction_count': 0
            }
            self.population.append(agent)
        
        # Initialize environment
        self.environment = {
            'resources': np.random.uniform(0, 1, (10, 10)),  # Resource grid
            'challenges': [],
            'time': 0
        }
    
    def simulate_population_evolution(self, generations: int) -> Dict[str, Any]:
        """Simulate population evolution and swarm behavior."""
        evolution_history = []
        
        for generation in range(generations):
            # Update environment
            self._update_environment()
            
            # Evaluate fitness of all agents
            self._evaluate_population_fitness()
            
            # Apply swarm intelligence for collective behavior
            self._apply_swarm_behavior()
            
            # Perform evolutionary operations
            self._evolutionary_step()
            
            # Record generation statistics
            generation_stats = self._collect_generation_statistics(generation)
            evolution_history.append(generation_stats)
            
            # Age population
            for agent in self.population:
                agent['age'] += 1
        
        return {
            'evolution_history': evolution_history,
            'final_population': len(self.population),
            'best_fitness': max(agent['fitness'] for agent in self.population),
            'genetic_diversity': self._measure_genetic_diversity()
        }
    
    def _update_environment(self) -> None:
        """Update environmental conditions."""
        self.environment['time'] += 1
        
        # Slowly change resource distribution
        noise = np.random.normal(0, 0.01, self.environment['resources'].shape)
        self.environment['resources'] += noise
        self.environment['resources'] = np.clip(self.environment['resources'], 0, 1)
        
        # Occasionally add challenges
        if np.random.random() < 0.1:
            challenge = {
                'type': 'environmental_stress',
                'severity': np.random.uniform(0.1, 0.5),
                'duration': np.random.randint(5, 15)
            }
            self.environment['challenges'].append(challenge)
        
        # Remove expired challenges
        self.environment['challenges'] = [
            c for c in self.environment['challenges'] 
            if c.get('duration', 0) > 0
        ]
        
        # Decrease challenge durations
        for challenge in self.environment['challenges']:
            challenge['duration'] -= 1
    
    def _evaluate_population_fitness(self) -> None:
        """Evaluate fitness of all agents in population."""
        for agent in self.population:
            # Base fitness from genome
            base_fitness = np.sum(agent['genome']**2) / len(agent['genome'])
            
            # Environmental adaptation bonus
            position = tuple(np.clip(agent['position'].astype(int), 0, 9))
            resource_bonus = self.environment['resources'][position] * 0.5
            
            # Age penalty (but not too harsh)
            age_penalty = min(agent['age'] * 0.01, 0.3)
            
            # Challenge penalty
            challenge_penalty = sum(c['severity'] for c in self.environment['challenges']) * 0.1
            
            agent['fitness'] = max(0, base_fitness + resource_bonus - age_penalty - challenge_penalty)
    
    def _apply_swarm_behavior(self) -> None:
        """Apply swarm intelligence for collective behavior."""
        # Simple swarm behaviors: aggregation, alignment, separation
        
        for agent in self.population:
            neighbors = self._find_neighbors(agent, radius=3.0)
            
            if neighbors:
                # Aggregation: move towards center of neighbors
                center = np.mean([n['position'] for n in neighbors], axis=0)
                aggregation_force = (center - agent['position']) * 0.1
                
                # Alignment: align with neighbor velocities (simplified)
                avg_genome_direction = np.mean([n['genome'][:2] for n in neighbors], axis=0)
                alignment_force = avg_genome_direction * 0.05
                
                # Separation: avoid overcrowding
                separation_force = np.zeros(2)
                for neighbor in neighbors:
                    if np.linalg.norm(neighbor['position'] - agent['position']) < 1.0:
                        separation_force += (agent['position'] - neighbor['position']) * 0.2
                
                # Apply forces to position
                total_force = aggregation_force + alignment_force + separation_force
                agent['position'] += total_force
                
                # Update behavior state based on swarm context
                if len(neighbors) > 5:
                    agent['behavior_state'] = 'schooling'
                elif len(neighbors) > 2:
                    agent['behavior_state'] = 'cooperating'
                else:
                    agent['behavior_state'] = 'exploring'
    
    def _evolutionary_step(self) -> None:
        """Perform one evolutionary step: selection, reproduction, mutation."""
        # Selection: remove bottom 20% of population
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        survivors = self.population[:int(len(self.population) * 0.8)]
        
        # Reproduction: top agents reproduce
        offspring = []
        top_reproducers = survivors[:int(len(survivors) * 0.3)]
        
        for parent in top_reproducers:
            if parent['fitness'] > 0.5 and np.random.random() < 0.7:  # Reproduction probability
                # Create offspring with mutation
                child_genome = parent['genome'].copy()
                mutation = np.random.normal(0, 0.1, child_genome.shape)
                child_genome += mutation
                
                child = {
                    'id': len(self.population) + len(offspring),
                    'genome': child_genome,
                    'fitness': 0.0,
                    'age': 0,
                    'position': parent['position'] + np.random.randn(2) * 0.5,
                    'behavior_state': 'exploring',
                    'energy': 100.0,
                    'reproduction_count': 0
                }
                offspring.append(child)
                parent['reproduction_count'] += 1
        
        # Update population
        self.population = survivors + offspring
    
    def _find_neighbors(self, agent: Dict[str, Any], radius: float) -> List[Dict[str, Any]]:
        """Find neighboring agents within radius."""
        neighbors = []
        for other in self.population:
            if other['id'] != agent['id']:
                distance = np.linalg.norm(other['position'] - agent['position'])
                if distance <= radius:
                    neighbors.append(other)
        return neighbors
    
    def _collect_generation_statistics(self, generation: int) -> Dict[str, Any]:
        """Collect statistics for current generation."""
        fitnesses = [agent['fitness'] for agent in self.population]
        ages = [agent['age'] for agent in self.population]
        
        return {
            'generation': generation,
            'population_size': len(self.population),
            'avg_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'avg_age': np.mean(ages),
            'genetic_diversity': self._measure_genetic_diversity(),
            'behavior_states': self._count_behavior_states()
        }
    
    def _measure_genetic_diversity(self) -> float:
        """Measure genetic diversity in population."""
        if len(self.population) < 2:
            return 0.0
        
        genomes = np.array([agent['genome'] for agent in self.population])
        pairwise_distances = []
        
        for i in range(len(genomes)):
            for j in range(i+1, len(genomes)):
                distance = np.linalg.norm(genomes[i] - genomes[j])
                pairwise_distances.append(distance)
        
        return np.mean(pairwise_distances) if pairwise_distances else 0.0
    
    def _count_behavior_states(self) -> Dict[str, int]:
        """Count agents in each behavior state."""
        states = {}
        for agent in self.population:
            state = agent.get('behavior_state', 'unknown')
            states[state] = states.get(state, 0) + 1
        return states

if __name__ == "__main__":
    config = {'swarm_intelligence_params': {'swarm_size': 100}}
    simulator = PopulationSimulator(config)
    simulator.initialize_population(50)
    results = simulator.simulate_population_evolution(20)
    print(f"Population evolution completed: {results['final_population']} agents, best fitness: {results['best_fitness']:.3f}")
