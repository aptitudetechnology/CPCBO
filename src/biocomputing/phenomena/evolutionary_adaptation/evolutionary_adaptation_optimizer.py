"""Optimizer for evolutionary_adaptation phenomenon."""

import numpy as np
from typing import Dict, Any, Callable, Optional
from .evolutionary_adaptation_core import EvolutionaryAdaptationCore

class EvolutionaryAdaptationOptimizer:
    """Optimizer for evolutionary_adaptation phenomenon parameters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_history = []
        self.best_parameters = None
        self.best_score = float('-inf')
    
    def optimize(self, objective_function: Callable[[Dict[str, Any]], float],
                parameter_ranges: Dict[str, tuple],
                iterations: int = 100) -> Dict[str, Any]:
        """Optimize phenomenon parameters using evolutionary approach."""
        
        # Initialize population
        population_size = 20
        population = self._initialize_population(parameter_ranges, population_size)
        
        for iteration in range(iterations):
            # Evaluate population
            scores = []
            for individual in population:
                core = EvolutionaryAdaptationCore({**self.config, 'evolutionary_adaptation_params': individual})
                core.initialize()
                score = objective_function(individual)
                scores.append(score)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_parameters = individual.copy()
            
            # Evolve population
            population = self._evolve_population(population, scores)
            
            self.optimization_history.append({
                'iteration': iteration,
                'best_score': max(scores),
                'avg_score': np.mean(scores)
            })
        
        return {
            'best_parameters': self.best_parameters,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history
        }
    
    def _initialize_population(self, ranges: Dict[str, tuple], size: int) -> List[Dict[str, Any]]:
        """Initialize random population within parameter ranges."""
        population = []
        for _ in range(size):
            individual = {}
            for param, (min_val, max_val) in ranges.items():
                individual[param] = np.random.uniform(min_val, max_val)
            population.append(individual)
        return population
    
    def _evolve_population(self, population: List[Dict[str, Any]], 
                          scores: List[float]) -> List[Dict[str, Any]]:
        """Evolve population using selection and mutation."""
        # Simple evolutionary step - can be enhanced
        sorted_pop = [x for _, x in sorted(zip(scores, population), reverse=True)]
        
        # Keep top half, mutate to create new individuals
        new_population = sorted_pop[:len(population)//2]
        
        for i in range(len(population)//2):
            mutated = self._mutate(sorted_pop[i % len(new_population)])
            new_population.append(mutated)
        
        return new_population
    
    def _mutate(self, individual: Dict[str, Any], mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Mutate an individual."""
        mutated = individual.copy()
        for key, value in mutated.items():
            if np.random.random() < mutation_rate:
                mutated[key] = value * (1 + np.random.normal(0, 0.1))
        return mutated
