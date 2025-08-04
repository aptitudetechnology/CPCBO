"""Visualization tools for biological computing phenomena."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

class PhenomenonVisualizer:
    """Visualize biological computing phenomena and their interactions."""
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_synergy_matrix(self, synergies: Dict[str, float], save_path: str = None) -> None:
        """Plot synergy interaction matrix."""
        # Create matrix from synergy dictionary
        phenomena = list(set([s.split('_')[0] for s in synergies.keys()] + 
                            [s.split('_')[1] for s in synergies.keys()]))
        
        matrix = np.zeros((len(phenomena), len(phenomena)))
        
        for synergy_name, strength in synergies.items():
            parts = synergy_name.split('_')
            if len(parts) >= 2:
                try:
                    i = phenomena.index(parts[0])
                    j = phenomena.index(parts[1])
                    matrix[i, j] = strength
                    matrix[j, i] = strength  # Symmetric
                except ValueError:
                    continue
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, 
                   xticklabels=phenomena, 
                   yticklabels=phenomena,
                   annot=True, 
                   cmap='viridis',
                   cbar_kws={'label': 'Synergy Strength'})
        
        plt.title('Biological Computing Phenomena Synergy Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_emergence_timeline(self, emergence_data: List[Dict[str, Any]], save_path: str = None) -> None:
        """Plot emergence of properties over time."""
        if not emergence_data:
            print("No emergence data to plot")
            return
        
        times = [d.get('timestamp', i) for i, d in enumerate(emergence_data)]
        complexity_gains = [d.get('complexity_gain', 0) for d in emergence_data]
        
        plt.figure(figsize=(12, 6))
        
        # Plot complexity gain over time
        plt.subplot(1, 2, 1)
        plt.plot(times, complexity_gains, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Time Step')
        plt.ylabel('Complexity Gain')
        plt.title('Emergent Complexity Over Time')
        plt.grid(True, alpha=0.3)
        
        # Plot cumulative emergence
        plt.subplot(1, 2, 2)
        cumulative_emergence = np.cumsum(complexity_gains)
        plt.plot(times, cumulative_emergence, 'r-o', linewidth=2, markersize=4)
        plt.xlabel('Time Step')
        plt.ylabel('Cumulative Emergence')
        plt.title('Cumulative Emergent Properties')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, performance_data: Dict[str, List[float]], save_path: str = None) -> None:
        """Plot performance comparison between different approaches."""
        plt.figure(figsize=(12, 8))
        
        # Box plot for performance comparison
        plt.subplot(2, 2, 1)
        data_for_boxplot = []
        labels = []
        for approach, values in performance_data.items():
            data_for_boxplot.append(values)
            labels.append(approach)
        
        plt.boxplot(data_for_boxplot, labels=labels)
        plt.title('Performance Distribution by Approach')
        plt.ylabel('Performance Metric')
        plt.xticks(rotation=45)
        
        # Line plot showing performance over time
        plt.subplot(2, 2, 2)
        for approach, values in performance_data.items():
            plt.plot(values, label=approach, linewidth=2)
        plt.title('Performance Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Bar plot of average performance
        plt.subplot(2, 2, 3)
        avg_performance = {k: np.mean(v) for k, v in performance_data.items()}
        plt.bar(avg_performance.keys(), avg_performance.values())
        plt.title('Average Performance by Approach')
        plt.ylabel('Average Performance')
        plt.xticks(rotation=45)
        
        # Histogram of all performance values
        plt.subplot(2, 2, 4)
        all_values = []
        for values in performance_data.values():
            all_values.extend(values)
        plt.hist(all_values, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Performance Distribution (All Approaches)')
        plt.xlabel('Performance Value')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_evolutionary_progress(self, evolution_history: List[Dict[str, Any]], save_path: str = None) -> None:
        """Plot evolutionary progress over generations."""
        if not evolution_history:
            print("No evolution history to plot")
            return
        
        generations = [h.get('generation', i) for i, h in enumerate(evolution_history)]
        avg_fitness = [h.get('avg_fitness', 0) for h in evolution_history]
        max_fitness = [h.get('max_fitness', 0) for h in evolution_history]
        diversity = [h.get('genetic_diversity', 0) for h in evolution_history]
        population_size = [h.get('population_size', 0) for h in evolution_history]
        
        plt.figure(figsize=(15, 10))
        
        # Fitness evolution
        plt.subplot(2, 3, 1)
        plt.plot(generations, avg_fitness, 'g-', label='Average Fitness', linewidth=2)
        plt.plot(generations, max_fitness, 'r-', label='Maximum Fitness', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Genetic diversity
        plt.subplot(2, 3, 2)
        plt.plot(generations, diversity, 'b-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Genetic Diversity')
        plt.title('Genetic Diversity Over Time')
        plt.grid(True, alpha=0.3)
        
        # Population size
        plt.subplot(2, 3, 3)
        plt.plot(generations, population_size, 'm-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Population Size')
        plt.title('Population Size Evolution')
        plt.grid(True, alpha=0.3)
        
        # Fitness distribution (final generation)
        if 'fitness_distribution' in evolution_history[-1]:
            plt.subplot(2, 3, 4)
            plt.hist(evolution_history[-1]['fitness_distribution'], bins=15, alpha=0.7, edgecolor='black')
            plt.xlabel('Fitness')
            plt.ylabel('Count')
            plt.title('Final Generation Fitness Distribution')
        
        # Fitness improvement rate
        plt.subplot(2, 3, 5)
        fitness_improvements = np.diff(max_fitness)
        plt.plot(generations[1:], fitness_improvements, 'orange', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Improvement')
        plt.title('Fitness Improvement Rate')
        plt.grid(True, alpha=0.3)
        
        # Diversity vs Fitness scatter
        plt.subplot(2, 3, 6)
        plt.scatter(diversity, max_fitness, alpha=0.6, c=generations, cmap='viridis')
        plt.xlabel('Genetic Diversity')
        plt.ylabel('Maximum Fitness')
        plt.title('Diversity vs Fitness')
        plt.colorbar(label='Generation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Example usage
    visualizer = PhenomenonVisualizer()
    
    # Example synergy data
    synergies = {
        'cellular_molecular': 0.8,
        'genetic_swarm': 0.6,
        'quantum_noise': 0.9,
        'metabolism_multiscale': 0.7
    }
    
    visualizer.plot_synergy_matrix(synergies)
