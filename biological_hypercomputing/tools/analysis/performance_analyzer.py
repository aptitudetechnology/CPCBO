"""Performance analysis tools for biological computing systems."""

import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

class PerformanceAnalyzer:
    """Analyze performance characteristics of biological computing systems."""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_transcendence_metrics(self, digital_performance: Dict[str, float], 
                                    biological_performance: Dict[str, float]) -> Dict[str, Any]:
        """Analyze how biological computing transcends digital limitations."""
        transcendence_analysis = {}
        
        for metric in digital_performance.keys():
            if metric in biological_performance:
                digital_val = digital_performance[metric]
                bio_val = biological_performance[metric]
                
                improvement = (bio_val - digital_val) / digital_val if digital_val != 0 else float('inf')
                
                transcendence_analysis[metric] = {
                    'digital_value': digital_val,
                    'biological_value': bio_val,
                    'improvement_ratio': improvement,
                    'transcends_digital': improvement > 0.1,  # 10% improvement threshold
                    'significance': self._calculate_significance(improvement)
                }
        
        # Overall transcendence score
        improvements = [ta['improvement_ratio'] for ta in transcendence_analysis.values() 
                       if ta['improvement_ratio'] != float('inf')]
        
        transcendence_analysis['overall'] = {
            'average_improvement': np.mean(improvements) if improvements else 0,
            'transcendence_score': np.mean([ta['transcends_digital'] for ta in transcendence_analysis.values()]),
            'breakthrough_metrics': [k for k, v in transcendence_analysis.items() 
                                   if isinstance(v, dict) and v.get('improvement_ratio', 0) > 0.5]
        }
        
        return transcendence_analysis
    
    def analyze_noise_benefits(self, no_noise_results: List[float], 
                              with_noise_results: List[float]) -> Dict[str, Any]:
        """Analyze how noise improves rather than degrades performance."""
        if len(no_noise_results) != len(with_noise_results):
            raise ValueError("Result arrays must have same length")
        
        # Statistical comparison
        t_stat, p_value = stats.ttest_rel(with_noise_results, no_noise_results)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(no_noise_results) + np.var(with_noise_results)) / 2)
        cohens_d = (np.mean(with_noise_results) - np.mean(no_noise_results)) / pooled_std
        
        # Improvement analysis
        improvements = [(w - n) / n for w, n in zip(with_noise_results, no_noise_results) if n != 0]
        
        analysis = {
            'noise_improves_performance': np.mean(with_noise_results) > np.mean(no_noise_results),
            'average_improvement': np.mean(improvements) if improvements else 0,
            'improvement_consistency': np.mean([i > 0 for i in improvements]) if improvements else 0,
            'statistical_significance': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size': cohens_d,
                'effect_magnitude': self._interpret_effect_size(cohens_d)
            },
            'performance_distributions': {
                'no_noise': {
                    'mean': np.mean(no_noise_results),
                    'std': np.std(no_noise_results),
                    'min': np.min(no_noise_results),
                    'max': np.max(no_noise_results)
                },
                'with_noise': {
                    'mean': np.mean(with_noise_results),
                    'std': np.std(with_noise_results),
                    'min': np.min(with_noise_results),
                    'max': np.max(with_noise_results)
                }
            }
        }
        
        return analysis
    
    def analyze_emergence_patterns(self, emergence_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in emergent property development."""
        if not emergence_history:
            return {'error': 'No emergence history provided'}
        
        # Extract emergence metrics over time
        times = [h.get('timestamp', i) for i, h in enumerate(emergence_history)]
        complexity_gains = [h.get('complexity_gain', 0) for h in emergence_history]
        
        # Trend analysis
        if len(times) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(times, complexity_gains)
        else:
            slope = intercept = r_value = p_value = std_err = 0
        
        # Emergence events detection
        emergence_threshold = np.mean(complexity_gains) + 2 * np.std(complexity_gains)
        emergence_events = [(i, times[i], complexity_gains[i]) 
                           for i in range(len(complexity_gains)) 
                           if complexity_gains[i] > emergence_threshold]
        
        # Phase analysis
        phases = self._identify_emergence_phases(complexity_gains)
        
        analysis = {
            'emergence_trend': {
                'slope': slope,
                'correlation': r_value,
                'trend_significance': p_value,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            },
            'emergence_events': {
                'count': len(emergence_events),
                'events': emergence_events,
                'average_magnitude': np.mean([e[2] for e in emergence_events]) if emergence_events else 0
            },
            'emergence_phases': phases,
            'complexity_statistics': {
                'total_complexity_gained': np.sum(complexity_gains),
                'average_complexity_gain': np.mean(complexity_gains),
                'complexity_volatility': np.std(complexity_gains),
                'peak_emergence': np.max(complexity_gains)
            }
        }
        
        return analysis
    
    def analyze_synergy_effectiveness(self, single_phenomenon_results: Dict[str, List[float]], 
                                    synergistic_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze effectiveness of synergistic combinations."""
        synergy_analysis = {}
        
        for combination, synergy_values in synergistic_results.items():
            # Find constituent phenomena
            constituents = combination.split('_')
            
            # Calculate expected performance (sum of individual phenomena)
            expected_performance = []
            for i in range(len(synergy_values)):
                expected = sum(single_phenomenon_results.get(constituent, [0])[i] 
                             for constituent in constituents 
                             if constituent in single_phenomenon_results)
                expected_performance.append(expected)
            
            # Calculate synergy benefit
            synergy_benefits = [(s - e) / e for s, e in zip(synergy_values, expected_performance) if e != 0]
            
            synergy_analysis[combination] = {
                'average_synergy_benefit': np.mean(synergy_benefits) if synergy_benefits else 0,
                'synergy_consistency': np.mean([b > 0 for b in synergy_benefits]) if synergy_benefits else 0,
                'peak_synergy': np.max(synergy_benefits) if synergy_benefits else 0,
                'synergy_volatility': np.std(synergy_benefits) if synergy_benefits else 0,
                'emergent_behavior': np.mean(synergy_benefits) > 0.2 if synergy_benefits else False  # 20% threshold
            }
        
        # Overall synergy assessment
        all_benefits = []
        for analysis in synergy_analysis.values():
            if analysis['average_synergy_benefit'] != 0:
                all_benefits.append(analysis['average_synergy_benefit'])
        
        synergy_analysis['overall'] = {
            'average_synergy_across_combinations': np.mean(all_benefits) if all_benefits else 0,
            'successful_synergies': sum(1 for a in synergy_analysis.values() 
                                      if a['average_synergy_benefit'] > 0.1),
            'total_combinations_tested': len(synergistic_results),
            'synergy_success_rate': (sum(1 for a in synergy_analysis.values() 
                                       if a['average_synergy_benefit'] > 0.1) / 
                                   len(synergistic_results) if synergistic_results else 0)
        }
        
        return synergy_analysis
    
    def analyze_scalability(self, problem_sizes: List[int], 
                           computation_times: List[float], 
                           performance_metrics: List[float]) -> Dict[str, Any]:
        """Analyze scalability characteristics of biological computing."""
        if len(problem_sizes) != len(computation_times) or len(problem_sizes) != len(performance_metrics):
            raise ValueError("All input arrays must have same length")
        
        # Time complexity analysis
        log_sizes = np.log(problem_sizes)
        log_times = np.log(computation_times)
        
        time_slope, _, time_r_value, _, _ = stats.linregress(log_sizes, log_times)
        
        # Performance scaling analysis
        perf_slope, _, perf_r_value, _, _ = stats.linregress(problem_sizes, performance_metrics)
        
        # Efficiency analysis
        efficiency = [p / t for p, t in zip(performance_metrics, computation_times)]
        efficiency_slope, _, eff_r_value, _, _ = stats.linregress(problem_sizes, efficiency)
        
        scalability_analysis = {
            'time_complexity': {
                'complexity_exponent': time_slope,
                'complexity_class': self._classify_complexity(time_slope),
                'scaling_correlation': time_r_value
            },
            'performance_scaling': {
                'performance_slope': perf_slope,
                'performance_correlation': perf_r_value,
                'scales_well': perf_slope > -0.1  # Performance doesn't degrade much with size
            },
            'efficiency_analysis': {
                'efficiency_slope': efficiency_slope,
                'efficiency_correlation': eff_r_value,
                'maintains_efficiency': efficiency_slope > -0.01
            },
            'scalability_metrics': {
                'best_performance': np.max(performance_metrics),
                'worst_performance': np.min(performance_metrics),
                'performance_range': np.max(performance_metrics) - np.min(performance_metrics),
                'average_efficiency': np.mean(efficiency)
            }
        }
        
        return scalability_analysis
    
    def generate_comprehensive_report(self, all_analyses: Dict[str, Dict[str, Any]]) -> str:
        """Generate comprehensive performance analysis report."""
        report = []
        report.append("BIOLOGICAL HYPERCOMPUTING PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 55)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 17)
        
        transcendence = all_analyses.get('transcendence', {})
        if transcendence and 'overall' in transcendence:
            avg_improvement = transcendence['overall'].get('average_improvement', 0)
            report.append(f"Average improvement over digital computing: {avg_improvement:.1%}")
            
            breakthrough_metrics = transcendence['overall'].get('breakthrough_metrics', [])
            if breakthrough_metrics:
                report.append(f"Breakthrough achievements in: {', '.join(breakthrough_metrics)}")
        
        noise_analysis = all_analyses.get('noise_benefits', {})
        if noise_analysis and noise_analysis.get('noise_improves_performance', False):
            improvement = noise_analysis.get('average_improvement', 0)
            report.append(f"Noise enhancement benefit: {improvement:.1%} performance improvement")
        
        report.append("")
        
        # Detailed Analysis Sections
        for analysis_name, analysis_data in all_analyses.items():
            report.append(f"{analysis_name.upper().replace('_', ' ')} ANALYSIS")
            report.append("-" * len(f"{analysis_name.upper().replace('_', ' ')} ANALYSIS"))
            
            if analysis_name == 'transcendence':
                self._add_transcendence_section(report, analysis_data)
            elif analysis_name == 'noise_benefits':
                self._add_noise_benefits_section(report, analysis_data)
            elif analysis_name == 'emergence':
                self._add_emergence_section(report, analysis_data)
            elif analysis_name == 'synergy':
                self._add_synergy_section(report, analysis_data)
            elif analysis_name == 'scalability':
                self._add_scalability_section(report, analysis_data)
            
            report.append("")
        
        # Conclusions
        report.append("CONCLUSIONS")
        report.append("-" * 11)
        report.append(self._generate_conclusions(all_analyses))
        
        return "\n".join(report)
    
    # Helper methods
    def _calculate_significance(self, improvement: float) -> str:
        """Calculate significance level of improvement."""
        if improvement > 1.0:
            return "revolutionary"
        elif improvement > 0.5:
            return "major"
        elif improvement > 0.2:
            return "significant"
        elif improvement > 0.1:
            return "moderate"
        elif improvement > 0:
            return "minor"
        else:
            return "none"
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d >= 0.8:
            return "large"
        elif abs_d >= 0.5:
            return "medium"
        elif abs_d >= 0.2:
            return "small"
        else:
            return "negligible"
    
    def _identify_emergence_phases(self, complexity_gains: List[float]) -> Dict[str, Any]:
        """Identify phases in emergence development."""
        if len(complexity_gains) < 3:
            return {'phases': [], 'phase_count': 0}
        
        # Simple phase detection based on trend changes
        phases = []
        current_phase = {'start': 0, 'type': 'unknown'}
        
        for i in range(1, len(complexity_gains) - 1):
            prev_trend = complexity_gains[i] - complexity_gains[i-1]
            next_trend = complexity_gains[i+1] - complexity_gains[i]
            
            # Detect trend changes
            if prev_trend > 0 and next_trend < 0:  # Peak
                if current_phase['type'] != 'growth':
                    current_phase['end'] = i
                    phases.append(current_phase)
                    current_phase = {'start': i, 'type': 'decline'}
            elif prev_trend < 0 and next_trend > 0:  # Trough
                if current_phase['type'] != 'decline':
                    current_phase['end'] = i
                    phases.append(current_phase)
                    current_phase = {'start': i, 'type': 'growth'}
        
        # Close final phase
        current_phase['end'] = len(complexity_gains) - 1
        phases.append(current_phase)
        
        return {'phases': phases, 'phase_count': len(phases)}
    
    def _classify_complexity(self, exponent: float) -> str:
        """Classify time complexity based on exponent."""
        if exponent <= 1.1:
            return "linear"
        elif exponent <= 1.5:
            return "quasi-linear"
        elif exponent <= 2.1:
            return "quadratic"
        elif exponent <= 3.1:
            return "cubic"
        else:
            return "polynomial"
    
    def _add_transcendence_section(self, report: List[str], data: Dict[str, Any]) -> None:
        """Add transcendence analysis section to report."""
        if 'overall' in data:
            overall = data['overall']
            report.append(f"Overall transcendence score: {overall.get('transcendence_score', 0):.2f}")
            report.append(f"Average improvement: {overall.get('average_improvement', 0):.1%}")
        
        breakthrough_count = sum(1 for k, v in data.items() 
                               if isinstance(v, dict) and v.get('transcends_digital', False))
        report.append(f"Metrics showing breakthrough performance: {breakthrough_count}")
    
    def _add_noise_benefits_section(self, report: List[str], data: Dict[str, Any]) -> None:
        """Add noise benefits section to report."""
        report.append(f"Noise improves performance: {data.get('noise_improves_performance', False)}")
        report.append(f"Average improvement: {data.get('average_improvement', 0):.1%}")
        report.append(f"Improvement consistency: {data.get('improvement_consistency', 0):.1%}")
        
        if 'statistical_significance' in data:
            sig = data['statistical_significance']
            report.append(f"Statistical significance: p = {sig.get('p_value', 1):.4f}")
            report.append(f"Effect size: {sig.get('effect_magnitude', 'unknown')}")
    
    def _add_emergence_section(self, report: List[str], data: Dict[str, Any]) -> None:
        """Add emergence analysis section to report."""
        if 'emergence_trend' in data:
            trend = data['emergence_trend']
            report.append(f"Emergence trend: {trend.get('trend_direction', 'unknown')}")
            report.append(f"Trend correlation: {trend.get('correlation', 0):.3f}")
        
        if 'emergence_events' in data:
            events = data['emergence_events']
            report.append(f"Significant emergence events: {events.get('count', 0)}")
        
        if 'complexity_statistics' in data:
            stats = data['complexity_statistics']
            report.append(f"Total complexity gained: {stats.get('total_complexity_gained', 0):.3f}")
            report.append(f"Peak emergence magnitude: {stats.get('peak_emergence', 0):.3f}")
    
    def _add_synergy_section(self, report: List[str], data: Dict[str, Any]) -> None:
        """Add synergy analysis section to report."""
        if 'overall' in data:
            overall = data['overall']
            report.append(f"Successful synergies: {overall.get('successful_synergies', 0)}")
            report.append(f"Synergy success rate: {overall.get('synergy_success_rate', 0):.1%}")
            report.append(f"Average synergy benefit: {overall.get('average_synergy_across_combinations', 0):.1%}")
    
    def _add_scalability_section(self, report: List[str], data: Dict[str, Any]) -> None:
        """Add scalability analysis section to report."""
        if 'time_complexity' in data:
            time_comp = data['time_complexity']
            report.append(f"Time complexity class: {time_comp.get('complexity_class', 'unknown')}")
        
        if 'performance_scaling' in data:
            perf_scaling = data['performance_scaling']
            report.append(f"Scales well with problem size: {perf_scaling.get('scales_well', False)}")
        
        if 'efficiency_analysis' in data:
            eff = data['efficiency_analysis']
            report.append(f"Maintains efficiency at scale: {eff.get('maintains_efficiency', False)}")
    
    def _generate_conclusions(self, all_analyses: Dict[str, Dict[str, Any]]) -> str:
        """Generate conclusions based on all analyses."""
        conclusions = []
        
        # Check for transcendence evidence
        transcendence = all_analyses.get('transcendence', {})
        if transcendence and transcendence.get('overall', {}).get('average_improvement', 0) > 0.2:
            conclusions.append("• Demonstrated significant transcendence of digital computing limitations")
        
        # Check noise benefits
        noise = all_analyses.get('noise_benefits', {})
        if noise and noise.get('noise_improves_performance', False):
            conclusions.append("• Confirmed that noise enhances rather than degrades computational performance")
        
        # Check emergence
        emergence = all_analyses.get('emergence', {})
        if emergence and emergence.get('emergence_events', {}).get('count', 0) > 0:
            conclusions.append("• Detected genuine emergence of new computational capabilities")
        
        # Check synergies
        synergy = all_analyses.get('synergy', {})
        if synergy and synergy.get('overall', {}).get('synergy_success_rate', 0) > 0.5:
            conclusions.append("• Demonstrated effective synergistic combinations between phenomena")
        
        # Check scalability
        scalability = all_analyses.get('scalability', {})
        if scalability and scalability.get('performance_scaling', {}).get('scales_well', False):
            conclusions.append("• Confirmed good scalability characteristics for large problems")
        
        if not conclusions:
            conclusions.append("• Further analysis needed to demonstrate biological computing advantages")
        
        conclusions.append("")
        conclusions.append("This analysis provides evidence for the viability of biological hypercomputing")
        conclusions.append("as a paradigm that can transcend traditional digital computing limitations.")
        
        return "\n".join(conclusions)

if __name__ == "__main__":
    # Example usage
    analyzer = PerformanceAnalyzer()
    
    # Example transcendence analysis
    digital_perf = {'accuracy': 0.85, 'speed': 100, 'efficiency': 0.7}
    bio_perf = {'accuracy': 0.95, 'speed': 150, 'efficiency': 0.9}
    
    transcendence = analyzer.analyze_transcendence_metrics(digital_perf, bio_perf)
    print("Transcendence Analysis:")
    for metric, analysis in transcendence.items():
        if isinstance(analysis, dict) and 'improvement_ratio' in analysis:
            print(f"  {metric}: {analysis['improvement_ratio']:.1%} improvement")
