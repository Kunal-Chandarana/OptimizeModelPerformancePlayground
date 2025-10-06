"""
Visualization Helpers for ML Model Performance Optimization
===========================================================

This module provides utility functions for creating consistent and informative
visualizations across all optimization demos.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set consistent styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OptimizationVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)
        
    def plot_performance_comparison(self, results: Dict[str, Any], 
                                  metrics: List[str] = None,
                                  title: str = "Performance Comparison") -> plt.Figure:
        """Create a comprehensive performance comparison plot"""
        if not metrics:
            metrics = ['accuracy', 'time', 'memory', 'size']
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics
            if i < len(axes):
                self._plot_single_metric(axes[i], results, metric)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_single_metric(self, ax, results: Dict[str, Any], metric: str):
        """Plot a single metric across different methods"""
        methods = []
        values = []
        
        for method, data in results.items():
            if isinstance(data, dict) and metric in data:
                methods.append(method.replace('_', ' ').title())
                values.append(data[metric])
        
        if methods and values:
            bars = ax.bar(methods, values, color=self.colors[:len(methods)])
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
    
    def plot_learning_curves(self, histories: Dict[str, List[float]], 
                           title: str = "Learning Curves") -> plt.Figure:
        """Plot learning curves for different methods"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for method, history in histories.items():
            ax.plot(history, label=method.replace('_', ' ').title(), 
                   linewidth=2, marker='o', markersize=4)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch/Iteration')
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_optimization_landscape(self, results: Dict[str, Any],
                                  x_metric: str, y_metric: str,
                                  title: str = "Optimization Landscape") -> plt.Figure:
        """Create a scatter plot showing the optimization landscape"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x_values = []
        y_values = []
        labels = []
        
        for method, data in results.items():
            if isinstance(data, dict) and x_metric in data and y_metric in data:
                x_values.append(data[x_metric])
                y_values.append(data[y_metric])
                labels.append(method.replace('_', ' ').title())
        
        if x_values and y_values:
            scatter = ax.scatter(x_values, y_values, s=100, alpha=0.7, 
                               c=self.colors[:len(x_values)])
            
            # Add labels
            for i, label in enumerate(labels):
                ax.annotate(label, (x_values[i], y_values[i]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, ha='left')
            
            ax.set_xlabel(x_metric.replace('_', ' ').title())
            ax.set_ylabel(y_metric.replace('_', ' ').title())
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_improvement_analysis(self, baseline: Dict[str, Any], 
                                optimized: Dict[str, Any],
                                title: str = "Optimization Improvement") -> plt.Figure:
        """Plot improvement analysis comparing baseline vs optimized"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate improvements
        improvements = {}
        for metric in baseline:
            if metric in optimized and baseline[metric] != 0:
                improvement = ((optimized[metric] - baseline[metric]) / baseline[metric]) * 100
                improvements[metric] = improvement
        
        # Plot 1: Before vs After comparison
        metrics = list(baseline.keys())
        baseline_values = [baseline[m] for m in metrics]
        optimized_values = [optimized[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0].bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        axes[0].bar(x + width/2, optimized_values, width, label='Optimized', alpha=0.8)
        
        axes[0].set_xlabel('Metrics')
        axes[0].set_ylabel('Values')
        axes[0].set_title('Before vs After Optimization')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Improvement percentages
        improvement_metrics = list(improvements.keys())
        improvement_values = list(improvements.values())
        
        colors = ['green' if v > 0 else 'red' for v in improvement_values]
        bars = axes[1].bar(improvement_metrics, improvement_values, color=colors, alpha=0.7)
        
        axes[1].set_xlabel('Metrics')
        axes[1].set_ylabel('Improvement (%)')
        axes[1].set_title('Optimization Improvement')
        axes[1].set_xticklabels([m.replace('_', ' ').title() for m in improvement_metrics], rotation=45)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, improvement_values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_hyperparameter_importance(self, study_results: Dict[str, Any],
                                     title: str = "Hyperparameter Importance") -> plt.Figure:
        """Plot hyperparameter importance from optimization study"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # This would need to be adapted based on the specific optimization library used
        # For now, create a placeholder
        ax.text(0.5, 0.5, 'Hyperparameter Importance Plot\n(Adapt based on your optimization library)',
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        return fig
    
    def create_dashboard(self, results: Dict[str, Any], 
                        title: str = "ML Optimization Dashboard") -> plt.Figure:
        """Create a comprehensive dashboard with multiple visualizations"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
        
        # Placeholder for different dashboard components
        # This would be customized based on the specific results structure
        
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """Save plot to file with high quality"""
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"📊 Plot saved to: {filename}")

def create_summary_table(results: Dict[str, Any], 
                        metrics: List[str] = None) -> pd.DataFrame:
    """Create a summary table of results"""
    if not metrics:
        metrics = ['accuracy', 'time', 'memory', 'size']
    
    data = []
    for method, result in results.items():
        if isinstance(result, dict):
            row = {'Method': method.replace('_', ' ').title()}
            for metric in metrics:
                if metric in result:
                    row[metric.replace('_', ' ').title()] = result[metric]
                else:
                    row[metric.replace('_', ' ').title()] = 'N/A'
            data.append(row)
    
    return pd.DataFrame(data)

def print_optimization_summary(results: Dict[str, Any]):
    """Print a formatted summary of optimization results"""
    print("\n" + "="*80)
    print("📊 OPTIMIZATION SUMMARY")
    print("="*80)
    
    for method, result in results.items():
        if isinstance(result, dict):
            print(f"\n🔧 {method.replace('_', ' ').upper()}")
            print("-" * 40)
            for metric, value in result.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"  {metric.replace('_', ' ').title()}: {value}")

# Example usage
if __name__ == "__main__":
    # Example results
    example_results = {
        'baseline': {'accuracy': 0.85, 'time': 10.5, 'memory': 100},
        'optimized': {'accuracy': 0.87, 'time': 8.2, 'memory': 80},
        'quantized': {'accuracy': 0.86, 'time': 6.1, 'memory': 60}
    }
    
    # Create visualizer
    viz = OptimizationVisualizer()
    
    # Create plots
    fig1 = viz.plot_performance_comparison(example_results)
    fig2 = viz.plot_improvement_analysis(example_results['baseline'], example_results['optimized'])
    
    # Show plots
    plt.show()
    
    # Print summary
    print_optimization_summary(example_results)
