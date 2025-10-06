"""
ML Infrastructure Cost Optimization Demo
========================================

This demo focuses on optimizing costs for ML infrastructure in production,
covering resource allocation, auto-scaling, and cost management strategies.

Key Topics:
- Resource cost analysis and optimization
- Auto-scaling strategies for cost efficiency
- Spot instances and reserved capacity
- Model optimization for cost reduction
- Cost monitoring and budgeting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class CostOptimizationDemo:
    def __init__(self):
        self.cost_models = {}
        self.resource_usage = {}
        self.optimization_results = {}
        
    def create_cost_models(self):
        """Create different cost models for infrastructure"""
        print("💰 Creating cost models for infrastructure...")
        
        # Define cost structures (per hour)
        cost_models = {
            'on_demand': {
                'cpu_cost_per_hour': 0.10,
                'memory_cost_per_gb_hour': 0.05,
                'gpu_cost_per_hour': 2.00,
                'storage_cost_per_gb_month': 0.10
            },
            'reserved_1year': {
                'cpu_cost_per_hour': 0.07,  # 30% discount
                'memory_cost_per_gb_hour': 0.035,
                'gpu_cost_per_hour': 1.40,
                'storage_cost_per_gb_month': 0.10
            },
            'reserved_3year': {
                'cpu_cost_per_hour': 0.05,  # 50% discount
                'memory_cost_per_gb_hour': 0.025,
                'gpu_cost_per_hour': 1.00,
                'storage_cost_per_gb_month': 0.10
            },
            'spot': {
                'cpu_cost_per_hour': 0.03,  # 70% discount
                'memory_cost_per_gb_hour': 0.015,
                'gpu_cost_per_hour': 0.60,
                'storage_cost_per_gb_month': 0.10
            }
        }
        
        return cost_models
    
    def simulate_workload_patterns(self, duration_days=30):
        """Simulate different workload patterns"""
        print("📊 Simulating workload patterns...")
        
        workload_patterns = {
            'batch_processing': {
                'description': 'Heavy batch processing during off-hours',
                'cpu_usage': self._generate_batch_pattern(duration_days),
                'memory_usage': self._generate_batch_pattern(duration_days, base=0.6),
                'gpu_usage': self._generate_batch_pattern(duration_days, base=0.8)
            },
            'real_time': {
                'description': 'Consistent real-time processing',
                'cpu_usage': self._generate_realtime_pattern(duration_days),
                'memory_usage': self._generate_realtime_pattern(duration_days, base=0.4),
                'gpu_usage': self._generate_realtime_pattern(duration_days, base=0.2)
            },
            'bursty': {
                'description': 'Bursty traffic with peaks and valleys',
                'cpu_usage': self._generate_bursty_pattern(duration_days),
                'memory_usage': self._generate_bursty_pattern(duration_days, base=0.5),
                'gpu_usage': self._generate_bursty_pattern(duration_days, base=0.3)
            }
        }
        
        return workload_patterns
    
    def _generate_batch_pattern(self, duration_days, base=0.3):
        """Generate batch processing workload pattern"""
        hours = duration_days * 24
        pattern = np.zeros(hours)
        
        for day in range(duration_days):
            # High usage during off-hours (2 AM - 6 AM)
            start_hour = day * 24 + 2
            end_hour = day * 24 + 6
            pattern[start_hour:end_hour] = np.random.uniform(0.8, 1.0, 4)
            
            # Low usage during business hours
            business_start = day * 24 + 9
            business_end = day * 24 + 17
            pattern[business_start:business_end] = np.random.uniform(base, base + 0.2, 8)
        
        return pattern
    
    def _generate_realtime_pattern(self, duration_days, base=0.4):
        """Generate real-time workload pattern"""
        hours = duration_days * 24
        pattern = np.full(hours, base)
        
        # Add some variation
        for i in range(hours):
            pattern[i] += np.random.normal(0, 0.1)
            pattern[i] = max(0, min(1, pattern[i]))  # Clamp to [0, 1]
        
        return pattern
    
    def _generate_bursty_pattern(self, duration_days, base=0.3):
        """Generate bursty workload pattern"""
        hours = duration_days * 24
        pattern = np.full(hours, base)
        
        # Add random bursts
        for day in range(duration_days):
            # Random burst during business hours
            burst_hour = day * 24 + np.random.randint(9, 17)
            burst_duration = np.random.randint(1, 4)
            
            for i in range(burst_duration):
                if burst_hour + i < hours:
                    pattern[burst_hour + i] = np.random.uniform(0.8, 1.0)
        
        return pattern
    
    def calculate_infrastructure_costs(self, workload_patterns, cost_models, 
                                     instance_specs):
        """Calculate costs for different infrastructure configurations"""
        print("💸 Calculating infrastructure costs...")
        
        cost_analysis = {}
        
        for workload_name, workload in workload_patterns.items():
            print(f"  Analyzing {workload_name} workload...")
            
            workload_costs = {}
            
            for cost_model_name, cost_model in cost_models.items():
                total_cost = 0
                cost_breakdown = {
                    'cpu_cost': 0,
                    'memory_cost': 0,
                    'gpu_cost': 0,
                    'storage_cost': 0
                }
                
                # Calculate costs for each hour
                for hour in range(len(workload['cpu_usage'])):
                    # CPU costs
                    cpu_usage = workload['cpu_usage'][hour]
                    cpu_cost = cpu_usage * instance_specs['cpu_cores'] * cost_model['cpu_cost_per_hour']
                    cost_breakdown['cpu_cost'] += cpu_cost
                    
                    # Memory costs
                    memory_usage = workload['memory_usage'][hour]
                    memory_cost = memory_usage * instance_specs['memory_gb'] * cost_model['memory_cost_per_gb_hour']
                    cost_breakdown['memory_cost'] += memory_cost
                    
                    # GPU costs
                    gpu_usage = workload['gpu_usage'][hour]
                    gpu_cost = gpu_usage * instance_specs['gpu_count'] * cost_model['gpu_cost_per_hour']
                    cost_breakdown['gpu_cost'] += gpu_cost
                    
                    # Storage costs (monthly)
                    storage_cost = instance_specs['storage_gb'] * cost_model['storage_cost_per_gb_month'] / 24
                    cost_breakdown['storage_cost'] += storage_cost
                
                total_cost = sum(cost_breakdown.values())
                
                workload_costs[cost_model_name] = {
                    'total_cost': total_cost,
                    'cost_breakdown': cost_breakdown,
                    'cost_per_hour': total_cost / len(workload['cpu_usage']),
                    'cost_per_day': total_cost / (len(workload['cpu_usage']) / 24)
                }
            
            cost_analysis[workload_name] = workload_costs
        
        return cost_analysis
    
    def optimize_auto_scaling(self, workload_patterns, cost_models, instance_specs):
        """Optimize auto-scaling for cost efficiency"""
        print("🔄 Optimizing auto-scaling strategies...")
        
        scaling_strategies = {
            'no_scaling': {'min_instances': 1, 'max_instances': 1, 'scale_up_threshold': 1.0, 'scale_down_threshold': 0.0},
            'conservative': {'min_instances': 1, 'max_instances': 3, 'scale_up_threshold': 0.8, 'scale_down_threshold': 0.3},
            'aggressive': {'min_instances': 0, 'max_instances': 10, 'scale_up_threshold': 0.6, 'scale_down_threshold': 0.2},
            'predictive': {'min_instances': 1, 'max_instances': 5, 'scale_up_threshold': 0.7, 'scale_down_threshold': 0.4}
        }
        
        scaling_results = {}
        
        for workload_name, workload in workload_patterns.items():
            print(f"  Optimizing scaling for {workload_name}...")
            
            workload_scaling = {}
            
            for strategy_name, strategy in scaling_strategies.items():
                # Simulate auto-scaling
                instances = self._simulate_auto_scaling(
                    workload['cpu_usage'], 
                    strategy,
                    instance_specs
                )
                
                # Calculate costs with scaling
                total_cost = 0
                for hour, instance_count in enumerate(instances):
                    hour_cost = 0
                    
                    # Calculate cost for this hour
                    cpu_usage = workload['cpu_usage'][hour] if hour < len(workload['cpu_usage']) else 0
                    memory_usage = workload['memory_usage'][hour] if hour < len(workload['memory_usage']) else 0
                    gpu_usage = workload['gpu_usage'][hour] if hour < len(workload['gpu_usage']) else 0
                    
                    # Use on-demand pricing for scaling
                    cost_model = cost_models['on_demand']
                    
                    cpu_cost = cpu_usage * instance_specs['cpu_cores'] * cost_model['cpu_cost_per_hour'] * instance_count
                    memory_cost = memory_usage * instance_specs['memory_gb'] * cost_model['memory_cost_per_gb_hour'] * instance_count
                    gpu_cost = gpu_usage * instance_specs['gpu_count'] * cost_model['gpu_cost_per_hour'] * instance_count
                    storage_cost = instance_specs['storage_gb'] * cost_model['storage_cost_per_gb_month'] / 24 * instance_count
                    
                    hour_cost = cpu_cost + memory_cost + gpu_cost + storage_cost
                    total_cost += hour_cost
                
                workload_scaling[strategy_name] = {
                    'total_cost': total_cost,
                    'avg_instances': np.mean(instances),
                    'max_instances': np.max(instances),
                    'scaling_events': self._count_scaling_events(instances),
                    'cost_per_hour': total_cost / len(instances)
                }
            
            scaling_results[workload_name] = workload_scaling
        
        return scaling_results
    
    def _simulate_auto_scaling(self, cpu_usage, strategy, instance_specs):
        """Simulate auto-scaling behavior"""
        instances = []
        current_instances = strategy['min_instances']
        
        for usage in cpu_usage:
            # Scale up if usage exceeds threshold
            if usage > strategy['scale_up_threshold'] and current_instances < strategy['max_instances']:
                current_instances = min(current_instances + 1, strategy['max_instances'])
            
            # Scale down if usage is below threshold
            elif usage < strategy['scale_down_threshold'] and current_instances > strategy['min_instances']:
                current_instances = max(current_instances - 1, strategy['min_instances'])
            
            instances.append(current_instances)
        
        return instances
    
    def _count_scaling_events(self, instances):
        """Count the number of scaling events"""
        events = 0
        for i in range(1, len(instances)):
            if instances[i] != instances[i-1]:
                events += 1
        return events
    
    def optimize_model_for_cost(self, model, test_data, test_labels):
        """Optimize model for cost efficiency"""
        print("🎯 Optimizing model for cost efficiency...")
        
        # Test different model configurations
        model_configs = {
            'baseline': {'n_estimators': 100, 'max_depth': 15},
            'reduced_trees': {'n_estimators': 50, 'max_depth': 15},
            'shallow_trees': {'n_estimators': 100, 'max_depth': 10},
            'minimal': {'n_estimators': 25, 'max_depth': 8}
        }
        
        model_results = {}
        
        for config_name, config in model_configs.items():
            print(f"  Testing {config_name} configuration...")
            
            # Create model with configuration
            test_model = RandomForestClassifier(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                random_state=42
            )
            
            # Train and evaluate
            start_time = time.time()
            test_model.fit(test_data, test_labels)
            training_time = time.time() - start_time
            
            start_time = time.time()
            predictions = test_model.predict(test_data)
            inference_time = time.time() - start_time
            
            accuracy = accuracy_score(test_labels, predictions)
            
            # Estimate costs (simplified)
            training_cost = training_time * 0.10  # $0.10 per second of training
            inference_cost = inference_time * 0.01  # $0.01 per second of inference
            
            model_results[config_name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'inference_time': inference_time,
                'training_cost': training_cost,
                'inference_cost': inference_cost,
                'total_cost': training_cost + inference_cost,
                'cost_per_prediction': inference_cost / len(predictions)
            }
        
        return model_results
    
    def run_comprehensive_demo(self):
        """Run comprehensive cost optimization demo"""
        print("🚀 Starting ML Infrastructure Cost Optimization Demo")
        print("=" * 60)
        
        # Create cost models
        cost_models = self.create_cost_models()
        
        # Define instance specifications
        instance_specs = {
            'cpu_cores': 8,
            'memory_gb': 32,
            'gpu_count': 1,
            'storage_gb': 100
        }
        
        # Simulate workload patterns
        workload_patterns = self.simulate_workload_patterns(duration_days=7)
        
        # Calculate infrastructure costs
        cost_analysis = self.calculate_infrastructure_costs(
            workload_patterns, cost_models, instance_specs
        )
        
        # Optimize auto-scaling
        scaling_results = self.optimize_auto_scaling(
            workload_patterns, cost_models, instance_specs
        )
        
        # Optimize model for cost
        X, y = make_classification(n_samples=1000, n_features=50, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        model_optimization = self.optimize_model_for_cost(
            RandomForestClassifier(), X_test_scaled, y_test
        )
        
        # Store results
        self.results = {
            'cost_analysis': cost_analysis,
            'scaling_results': scaling_results,
            'model_optimization': model_optimization,
            'workload_patterns': workload_patterns,
            'cost_models': cost_models
        }
        
        # Display results
        self.display_results()
        
        return self.results
    
    def display_results(self):
        """Display comprehensive results"""
        print("\n" + "=" * 80)
        print("📊 ML INFRASTRUCTURE COST OPTIMIZATION RESULTS")
        print("=" * 80)
        
        # Cost analysis by workload
        print("\n💰 COST ANALYSIS BY WORKLOAD")
        print("-" * 50)
        for workload_name, workload_costs in self.results['cost_analysis'].items():
            print(f"\n{workload_name.upper()}:")
            for cost_model, costs in workload_costs.items():
                print(f"  {cost_model:15} | Total: ${costs['total_cost']:8.2f} | "
                      f"Per Day: ${costs['cost_per_day']:6.2f}")
        
        # Auto-scaling optimization
        print("\n🔄 AUTO-SCALING OPTIMIZATION")
        print("-" * 50)
        for workload_name, scaling_costs in self.results['scaling_results'].items():
            print(f"\n{workload_name.upper()}:")
            for strategy, costs in scaling_costs.items():
                print(f"  {strategy:15} | Cost: ${costs['total_cost']:8.2f} | "
                      f"Avg Instances: {costs['avg_instances']:5.1f} | "
                      f"Scaling Events: {costs['scaling_events']:3d}")
        
        # Model optimization
        print("\n🎯 MODEL OPTIMIZATION")
        print("-" * 50)
        for config_name, results in self.results['model_optimization'].items():
            print(f"{config_name:15} | Accuracy: {results['accuracy']:6.4f} | "
                  f"Cost: ${results['total_cost']:6.4f} | "
                  f"Cost/Prediction: ${results['cost_per_prediction']:8.6f}")
        
        # Cost optimization recommendations
        self.print_cost_recommendations()
    
    def print_cost_recommendations(self):
        """Print cost optimization recommendations"""
        print(f"\n💡 COST OPTIMIZATION RECOMMENDATIONS")
        print("-" * 50)
        print("• Use reserved instances for predictable workloads")
        print("• Implement spot instances for fault-tolerant workloads")
        print("• Optimize auto-scaling thresholds for your traffic patterns")
        print("• Consider model compression for inference cost reduction")
        print("• Monitor and optimize resource utilization continuously")
        print("• Use right-sizing to match instance capacity to workload")
        print("• Implement cost alerts and budgets")
        print("• Regular cost reviews and optimization cycles")
    
    def plot_cost_analysis(self):
        """Create cost analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ML Infrastructure Cost Analysis', fontsize=16)
        
        # Plot 1: Cost comparison by workload
        ax1 = axes[0, 0]
        workloads = list(self.results['cost_analysis'].keys())
        cost_models = list(self.results['cost_analysis'][workloads[0]].keys())
        
        x = np.arange(len(workloads))
        width = 0.2
        
        for i, cost_model in enumerate(cost_models):
            costs = [self.results['cost_analysis'][w][cost_model]['total_cost'] for w in workloads]
            ax1.bar(x + i * width, costs, width, label=cost_model)
        
        ax1.set_xlabel('Workload Type')
        ax1.set_ylabel('Total Cost ($)')
        ax1.set_title('Cost Comparison by Workload and Pricing Model')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(workloads)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Auto-scaling cost comparison
        ax2 = axes[0, 1]
        strategies = list(self.results['scaling_results'][workloads[0]].keys())
        
        for workload in workloads:
            costs = [self.results['scaling_results'][workload][s]['total_cost'] for s in strategies]
            ax2.plot(strategies, costs, marker='o', label=workload)
        
        ax2.set_xlabel('Scaling Strategy')
        ax2.set_ylabel('Total Cost ($)')
        ax2.set_title('Auto-scaling Cost Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Model optimization trade-offs
        ax3 = axes[1, 0]
        configs = list(self.results['model_optimization'].keys())
        accuracies = [self.results['model_optimization'][c]['accuracy'] for c in configs]
        costs = [self.results['model_optimization'][c]['total_cost'] for c in configs]
        
        scatter = ax3.scatter(costs, accuracies, s=100, alpha=0.7)
        for i, config in enumerate(configs):
            ax3.annotate(config, (costs[i], accuracies[i]), xytext=(5, 5), textcoords='offset points')
        
        ax3.set_xlabel('Total Cost ($)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Model Optimization: Cost vs Accuracy')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Workload patterns
        ax4 = axes[1, 1]
        workload = self.results['workload_patterns']['real_time']
        hours = range(len(workload['cpu_usage']))
        
        ax4.plot(hours, workload['cpu_usage'], label='CPU Usage', alpha=0.7)
        ax4.plot(hours, workload['memory_usage'], label='Memory Usage', alpha=0.7)
        ax4.plot(hours, workload['gpu_usage'], label='GPU Usage', alpha=0.7)
        
        ax4.set_xlabel('Hour')
        ax4.set_ylabel('Resource Usage')
        ax4.set_title('Workload Pattern (Real-time)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Run the cost optimization demo"""
    demo = CostOptimizationDemo()
    results = demo.run_comprehensive_demo()
    
    # Create visualizations
    print("\n📊 Creating cost analysis visualizations...")
    demo.plot_cost_analysis()
    
    print("\n✅ Cost Optimization Demo Complete!")
    print("\nKey Takeaways for ML Infrastructure Engineers:")
    print("• Cost optimization requires understanding workload patterns")
    print("• Auto-scaling can significantly reduce costs for variable workloads")
    print("• Model optimization can reduce inference costs")
    print("• Reserved instances provide savings for predictable workloads")
    print("• Regular cost monitoring and optimization is essential")

if __name__ == "__main__":
    main()
