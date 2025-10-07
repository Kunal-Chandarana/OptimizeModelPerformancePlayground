"""
Model Serving Optimization for ML Infrastructure
===============================================

This demo focuses on optimizing model serving for production environments,
covering serving strategies, load balancing, and resource management.

Key Topics:
- Model serving architectures (REST, gRPC, streaming)
- Load balancing and auto-scaling
- Resource utilization optimization
- Latency vs throughput trade-offs
- Batch vs real-time inference
"""

import time
import threading
import queue
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import requests
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelServingDemo:
    def __init__(self):
        self.results = {}
        self.metrics = {
            'requests_per_second': [],
            'average_latency': [],
            'p95_latency': [],
            'cpu_usage': [],
            'memory_usage': [],
            'error_rate': []
        }
    
    def create_production_model(self):
        """Create a model optimized for production serving"""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=10000,
            n_features=50,
            n_informative=40,
            n_redundant=10,
            n_classes=3,
            random_state=42
        )
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        return model, scaler
    
    def simulate_rest_api_serving(self, model, scaler, num_requests=1000, 
                                concurrent_users=10, batch_size=1):
        """Simulate REST API model serving"""
        print(f"🌐 Testing REST API Serving (Concurrent Users: {concurrent_users}, Batch Size: {batch_size})")
        
        # Generate test data
        X_test, y_test = make_classification(
            n_samples=num_requests,
            n_features=50,
            n_informative=40,
            n_redundant=10,
            n_classes=3,
            random_state=42
        )
        
        def process_request(request_data):
            """Process a single request"""
            start_time = time.time()
            
            try:
                # Simulate preprocessing
                X_scaled = scaler.transform([request_data])
                
                # Make prediction
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0].max()
                
                processing_time = time.time() - start_time
                
                return {
                    'success': True,
                    'prediction': prediction,
                    'confidence': probability,
                    'latency': processing_time,
                    'error': None
                }
            except Exception as e:
                processing_time = time.time() - start_time
                return {
                    'success': False,
                    'prediction': None,
                    'confidence': None,
                    'latency': processing_time,
                    'error': str(e)
                }
        
        def worker(requests_queue, results_queue):
            """Worker thread for processing requests"""
            while True:
                try:
                    request_data = requests_queue.get(timeout=1)
                    result = process_request(request_data)
                    results_queue.put(result)
                    requests_queue.task_done()
                except queue.Empty:
                    break
        
        # Create queues
        requests_queue = queue.Queue()
        results_queue = queue.Queue()
        
        # Add requests to queue
        for i in range(num_requests):
            requests_queue.put(X_test[i])
        
        # Start worker threads
        start_time = time.time()
        workers = []
        for _ in range(concurrent_users):
            worker_thread = threading.Thread(target=worker, args=(requests_queue, results_queue))
            worker_thread.start()
            workers.append(worker_thread)
        
        # Collect results
        results = []
        while len(results) < num_requests:
            try:
                result = results_queue.get(timeout=0.1)
                results.append(result)
            except queue.Empty:
                continue
        
        # Wait for all workers to finish
        for worker in workers:
            worker.join()
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        latencies = [r['latency'] for r in successful_requests]
        
        metrics = {
            'total_requests': num_requests,
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / num_requests,
            'requests_per_second': num_requests / total_time,
            'average_latency': np.mean(latencies) if latencies else 0,
            'p95_latency': np.percentile(latencies, 95) if latencies else 0,
            'p99_latency': np.percentile(latencies, 99) if latencies else 0,
            'total_time': total_time,
            'concurrent_users': concurrent_users,
            'batch_size': batch_size
        }
        
        return metrics
    
    def simulate_batch_processing(self, model, scaler, batch_sizes=[1, 10, 50, 100]):
        """Simulate batch processing with different batch sizes"""
        print(f"📦 Testing Batch Processing")
        
        # Generate test data
        X_test, y_test = make_classification(
            n_samples=1000,
            n_features=50,
            n_informative=40,
            n_redundant=10,
            n_classes=3,
            random_state=42
        )
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            start_time = time.time()
            
            # Process in batches
            predictions = []
            for i in range(0, len(X_test), batch_size):
                batch = X_test[i:i+batch_size]
                X_scaled = scaler.transform(batch)
                batch_predictions = model.predict(X_scaled)
                predictions.extend(batch_predictions)
            
            total_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            
            batch_results[batch_size] = {
                'batch_size': batch_size,
                'total_samples': len(X_test),
                'total_time': total_time,
                'samples_per_second': len(X_test) / total_time,
                'accuracy': accuracy,
                'latency_per_sample': total_time / len(X_test)
            }
            
            print(f"    Samples/sec: {len(X_test) / total_time:.2f}, Accuracy: {accuracy:.4f}")
        
        return batch_results
    
    def test_load_balancing_scenarios(self, model, scaler):
        """Test different load balancing scenarios"""
        print(f"⚖️ Testing Load Balancing Scenarios")
        
        scenarios = {
            'Single Instance': {'instances': 1, 'concurrent_users': 10},
            'Load Balanced (2 instances)': {'instances': 2, 'concurrent_users': 20},
            'Load Balanced (5 instances)': {'instances': 5, 'concurrent_users': 50},
            'High Load (10 instances)': {'instances': 10, 'concurrent_users': 100}
        }
        
        load_balancing_results = {}
        
        for scenario_name, config in scenarios.items():
            print(f"  Testing {scenario_name}")
            
            # Simulate load balancing by distributing requests across instances
            requests_per_instance = 100 // config['instances']
            concurrent_per_instance = config['concurrent_users'] // config['instances']
            
            instance_results = []
            
            for instance_id in range(config['instances']):
                instance_metrics = self.simulate_rest_api_serving(
                    model, scaler, 
                    num_requests=requests_per_instance,
                    concurrent_users=concurrent_per_instance
                )
                instance_results.append(instance_metrics)
            
            # Aggregate results
            total_requests = sum(r['total_requests'] for r in instance_results)
            total_successful = sum(r['successful_requests'] for r in instance_results)
            total_time = max(r['total_time'] for r in instance_results)  # Max time across instances
            
            avg_latency = np.mean([r['average_latency'] for r in instance_results])
            p95_latency = np.mean([r['p95_latency'] for r in instance_results])
            
            load_balancing_results[scenario_name] = {
                'instances': config['instances'],
                'concurrent_users': config['concurrent_users'],
                'total_requests': total_requests,
                'successful_requests': total_successful,
                'success_rate': total_successful / total_requests,
                'requests_per_second': total_requests / total_time,
                'average_latency': avg_latency,
                'p95_latency': p95_latency,
                'total_time': total_time
            }
            
            print(f"    RPS: {total_requests / total_time:.2f}, Latency: {avg_latency:.4f}s")
        
        return load_balancing_results
    
    def test_resource_utilization(self, model, scaler):
        """Test resource utilization patterns"""
        print(f"💻 Testing Resource Utilization")
        
        # Monitor system resources during model serving
        def monitor_resources(duration=10):
            """Monitor CPU and memory usage"""
            cpu_usage = []
            memory_usage = []
            
            start_time = time.time()
            while time.time() - start_time < duration:
                cpu_usage.append(psutil.cpu_percent())
                memory_usage.append(psutil.virtual_memory().percent)
                time.sleep(0.1)
            
            return {
                'avg_cpu': np.mean(cpu_usage),
                'max_cpu': np.max(cpu_usage),
                'avg_memory': np.mean(memory_usage),
                'max_memory': np.max(memory_usage)
            }
        
        # Test different load levels
        load_levels = [10, 50, 100, 200]
        resource_results = {}
        
        for load in load_levels:
            print(f"  Testing load level: {load} concurrent users")
            
            # Start resource monitoring
            monitor_thread = threading.Thread(target=lambda: monitor_resources(5))
            monitor_thread.start()
            
            # Run load test
            metrics = self.simulate_rest_api_serving(
                model, scaler,
                num_requests=load * 10,  # 10 requests per user
                concurrent_users=load
            )
            
            monitor_thread.join()
            
            resource_results[f'{load} users'] = {
                'concurrent_users': load,
                'requests_per_second': metrics['requests_per_second'],
                'average_latency': metrics['average_latency'],
                'success_rate': metrics['success_rate']
            }
        
        return resource_results
    
    def test_auto_scaling_simulation(self, model, scaler):
        """Simulate auto-scaling based on load"""
        print(f"🔄 Testing Auto-Scaling Simulation")
        
        # Simulate traffic patterns
        traffic_patterns = {
            'Low Traffic': {'base_load': 10, 'peak_load': 20, 'duration': 30},
            'Medium Traffic': {'base_load': 50, 'peak_load': 100, 'duration': 30},
            'High Traffic': {'base_load': 100, 'peak_load': 300, 'duration': 30}
        }
        
        scaling_results = {}
        
        for pattern_name, pattern in traffic_patterns.items():
            print(f"  Testing {pattern_name}")
            
            # Simulate traffic increase
            base_metrics = self.simulate_rest_api_serving(
                model, scaler,
                num_requests=pattern['base_load'] * 5,
                concurrent_users=pattern['base_load']
            )
            
            peak_metrics = self.simulate_rest_api_serving(
                model, scaler,
                num_requests=pattern['peak_load'] * 5,
                concurrent_users=pattern['peak_load']
            )
            
            # Calculate scaling efficiency
            base_rps = base_metrics['requests_per_second']
            peak_rps = peak_metrics['requests_per_second']
            scaling_factor = peak_rps / base_rps if base_rps > 0 else 0
            
            scaling_results[pattern_name] = {
                'base_load': pattern['base_load'],
                'peak_load': pattern['peak_load'],
                'base_rps': base_rps,
                'peak_rps': peak_rps,
                'scaling_factor': scaling_factor,
                'base_latency': base_metrics['average_latency'],
                'peak_latency': peak_metrics['average_latency'],
                'latency_degradation': (peak_metrics['average_latency'] - base_metrics['average_latency']) / base_metrics['average_latency'] if base_metrics['average_latency'] > 0 else 0
            }
            
            print(f"    Scaling factor: {scaling_factor:.2f}x, Latency degradation: {scaling_results[pattern_name]['latency_degradation']:.2%}")
        
        return scaling_results
    
    def run_comprehensive_demo(self):
        """Run comprehensive model serving optimization demo"""
        print("🚀 Starting Model Serving Optimization Demo")
        print("=" * 60)
        
        # Create production model
        print("🤖 Creating production model...")
        model, scaler = self.create_production_model()
        print(f"Model trained with {model.n_estimators} estimators")
        
        # Test different serving strategies
        print("\n📊 Testing Serving Strategies...")
        serving_results = {}
        
        # REST API serving with different configurations
        configs = [
            {'concurrent_users': 10, 'batch_size': 1, 'name': 'Low Load'},
            {'concurrent_users': 50, 'batch_size': 1, 'name': 'Medium Load'},
            {'concurrent_users': 100, 'batch_size': 1, 'name': 'High Load'},
            {'concurrent_users': 50, 'batch_size': 10, 'name': 'Batch Processing'}
        ]
        
        for config in configs:
            metrics = self.simulate_rest_api_serving(
                model, scaler,
                num_requests=500,
                concurrent_users=config['concurrent_users'],
                batch_size=config['batch_size']
            )
            serving_results[config['name']] = metrics
        
        # Test batch processing
        batch_results = self.simulate_batch_processing(model, scaler)
        
        # Test load balancing
        load_balancing_results = self.test_load_balancing_scenarios(model, scaler)
        
        # Test resource utilization
        resource_results = self.test_resource_utilization(model, scaler)
        
        # Test auto-scaling
        scaling_results = self.test_auto_scaling_simulation(model, scaler)
        
        # Store all results
        self.results = {
            'serving_strategies': serving_results,
            'batch_processing': batch_results,
            'load_balancing': load_balancing_results,
            'resource_utilization': resource_results,
            'auto_scaling': scaling_results
        }
        
        # Display results
        self.display_results()
        
        return self.results
    
    def display_results(self):
        """Display comprehensive results"""
        print("\n" + "=" * 80)
        print("📊 MODEL SERVING OPTIMIZATION RESULTS")
        print("=" * 80)
        
        # Serving strategies
        print("\n🌐 SERVING STRATEGIES")
        print("-" * 50)
        for strategy, metrics in self.results['serving_strategies'].items():
            print(f"{strategy:20} | RPS: {metrics['requests_per_second']:6.2f} | "
                  f"Latency: {metrics['average_latency']:6.4f}s | "
                  f"Success: {metrics['success_rate']:6.2%}")
        
        # Batch processing
        print("\n📦 BATCH PROCESSING")
        print("-" * 50)
        for batch_size, metrics in self.results['batch_processing'].items():
            print(f"Batch Size {batch_size:3} | Samples/sec: {metrics['samples_per_second']:6.2f} | "
                  f"Latency: {metrics['latency_per_sample']:6.4f}s | "
                  f"Accuracy: {metrics['accuracy']:6.4f}")
        
        # Load balancing
        print("\n⚖️ LOAD BALANCING")
        print("-" * 50)
        for scenario, metrics in self.results['load_balancing'].items():
            print(f"{scenario:25} | RPS: {metrics['requests_per_second']:6.2f} | "
                  f"Latency: {metrics['average_latency']:6.4f}s | "
                  f"Instances: {metrics['instances']:2}")
        
        # Auto-scaling
        print("\n🔄 AUTO-SCALING")
        print("-" * 50)
        for pattern, metrics in self.results['auto_scaling'].items():
            print(f"{pattern:15} | Scaling: {metrics['scaling_factor']:5.2f}x | "
                  f"Latency Degradation: {metrics['latency_degradation']:6.2%}")
        
        # Find best configurations
        self.find_optimal_configurations()
    
    def find_optimal_configurations(self):
        """Find optimal serving configurations"""
        print("\n🏆 OPTIMAL CONFIGURATIONS")
        print("-" * 50)
        
        # Best RPS
        best_rps = max(
            [(name, metrics['requests_per_second']) for name, metrics in self.results['serving_strategies'].items()],
            key=lambda x: x[1]
        )
        print(f"Best RPS: {best_rps[0]} ({best_rps[1]:.2f} requests/sec)")
        
        # Best latency
        best_latency = min(
            [(name, metrics['average_latency']) for name, metrics in self.results['serving_strategies'].items()],
            key=lambda x: x[1]
        )
        print(f"Best Latency: {best_latency[0]} ({best_latency[1]:.4f}s)")
        
        # Best batch size
        best_batch = max(
            [(size, metrics['samples_per_second']) for size, metrics in self.results['batch_processing'].items()],
            key=lambda x: x[1]
        )
        print(f"Best Batch Size: {best_batch[0]} ({best_batch[1]:.2f} samples/sec)")
        
        print("\n💡 INFRASTRUCTURE RECOMMENDATIONS:")
        print("-" * 50)
        print("• Use load balancing for high availability")
        print("• Implement auto-scaling based on CPU/memory usage")
        print("• Consider batch processing for non-real-time use cases")
        print("• Monitor latency percentiles (P95, P99) for SLA compliance")
        print("• Use circuit breakers and retry logic for fault tolerance")
    
    def plot_results(self):
        """Create visualizations of the results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Serving Optimization Analysis', fontsize=16)
        
        # Plot 1: RPS vs Latency
        ax1 = axes[0, 0]
        strategies = list(self.results['serving_strategies'].keys())
        rps_values = [self.results['serving_strategies'][s]['requests_per_second'] for s in strategies]
        latency_values = [self.results['serving_strategies'][s]['average_latency'] for s in strategies]
        
        ax1.scatter(rps_values, latency_values, s=100, alpha=0.7)
        for i, strategy in enumerate(strategies):
            ax1.annotate(strategy, (rps_values[i], latency_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax1.set_xlabel('Requests Per Second')
        ax1.set_ylabel('Average Latency (s)')
        ax1.set_title('RPS vs Latency Trade-off')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Batch Size Performance
        ax2 = axes[0, 1]
        batch_sizes = list(self.results['batch_processing'].keys())
        samples_per_sec = [self.results['batch_processing'][s]['samples_per_second'] for s in batch_sizes]
        
        ax2.plot(batch_sizes, samples_per_sec, marker='o', linewidth=2, markersize=8)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Samples Per Second')
        ax2.set_title('Batch Processing Performance')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Load Balancing Scaling
        ax3 = axes[0, 2]
        scenarios = list(self.results['load_balancing'].keys())
        instances = [self.results['load_balancing'][s]['instances'] for s in scenarios]
        rps_values = [self.results['load_balancing'][s]['requests_per_second'] for s in scenarios]
        
        ax3.plot(instances, rps_values, marker='o', linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Instances')
        ax3.set_ylabel('Total RPS')
        ax3.set_title('Load Balancing Scaling')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Auto-scaling Performance
        ax4 = axes[1, 0]
        patterns = list(self.results['auto_scaling'].keys())
        scaling_factors = [self.results['auto_scaling'][p]['scaling_factor'] for p in patterns]
        
        bars = ax4.bar(patterns, scaling_factors)
        ax4.set_ylabel('Scaling Factor')
        ax4.set_title('Auto-scaling Performance')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Latency Distribution
        ax5 = axes[1, 1]
        # Simulate latency distribution for visualization
        latencies = np.random.exponential(0.01, 1000)  # Exponential distribution
        ax5.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Latency (s)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Latency Distribution')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Resource Utilization
        ax6 = axes[1, 2]
        # Simulate resource utilization
        cpu_usage = np.random.normal(70, 15, 100)
        memory_usage = np.random.normal(60, 10, 100)
        
        ax6.scatter(cpu_usage, memory_usage, alpha=0.6)
        ax6.set_xlabel('CPU Usage (%)')
        ax6.set_ylabel('Memory Usage (%)')
        ax6.set_title('Resource Utilization')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Run the model serving optimization demo"""
    demo = ModelServingDemo()
    results = demo.run_comprehensive_demo()
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    demo.plot_results()
    
    print("\n✅ Model Serving Optimization Demo Complete!")
    print("\nKey Takeaways for ML Infrastructure Engineers:")
    print("• Load balancing significantly improves throughput")
    print("• Batch processing can dramatically increase efficiency")
    print("• Auto-scaling helps handle traffic spikes")
    print("• Monitor both average and percentile latencies")
    print("• Resource utilization patterns inform capacity planning")

if __name__ == "__main__":
    main()

