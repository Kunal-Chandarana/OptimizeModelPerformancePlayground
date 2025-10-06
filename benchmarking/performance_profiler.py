"""
Performance Profiler for ML Model Optimization
==============================================

This comprehensive profiler helps you measure and analyze the performance
of your ML models across different optimization techniques.

Key Features:
- Memory usage tracking
- CPU/GPU utilization monitoring
- Training time profiling
- Inference speed measurement
- Model size analysis
- Performance regression detection
- Automated benchmarking
- Performance visualization
"""

import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from typing import Dict, List, Any, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.profiler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class PerformanceProfiler:
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        self.results = {}
        self.baseline_results = {}
        self.current_session = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize system info
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            'timestamp': datetime.now().isoformat()
        }
        
        if TORCH_AVAILABLE:
            info['pytorch_version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_version'] = torch.version.cuda
                info['gpu_count'] = torch.cuda.device_count()
                info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if TF_AVAILABLE:
            info['tensorflow_version'] = tf.__version__
            info['tensorflow_gpu_available'] = len(tf.config.list_physical_devices('GPU')) > 0
        
        return info
    
    def start_session(self, session_name: str, description: str = ""):
        """Start a new profiling session"""
        self.current_session = {
            'name': session_name,
            'description': description,
            'start_time': time.time(),
            'start_memory': psutil.virtual_memory().used / (1024**2),  # MB
            'measurements': []
        }
        print(f"🚀 Started profiling session: {session_name}")
        if description:
            print(f"   Description: {description}")
    
    def end_session(self):
        """End the current profiling session"""
        if not self.current_session:
            print("⚠️ No active session to end")
            return
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024**2)  # MB
        
        self.current_session['end_time'] = end_time
        self.current_session['duration'] = end_time - self.current_session['start_time']
        self.current_session['end_memory'] = end_memory
        self.current_session['memory_delta'] = end_memory - self.current_session['start_memory']
        
        # Save session results
        self.results[self.current_session['name']] = self.current_session.copy()
        
        print(f"✅ Ended profiling session: {self.current_session['name']}")
        print(f"   Duration: {self.current_session['duration']:.2f} seconds")
        print(f"   Memory delta: {self.current_session['memory_delta']:.2f} MB")
        
        self.current_session = None
    
    def measure_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Measure the performance of a function call"""
        if not self.current_session:
            print("⚠️ No active session. Starting default session...")
            self.start_session("default", "Default profiling session")
        
        # Measure memory before
        memory_before = psutil.virtual_memory().used / (1024**2)
        cpu_before = psutil.cpu_percent()
        
        # Measure execution time
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        end_time = time.time()
        
        # Measure memory after
        memory_after = psutil.virtual_memory().used / (1024**2)
        cpu_after = psutil.cpu_percent()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_delta = memory_after - memory_before
        cpu_usage = (cpu_before + cpu_after) / 2
        
        measurement = {
            'function_name': func.__name__,
            'execution_time': execution_time,
            'memory_delta_mb': memory_delta,
            'cpu_usage_percent': cpu_usage,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to current session
        if self.current_session:
            self.current_session['measurements'].append(measurement)
        
        return {
            'result': result,
            'metrics': measurement
        }
    
    def profile_training(self, model, train_loader, optimizer, criterion, 
                        num_epochs: int = 10, device: str = 'cpu') -> Dict[str, Any]:
        """Profile model training performance"""
        print(f"🏋️ Profiling training for {num_epochs} epochs...")
        
        model.to(device)
        training_metrics = {
            'epoch_times': [],
            'epoch_memory': [],
            'epoch_cpu': [],
            'total_time': 0,
            'total_memory_delta': 0,
            'average_cpu_usage': 0
        }
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2)
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_memory_start = psutil.virtual_memory().used / (1024**2)
            epoch_cpu_start = psutil.cpu_percent()
            
            # Training loop
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            epoch_end = time.time()
            epoch_memory_end = psutil.virtual_memory().used / (1024**2)
            epoch_cpu_end = psutil.cpu_percent()
            
            # Record epoch metrics
            epoch_time = epoch_end - epoch_start
            epoch_memory_delta = epoch_memory_end - epoch_memory_start
            epoch_cpu_avg = (epoch_cpu_start + epoch_cpu_end) / 2
            
            training_metrics['epoch_times'].append(epoch_time)
            training_metrics['epoch_memory'].append(epoch_memory_delta)
            training_metrics['epoch_cpu'].append(epoch_cpu_avg)
            
            print(f"   Epoch {epoch+1}/{num_epochs}: {epoch_time:.2f}s, "
                  f"Memory: {epoch_memory_delta:.2f}MB, CPU: {epoch_cpu_avg:.1f}%")
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024**2)
        
        # Calculate total metrics
        training_metrics['total_time'] = end_time - start_time
        training_metrics['total_memory_delta'] = end_memory - start_memory
        training_metrics['average_cpu_usage'] = np.mean(training_metrics['epoch_cpu'])
        training_metrics['average_epoch_time'] = np.mean(training_metrics['epoch_times'])
        training_metrics['std_epoch_time'] = np.std(training_metrics['epoch_times'])
        
        return training_metrics
    
    def profile_inference(self, model, test_loader, num_runs: int = 100, 
                         device: str = 'cpu') -> Dict[str, Any]:
        """Profile model inference performance"""
        print(f"⚡ Profiling inference for {num_runs} runs...")
        
        model.to(device)
        model.eval()
        
        inference_times = []
        memory_usage = []
        cpu_usage = []
        
        # Warm up
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                _ = model(data)
                break
        
        # Profile inference
        for run in range(num_runs):
            run_start = time.time()
            run_memory_start = psutil.virtual_memory().used / (1024**2)
            run_cpu_start = psutil.cpu_percent()
            
            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.to(device)
                    _ = model(data)
            
            run_end = time.time()
            run_memory_end = psutil.virtual_memory().used / (1024**2)
            run_cpu_end = psutil.cpu_percent()
            
            inference_times.append(run_end - run_start)
            memory_usage.append(run_memory_end - run_memory_start)
            cpu_usage.append((run_cpu_start + run_cpu_end) / 2)
        
        return {
            'inference_times': inference_times,
            'memory_usage': memory_usage,
            'cpu_usage': cpu_usage,
            'average_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'throughput_samples_per_second': len(test_loader.dataset) / np.mean(inference_times)
        }
    
    def measure_model_size(self, model) -> Dict[str, Any]:
        """Measure model size and complexity"""
        if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            return self._measure_pytorch_model_size(model)
        else:
            return self._measure_generic_model_size(model)
    
    def _measure_pytorch_model_size(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Measure PyTorch model size"""
        param_size = 0
        buffer_size = 0
        param_count = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_count += param.nelement()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size_mb = (param_size + buffer_size) / (1024**2)
        
        return {
            'total_size_mb': total_size_mb,
            'param_size_mb': param_size / (1024**2),
            'buffer_size_mb': buffer_size / (1024**2),
            'total_parameters': param_count,
            'model_size_bytes': param_size + buffer_size
        }
    
    def _measure_generic_model_size(self, model) -> Dict[str, Any]:
        """Measure generic model size (fallback)"""
        import sys
        
        size_bytes = sys.getsizeof(model)
        
        return {
            'total_size_mb': size_bytes / (1024**2),
            'model_size_bytes': size_bytes,
            'total_parameters': 'Unknown',
            'param_size_mb': 'Unknown',
            'buffer_size_mb': 'Unknown'
        }
    
    def compare_with_baseline(self, current_results: Dict[str, Any], 
                            baseline_name: str = "baseline") -> Dict[str, Any]:
        """Compare current results with baseline"""
        if baseline_name not in self.baseline_results:
            print(f"⚠️ No baseline found: {baseline_name}")
            return {}
        
        baseline = self.baseline_results[baseline_name]
        comparison = {}
        
        # Compare common metrics
        for metric in ['execution_time', 'memory_delta_mb', 'cpu_usage_percent']:
            if metric in current_results and metric in baseline:
                current_val = current_results[metric]
                baseline_val = baseline[metric]
                
                if baseline_val != 0:
                    improvement = ((current_val - baseline_val) / baseline_val) * 100
                    comparison[f'{metric}_improvement_percent'] = improvement
                    comparison[f'{metric}_baseline'] = baseline_val
                    comparison[f'{metric}_current'] = current_val
        
        return comparison
    
    def set_baseline(self, baseline_name: str, results: Dict[str, Any]):
        """Set baseline results for comparison"""
        self.baseline_results[baseline_name] = results
        print(f"📊 Set baseline: {baseline_name}")
    
    def generate_report(self, session_name: str = None) -> str:
        """Generate a comprehensive performance report"""
        if session_name:
            sessions = {session_name: self.results[session_name]} if session_name in self.results else {}
        else:
            sessions = self.results
        
        if not sessions:
            return "No profiling data available"
        
        report = []
        report.append("=" * 80)
        report.append("ML MODEL PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"System: {self.system_info['cpu_count']} CPUs, "
                     f"{self.system_info['memory_total_gb']:.1f}GB RAM")
        report.append("")
        
        for session_name, session_data in sessions.items():
            report.append(f"SESSION: {session_name}")
            report.append("-" * 40)
            report.append(f"Description: {session_data.get('description', 'N/A')}")
            report.append(f"Duration: {session_data.get('duration', 0):.2f} seconds")
            report.append(f"Memory Delta: {session_data.get('memory_delta', 0):.2f} MB")
            report.append("")
            
            if session_data.get('measurements'):
                report.append("MEASUREMENTS:")
                for i, measurement in enumerate(session_data['measurements'], 1):
                    report.append(f"  {i}. {measurement['function_name']}")
                    report.append(f"     Time: {measurement['execution_time']:.4f}s")
                    report.append(f"     Memory: {measurement['memory_delta_mb']:.2f}MB")
                    report.append(f"     CPU: {measurement['cpu_usage_percent']:.1f}%")
                    report.append(f"     Success: {measurement['success']}")
                    if measurement['error']:
                        report.append(f"     Error: {measurement['error']}")
                    report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """Save profiling results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profiling_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare data for JSON serialization
        serializable_results = {}
        for session_name, session_data in self.results.items():
            serializable_results[session_name] = {
                'name': session_data['name'],
                'description': session_data.get('description', ''),
                'start_time': session_data['start_time'],
                'end_time': session_data.get('end_time', 0),
                'duration': session_data.get('duration', 0),
                'start_memory': session_data.get('start_memory', 0),
                'end_memory': session_data.get('end_memory', 0),
                'memory_delta': session_data.get('memory_delta', 0),
                'measurements': session_data.get('measurements', [])
            }
        
        data = {
            'system_info': self.system_info,
            'baseline_results': self.baseline_results,
            'profiling_results': serializable_results,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Results saved to: {filepath}")
        return filepath
    
    def load_results(self, filename: str):
        """Load profiling results from file"""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.system_info = data.get('system_info', {})
        self.baseline_results = data.get('baseline_results', {})
        self.results = data.get('profiling_results', {})
        
        print(f"📂 Results loaded from: {filepath}")
    
    def plot_performance_comparison(self, metrics: List[str] = None):
        """Create performance comparison plots"""
        if not self.results:
            print("No data to plot")
            return
        
        # Default metrics to plot
        if not metrics:
            metrics = ['execution_time', 'memory_delta_mb', 'cpu_usage_percent']
        
        # Prepare data for plotting
        plot_data = []
        for session_name, session_data in self.results.items():
            for measurement in session_data.get('measurements', []):
                row = {'session': session_name, 'function': measurement['function_name']}
                for metric in metrics:
                    if metric in measurement:
                        row[metric] = measurement[metric]
                plot_data.append(row)
        
        if not plot_data:
            print("No measurement data to plot")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                # Group by session and function
                pivot_data = df.pivot_table(
                    values=metric, 
                    index='function', 
                    columns='session', 
                    aggfunc='mean'
                )
                
                pivot_data.plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Function')
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].legend(title='Session')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def detect_performance_regression(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Detect performance regressions compared to baseline"""
        regressions = []
        
        for session_name, session_data in self.results.items():
            if session_name in self.baseline_results:
                baseline = self.baseline_results[session_name]
                
                for measurement in session_data.get('measurements', []):
                    function_name = measurement['function_name']
                    
                    # Find corresponding baseline measurement
                    baseline_measurement = None
                    for bm in baseline.get('measurements', []):
                        if bm['function_name'] == function_name:
                            baseline_measurement = bm
                            break
                    
                    if baseline_measurement:
                        # Check for regressions
                        for metric in ['execution_time', 'memory_delta_mb']:
                            if metric in measurement and metric in baseline_measurement:
                                current_val = measurement[metric]
                                baseline_val = baseline_measurement[metric]
                                
                                if baseline_val > 0:
                                    regression = (current_val - baseline_val) / baseline_val
                                    
                                    if regression > threshold:
                                        regressions.append({
                                            'session': session_name,
                                            'function': function_name,
                                            'metric': metric,
                                            'baseline': baseline_val,
                                            'current': current_val,
                                            'regression_percent': regression * 100
                                        })
        
        return regressions

def main():
    """Example usage of the PerformanceProfiler"""
    profiler = PerformanceProfiler()
    
    # Example function to profile
    def example_function(n=1000000):
        """Example function that does some computation"""
        result = sum(i**2 for i in range(n))
        return result
    
    # Start profiling session
    profiler.start_session("example_session", "Testing example function performance")
    
    # Profile the function
    result = profiler.measure_function(example_function, 1000000)
    print(f"Function result: {result['result']}")
    print(f"Execution time: {result['metrics']['execution_time']:.4f}s")
    
    # End session
    profiler.end_session()
    
    # Generate report
    report = profiler.generate_report()
    print("\n" + report)
    
    # Save results
    profiler.save_results()

if __name__ == "__main__":
    main()
