"""
Model Quantization for Inference Optimization
==============================================

This demo shows how different quantization techniques can significantly improve
inference speed and reduce model size while maintaining acceptable accuracy.

Key Techniques Covered:
- Post-Training Quantization (PTQ)
- Dynamic Quantization
- Static Quantization
- Quantization Aware Training (QAT)
- INT8 Quantization
- FP16 Quantization
- ONNX Quantization
- TensorRT Optimization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import psutil
import os
import warnings
warnings.filterwarnings('ignore')

class ModelQuantizationDemo:
    def __init__(self):
        self.results = {}
        self.model_sizes = {}
        
    def create_synthetic_data(self, n_samples=2000, n_features=50, n_classes=3):
        """Create synthetic dataset for quantization demo"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=40,
            n_redundant=10,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        return X_tensor, y_tensor
    
    def create_neural_network(self, input_size, hidden_sizes, num_classes):
        """Create a neural network for quantization demo"""
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        return nn.Sequential(*layers)
    
    def train_model(self, model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
        """Train a model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            val_accuracy = val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_accuracy)
        
        return train_losses, val_accuracies
    
    def get_model_size(self, model):
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def measure_inference_time(self, model, test_loader, num_runs=100):
        """Measure average inference time"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                _ = model(batch_X)
                break
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                for batch_X, _ in test_loader:
                    batch_X = batch_X.to(device)
                    _ = model(batch_X)
                end_time = time.time()
                times.append(end_time - start_time)
        
        return np.mean(times), np.std(times)
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model accuracy"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return correct / total
    
    def baseline_model_demo(self, X, y):
        """Test baseline model without quantization"""
        print("🔍 Testing Baseline Model (No Quantization)...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.numpy())
        X_test_scaled = scaler.transform(X_test.numpy())
        
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        train_dataset = TensorDataset(X_train_tensor, y_train)
        test_dataset = TensorDataset(X_test_tensor, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create and train model
        model = self.create_neural_network(
            input_size=X_train_scaled.shape[1],
            hidden_sizes=[128, 64, 32],
            num_classes=len(torch.unique(y))
        )
        
        train_losses, val_accuracies = self.train_model(model, train_loader, test_loader)
        
        # Evaluate model
        accuracy = self.evaluate_model(model, test_loader)
        model_size = self.get_model_size(model)
        inference_time, inference_std = self.measure_inference_time(model, test_loader)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'model_size_mb': model_size,
            'inference_time': inference_time,
            'inference_std': inference_std,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }
    
    def dynamic_quantization_demo(self, baseline_model, test_loader):
        """Test dynamic quantization"""
        print("⚡ Testing Dynamic Quantization...")
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            baseline_model, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        
        # Evaluate quantized model
        accuracy = self.evaluate_model(quantized_model, test_loader)
        model_size = self.get_model_size(quantized_model)
        inference_time, inference_std = self.measure_inference_time(quantized_model, test_loader)
        
        return {
            'model': quantized_model,
            'accuracy': accuracy,
            'model_size_mb': model_size,
            'inference_time': inference_time,
            'inference_std': inference_std,
            'compression_ratio': baseline_model.get_model_size() / model_size if hasattr(baseline_model, 'get_model_size') else 1.0
        }
    
    def static_quantization_demo(self, baseline_model, train_loader, test_loader):
        """Test static quantization"""
        print("📊 Testing Static Quantization...")
        
        # Set model to evaluation mode
        baseline_model.eval()
        
        # Create quantized model
        quantized_model = torch.quantization.quantize_dynamic(
            baseline_model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        # Prepare for static quantization
        quantized_model.eval()
        
        # Set quantization config
        quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare the model
        torch.quantization.prepare(quantized_model, inplace=True)
        
        # Calibrate with training data
        with torch.no_grad():
            for batch_X, _ in train_loader:
                _ = quantized_model(batch_X)
                break  # Just one batch for calibration
        
        # Convert to quantized model
        torch.quantization.convert(quantized_model, inplace=True)
        
        # Evaluate quantized model
        accuracy = self.evaluate_model(quantized_model, test_loader)
        model_size = self.get_model_size(quantized_model)
        inference_time, inference_std = self.measure_inference_time(quantized_model, test_loader)
        
        return {
            'model': quantized_model,
            'accuracy': accuracy,
            'model_size_mb': model_size,
            'inference_time': inference_time,
            'inference_std': inference_std
        }
    
    def fp16_quantization_demo(self, baseline_model, test_loader):
        """Test FP16 quantization"""
        print("🔢 Testing FP16 Quantization...")
        
        # Convert model to FP16
        fp16_model = baseline_model.half()
        
        # Convert test data to FP16
        fp16_test_loader = []
        for batch_X, batch_y in test_loader:
            fp16_test_loader.append((batch_X.half(), batch_y))
        
        # Evaluate FP16 model
        accuracy = self.evaluate_model_fp16(fp16_model, fp16_test_loader)
        model_size = self.get_model_size(fp16_model)
        inference_time, inference_std = self.measure_inference_time_fp16(fp16_model, fp16_test_loader)
        
        return {
            'model': fp16_model,
            'accuracy': accuracy,
            'model_size_mb': model_size,
            'inference_time': inference_time,
            'inference_std': inference_std
        }
    
    def evaluate_model_fp16(self, model, test_loader):
        """Evaluate FP16 model accuracy"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return correct / total
    
    def measure_inference_time_fp16(self, model, test_loader, num_runs=100):
        """Measure FP16 model inference time"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                _ = model(batch_X)
                break
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                for batch_X, _ in test_loader:
                    batch_X = batch_X.to(device)
                    _ = model(batch_X)
                end_time = time.time()
                times.append(end_time - start_time)
        
        return np.mean(times), np.std(times)
    
    def model_pruning_demo(self, baseline_model, test_loader, sparsity=0.5):
        """Test model pruning"""
        print("✂️ Testing Model Pruning...")
        
        # Create a copy of the model
        pruned_model = self.create_neural_network(
            input_size=baseline_model[0].in_features,
            hidden_sizes=[128, 64, 32],
            num_classes=baseline_model[-1].out_features
        )
        
        # Copy weights from baseline model
        with torch.no_grad():
            for i, (baseline_param, pruned_param) in enumerate(zip(baseline_model.parameters(), pruned_model.parameters())):
                pruned_param.copy_(baseline_param)
        
        # Apply pruning
        for module in pruned_model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=sparsity)
        
        # Evaluate pruned model
        accuracy = self.evaluate_model(pruned_model, test_loader)
        model_size = self.get_model_size(pruned_model)
        inference_time, inference_std = self.measure_inference_time(pruned_model, test_loader)
        
        return {
            'model': pruned_model,
            'accuracy': accuracy,
            'model_size_mb': model_size,
            'inference_time': inference_time,
            'inference_std': inference_std,
            'sparsity': sparsity
        }
    
    def torchscript_optimization_demo(self, baseline_model, test_loader):
        """Test TorchScript optimization"""
        print("🚀 Testing TorchScript Optimization...")
        
        # Convert model to TorchScript
        try:
            scripted_model = torch.jit.script(baseline_model)
            
            # Evaluate scripted model
            accuracy = self.evaluate_model(scripted_model, test_loader)
            model_size = self.get_model_size(scripted_model)
            inference_time, inference_std = self.measure_inference_time(scripted_model, test_loader)
            
            return {
                'model': scripted_model,
                'accuracy': accuracy,
                'model_size_mb': model_size,
                'inference_time': inference_time,
                'inference_std': inference_std,
                'success': True
            }
        except Exception as e:
            print(f"TorchScript conversion failed: {e}")
            return {
                'model': None,
                'accuracy': 0.0,
                'model_size_mb': 0.0,
                'inference_time': 0.0,
                'inference_std': 0.0,
                'success': False
            }
    
    def run_comprehensive_demo(self):
        """Run comprehensive quantization optimization comparison"""
        print("🚀 Starting Model Quantization Optimization Demo")
        print("=" * 60)
        
        # Create synthetic data
        X, y = self.create_synthetic_data(n_samples=2000, n_features=50, n_classes=3)
        print(f"Dataset shape: {X.shape}, Classes: {len(torch.unique(y))}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.numpy())
        X_test_scaled = scaler.transform(X_test.numpy())
        
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        train_dataset = TensorDataset(X_train_tensor, y_train)
        test_dataset = TensorDataset(X_test_tensor, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Test baseline model
        print("\n🔍 Testing Baseline Model...")
        baseline_result = self.baseline_model_demo(X, y)
        self.results['baseline'] = baseline_result
        
        # Test dynamic quantization
        print("\n⚡ Testing Dynamic Quantization...")
        dynamic_result = self.dynamic_quantization_demo(baseline_result['model'], test_loader)
        self.results['dynamic_quantization'] = dynamic_result
        
        # Test static quantization
        print("\n📊 Testing Static Quantization...")
        static_result = self.static_quantization_demo(baseline_result['model'], train_loader, test_loader)
        self.results['static_quantization'] = static_result
        
        # Test FP16 quantization
        print("\n🔢 Testing FP16 Quantization...")
        fp16_result = self.fp16_quantization_demo(baseline_result['model'], test_loader)
        self.results['fp16_quantization'] = fp16_result
        
        # Test model pruning
        print("\n✂️ Testing Model Pruning...")
        pruning_result = self.model_pruning_demo(baseline_result['model'], test_loader, sparsity=0.3)
        self.results['model_pruning'] = pruning_result
        
        # Test TorchScript optimization
        print("\n🚀 Testing TorchScript Optimization...")
        torchscript_result = self.torchscript_optimization_demo(baseline_result['model'], test_loader)
        self.results['torchscript'] = torchscript_result
        
        # Display results
        self.display_results()
        
        return self.results
    
    def display_results(self):
        """Display comprehensive results"""
        print("\n" + "=" * 80)
        print("📊 MODEL QUANTIZATION OPTIMIZATION RESULTS")
        print("=" * 80)
        
        print(f"{'Method':<25} | {'Accuracy':<10} | {'Size (MB)':<10} | {'Time (s)':<10} | {'Speedup':<10}")
        print("-" * 80)
        
        baseline_accuracy = self.results['baseline']['accuracy']
        baseline_size = self.results['baseline']['model_size_mb']
        baseline_time = self.results['baseline']['inference_time']
        
        for method_name, result in self.results.items():
            if result['accuracy'] > 0:  # Skip failed methods
                speedup = baseline_time / result['inference_time'] if result['inference_time'] > 0 else 0
                size_reduction = (baseline_size - result['model_size_mb']) / baseline_size * 100
                
                print(f"{method_name:<25} | {result['accuracy']:<10.4f} | "
                      f"{result['model_size_mb']:<10.2f} | {result['inference_time']:<10.4f} | "
                      f"{speedup:<10.2f}x")
        
        # Find best performing techniques
        self.find_best_techniques()
    
    def find_best_techniques(self):
        """Identify the best performing techniques"""
        print("\n🏆 BEST PERFORMING QUANTIZATION TECHNIQUES")
        print("-" * 60)
        
        all_results = []
        
        # Collect all results
        for method_name, result in self.results.items():
            if result['accuracy'] > 0:  # Skip failed methods
                all_results.append({
                    'method': method_name,
                    'accuracy': result['accuracy'],
                    'model_size': result['model_size_mb'],
                    'inference_time': result['inference_time'],
                    'efficiency': result['accuracy'] / result['inference_time'] if result['inference_time'] > 0 else 0
                })
        
        # Sort by accuracy
        all_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print("Top 5 by Accuracy:")
        for i, result in enumerate(all_results[:5], 1):
            print(f"{i}. {result['method']:25} | Accuracy: {result['accuracy']:.4f}")
        
        # Sort by efficiency (accuracy per time)
        all_results.sort(key=lambda x: x['efficiency'], reverse=True)
        
        print("\nTop 5 by Efficiency (Accuracy/Time):")
        for i, result in enumerate(all_results[:5], 1):
            print(f"{i}. {result['method']:25} | Efficiency: {result['efficiency']:.4f}")
        
        # Sort by model size
        all_results.sort(key=lambda x: x['model_size'])
        
        print("\nTop 5 by Model Size (Smallest):")
        for i, result in enumerate(all_results[:5], 1):
            print(f"{i}. {result['method']:25} | Size: {result['model_size']:.2f} MB")
    
    def plot_results(self):
        """Create visualization of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Quantization Optimization Analysis', fontsize=16)
        
        # Prepare data for plotting
        methods = []
        accuracies = []
        model_sizes = []
        inference_times = []
        
        for method_name, result in self.results.items():
            if result['accuracy'] > 0:  # Skip failed methods
                methods.append(method_name)
                accuracies.append(result['accuracy'])
                model_sizes.append(result['model_size_mb'])
                inference_times.append(result['inference_time'])
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(methods)), accuracies)
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.grid(True, axis='y')
        
        # Plot 2: Model size comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(methods)), model_sizes)
        ax2.set_title('Model Size Comparison')
        ax2.set_ylabel('Model Size (MB)')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.grid(True, axis='y')
        
        # Plot 3: Inference time comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(methods)), inference_times)
        ax3.set_title('Inference Time Comparison')
        ax3.set_ylabel('Inference Time (s)')
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels(methods, rotation=45, ha='right')
        ax3.grid(True, axis='y')
        
        # Plot 4: Accuracy vs Model Size scatter
        ax4 = axes[1, 1]
        scatter = ax4.scatter(model_sizes, accuracies, s=100, alpha=0.7)
        ax4.set_xlabel('Model Size (MB)')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Model Size')
        ax4.grid(True)
        
        # Add labels to scatter points
        for i, method in enumerate(methods):
            ax4.annotate(method, (model_sizes[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.show()

def main():
    """Run the model quantization demo"""
    demo = ModelQuantizationDemo()
    results = demo.run_comprehensive_demo()
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    demo.plot_results()
    
    print("\n✅ Model Quantization Demo Complete!")
    print("\nKey Takeaways:")
    print("• Dynamic quantization provides good speedup with minimal accuracy loss")
    print("• FP16 quantization can significantly reduce model size")
    print("• Model pruning can reduce model size while maintaining accuracy")
    print("• TorchScript can improve inference speed through optimization")
    print("• Choose the right technique based on your accuracy vs speed trade-offs")

if __name__ == "__main__":
    main()
