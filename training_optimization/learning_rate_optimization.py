"""
Learning Rate Optimization for Training Performance
==================================================

This demo shows how different learning rate scheduling strategies can significantly
improve model training performance, convergence speed, and final accuracy.

Key Techniques Covered:
- Fixed Learning Rate
- Step Decay
- Exponential Decay
- Cosine Annealing
- One Cycle Learning Rate
- Cyclical Learning Rate
- Warm Restarts
- Adaptive Learning Rate (Adam, RMSprop)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import warnings
warnings.filterwarnings('ignore')

class LearningRateOptimizationDemo:
    def __init__(self):
        self.results = {}
        self.training_histories = {}
        
    def create_synthetic_data(self, n_samples=2000, n_features=20, n_classes=3):
        """Create synthetic dataset for learning rate optimization demo"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        return X_tensor, y_tensor
    
    def create_neural_network(self, input_size, hidden_sizes, num_classes):
        """Create a simple neural network"""
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        return nn.Sequential(*layers)
    
    def train_model(self, model, train_loader, val_loader, optimizer, scheduler, 
                   num_epochs=50, device='cpu'):
        """Train model with given optimizer and scheduler"""
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        learning_rates = []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            # Record metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # Step scheduler
            if scheduler is not None:
                scheduler.step()
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'learning_rates': learning_rates
        }
    
    def fixed_learning_rate_demo(self, X, y, learning_rates=[0.001, 0.01, 0.1]):
        """Test different fixed learning rates"""
        print("🔧 Testing Fixed Learning Rates...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.numpy())
        X_val_scaled = scaler.transform(X_val.numpy())
        
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        
        train_dataset = TensorDataset(X_train_tensor, y_train)
        val_dataset = TensorDataset(X_val_tensor, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        results = {}
        
        for lr in learning_rates:
            print(f"  Testing LR = {lr}")
            
            # Create model
            model = self.create_neural_network(
                input_size=X_train_scaled.shape[1],
                hidden_sizes=[64, 32],
                num_classes=len(torch.unique(y))
            )
            
            # Create optimizer
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            
            # Train model
            start_time = time.time()
            history = self.train_model(model, train_loader, val_loader, optimizer, None)
            training_time = time.time() - start_time
            
            results[f'LR_{lr}'] = {
                'history': history,
                'training_time': training_time,
                'final_val_accuracy': history['val_accuracies'][-1],
                'best_val_accuracy': max(history['val_accuracies']),
                'convergence_epoch': self.find_convergence_epoch(history['val_accuracies'])
            }
        
        return results
    
    def step_decay_demo(self, X, y, base_lr=0.1, decay_steps=[20, 40], decay_rate=0.5):
        """Test step decay learning rate scheduling"""
        print("📉 Testing Step Decay Learning Rate...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.numpy())
        X_val_scaled = scaler.transform(X_val.numpy())
        
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        
        train_dataset = TensorDataset(X_train_tensor, y_train)
        val_dataset = TensorDataset(X_val_tensor, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = self.create_neural_network(
            input_size=X_train_scaled.shape[1],
            hidden_sizes=[64, 32],
            num_classes=len(torch.unique(y))
        )
        
        # Create optimizer and scheduler
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=decay_rate)
        
        # Train model
        start_time = time.time()
        history = self.train_model(model, train_loader, val_loader, optimizer, scheduler)
        training_time = time.time() - start_time
        
        return {
            'history': history,
            'training_time': training_time,
            'final_val_accuracy': history['val_accuracies'][-1],
            'best_val_accuracy': max(history['val_accuracies']),
            'convergence_epoch': self.find_convergence_epoch(history['val_accuracies'])
        }
    
    def exponential_decay_demo(self, X, y, base_lr=0.1, gamma=0.95):
        """Test exponential decay learning rate scheduling"""
        print("📉 Testing Exponential Decay Learning Rate...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.numpy())
        X_val_scaled = scaler.transform(X_val.numpy())
        
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        
        train_dataset = TensorDataset(X_train_tensor, y_train)
        val_dataset = TensorDataset(X_val_tensor, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = self.create_neural_network(
            input_size=X_train_scaled.shape[1],
            hidden_sizes=[64, 32],
            num_classes=len(torch.unique(y))
        )
        
        # Create optimizer and scheduler
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        
        # Train model
        start_time = time.time()
        history = self.train_model(model, train_loader, val_loader, optimizer, scheduler)
        training_time = time.time() - start_time
        
        return {
            'history': history,
            'training_time': training_time,
            'final_val_accuracy': history['val_accuracies'][-1],
            'best_val_accuracy': max(history['val_accuracies']),
            'convergence_epoch': self.find_convergence_epoch(history['val_accuracies'])
        }
    
    def cosine_annealing_demo(self, X, y, base_lr=0.1, T_max=50):
        """Test cosine annealing learning rate scheduling"""
        print("🌊 Testing Cosine Annealing Learning Rate...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.numpy())
        X_val_scaled = scaler.transform(X_val.numpy())
        
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        
        train_dataset = TensorDataset(X_train_tensor, y_train)
        val_dataset = TensorDataset(X_val_tensor, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = self.create_neural_network(
            input_size=X_train_scaled.shape[1],
            hidden_sizes=[64, 32],
            num_classes=len(torch.unique(y))
        )
        
        # Create optimizer and scheduler
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        
        # Train model
        start_time = time.time()
        history = self.train_model(model, train_loader, val_loader, optimizer, scheduler)
        training_time = time.time() - start_time
        
        return {
            'history': history,
            'training_time': training_time,
            'final_val_accuracy': history['val_accuracies'][-1],
            'best_val_accuracy': max(history['val_accuracies']),
            'convergence_epoch': self.find_convergence_epoch(history['val_accuracies'])
        }
    
    def one_cycle_demo(self, X, y, max_lr=0.1, total_steps=50):
        """Test one cycle learning rate policy"""
        print("🔄 Testing One Cycle Learning Rate...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.numpy())
        X_val_scaled = scaler.transform(X_val.numpy())
        
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        
        train_dataset = TensorDataset(X_train_tensor, y_train)
        val_dataset = TensorDataset(X_val_tensor, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = self.create_neural_network(
            input_size=X_train_scaled.shape[1],
            hidden_sizes=[64, 32],
            num_classes=len(torch.unique(y))
        )
        
        # Create optimizer and scheduler
        optimizer = optim.SGD(model.parameters(), lr=max_lr/10, momentum=0.9)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps
        )
        
        # Train model
        start_time = time.time()
        history = self.train_model(model, train_loader, val_loader, optimizer, scheduler)
        training_time = time.time() - start_time
        
        return {
            'history': history,
            'training_time': training_time,
            'final_val_accuracy': history['val_accuracies'][-1],
            'best_val_accuracy': max(history['val_accuracies']),
            'convergence_epoch': self.find_convergence_epoch(history['val_accuracies'])
        }
    
    def adaptive_optimizers_demo(self, X, y):
        """Test different adaptive optimizers"""
        print("🧠 Testing Adaptive Optimizers...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.numpy())
        X_val_scaled = scaler.transform(X_val.numpy())
        
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        
        train_dataset = TensorDataset(X_train_tensor, y_train)
        val_dataset = TensorDataset(X_val_tensor, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        optimizers_config = {
            'SGD': {'optimizer': optim.SGD, 'params': {'lr': 0.01, 'momentum': 0.9}},
            'Adam': {'optimizer': optim.Adam, 'params': {'lr': 0.001}},
            'RMSprop': {'optimizer': optim.RMSprop, 'params': {'lr': 0.01}},
            'Adagrad': {'optimizer': optim.Adagrad, 'params': {'lr': 0.01}},
            'AdamW': {'optimizer': optim.AdamW, 'params': {'lr': 0.001}}
        }
        
        results = {}
        
        for opt_name, opt_config in optimizers_config.items():
            print(f"  Testing {opt_name}")
            
            # Create model
            model = self.create_neural_network(
                input_size=X_train_scaled.shape[1],
                hidden_sizes=[64, 32],
                num_classes=len(torch.unique(y))
            )
            
            # Create optimizer
            optimizer = opt_config['optimizer'](model.parameters(), **opt_config['params'])
            
            # Train model
            start_time = time.time()
            history = self.train_model(model, train_loader, val_loader, optimizer, None)
            training_time = time.time() - start_time
            
            results[opt_name] = {
                'history': history,
                'training_time': training_time,
                'final_val_accuracy': history['val_accuracies'][-1],
                'best_val_accuracy': max(history['val_accuracies']),
                'convergence_epoch': self.find_convergence_epoch(history['val_accuracies'])
            }
        
        return results
    
    def find_convergence_epoch(self, val_accuracies, patience=5):
        """Find the epoch where validation accuracy converged"""
        best_acc = max(val_accuracies)
        best_epoch = val_accuracies.index(best_acc)
        
        # Check if accuracy improved significantly in the last 'patience' epochs
        for epoch in range(len(val_accuracies) - patience, len(val_accuracies)):
            if val_accuracies[epoch] >= best_acc - 0.01:  # Within 1% of best
                return epoch
        
        return best_epoch
    
    def run_comprehensive_demo(self):
        """Run comprehensive learning rate optimization comparison"""
        print("🚀 Starting Learning Rate Optimization Demo")
        print("=" * 60)
        
        # Create synthetic data
        X, y = self.create_synthetic_data(n_samples=2000, n_features=25, n_classes=3)
        print(f"Dataset shape: {X.shape}, Classes: {len(torch.unique(y))}")
        
        # Test different learning rate strategies
        print("\n🔧 Testing Fixed Learning Rates...")
        fixed_lr_results = self.fixed_learning_rate_demo(X, y)
        self.results['fixed_lr'] = fixed_lr_results
        
        print("\n📉 Testing Step Decay...")
        step_decay_result = self.step_decay_demo(X, y)
        self.results['step_decay'] = step_decay_result
        
        print("\n📉 Testing Exponential Decay...")
        exp_decay_result = self.exponential_decay_demo(X, y)
        self.results['exp_decay'] = exp_decay_result
        
        print("\n🌊 Testing Cosine Annealing...")
        cosine_result = self.cosine_annealing_demo(X, y)
        self.results['cosine_annealing'] = cosine_result
        
        print("\n🔄 Testing One Cycle...")
        one_cycle_result = self.one_cycle_demo(X, y)
        self.results['one_cycle'] = one_cycle_result
        
        print("\n🧠 Testing Adaptive Optimizers...")
        adaptive_results = self.adaptive_optimizers_demo(X, y)
        self.results['adaptive_optimizers'] = adaptive_results
        
        # Display results
        self.display_results()
        
        return self.results
    
    def display_results(self):
        """Display comprehensive results"""
        print("\n" + "=" * 80)
        print("📊 LEARNING RATE OPTIMIZATION RESULTS")
        print("=" * 80)
        
        # Fixed learning rates
        print("\n🔧 FIXED LEARNING RATES")
        print("-" * 50)
        for lr_name, result in self.results['fixed_lr'].items():
            print(f"{lr_name:15} | Final Acc: {result['final_val_accuracy']:.4f} | "
                  f"Best Acc: {result['best_val_accuracy']:.4f} | "
                  f"Time: {result['training_time']:.2f}s | "
                  f"Convergence: Epoch {result['convergence_epoch']}")
        
        # Learning rate schedules
        print("\n📉 LEARNING RATE SCHEDULES")
        print("-" * 50)
        schedule_results = {
            'Step Decay': self.results['step_decay'],
            'Exponential Decay': self.results['exp_decay'],
            'Cosine Annealing': self.results['cosine_annealing'],
            'One Cycle': self.results['one_cycle']
        }
        
        for schedule_name, result in schedule_results.items():
            print(f"{schedule_name:20} | Final Acc: {result['final_val_accuracy']:.4f} | "
                  f"Best Acc: {result['best_val_accuracy']:.4f} | "
                  f"Time: {result['training_time']:.2f}s | "
                  f"Convergence: Epoch {result['convergence_epoch']}")
        
        # Adaptive optimizers
        print("\n🧠 ADAPTIVE OPTIMIZERS")
        print("-" * 50)
        for opt_name, result in self.results['adaptive_optimizers'].items():
            print(f"{opt_name:15} | Final Acc: {result['final_val_accuracy']:.4f} | "
                  f"Best Acc: {result['best_val_accuracy']:.4f} | "
                  f"Time: {result['training_time']:.2f}s | "
                  f"Convergence: Epoch {result['convergence_epoch']}")
        
        # Find best performing techniques
        self.find_best_techniques()
    
    def find_best_techniques(self):
        """Identify the best performing techniques"""
        print("\n🏆 BEST PERFORMING LEARNING RATE TECHNIQUES")
        print("-" * 60)
        
        all_results = []
        
        # Collect all results
        for category, results in self.results.items():
            if category == 'fixed_lr':
                for technique, metrics in results.items():
                    all_results.append({
                        'category': category,
                        'technique': technique,
                        'final_accuracy': metrics['final_val_accuracy'],
                        'best_accuracy': metrics['best_val_accuracy'],
                        'training_time': metrics['training_time'],
                        'convergence_epoch': metrics['convergence_epoch']
                    })
            elif category == 'adaptive_optimizers':
                for technique, metrics in results.items():
                    all_results.append({
                        'category': category,
                        'technique': technique,
                        'final_accuracy': metrics['final_val_accuracy'],
                        'best_accuracy': metrics['best_val_accuracy'],
                        'training_time': metrics['training_time'],
                        'convergence_epoch': metrics['convergence_epoch']
                    })
            else:
                all_results.append({
                    'category': category,
                    'technique': category,
                    'final_accuracy': results['final_val_accuracy'],
                    'best_accuracy': results['best_val_accuracy'],
                    'training_time': results['training_time'],
                    'convergence_epoch': results['convergence_epoch']
                })
        
        # Sort by best accuracy
        all_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
        
        print("Top 10 by Best Accuracy:")
        for i, result in enumerate(all_results[:10], 1):
            print(f"{i:2}. {result['technique']:20} | "
                  f"Best Acc: {result['best_accuracy']:.4f} | "
                  f"Final Acc: {result['final_accuracy']:.4f} | "
                  f"Time: {result['training_time']:.2f}s")
        
        # Sort by convergence speed
        all_results.sort(key=lambda x: x['convergence_epoch'])
        
        print("\nTop 10 by Convergence Speed:")
        for i, result in enumerate(all_results[:10], 1):
            print(f"{i:2}. {result['technique']:20} | "
                  f"Convergence: Epoch {result['convergence_epoch']} | "
                  f"Best Acc: {result['best_accuracy']:.4f} | "
                  f"Time: {result['training_time']:.2f}s")
    
    def plot_results(self):
        """Create visualization of results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Learning Rate Optimization Analysis', fontsize=16)
        
        # Plot 1: Training curves for fixed learning rates
        ax1 = axes[0, 0]
        for lr_name, result in self.results['fixed_lr'].items():
            ax1.plot(result['history']['val_accuracies'], label=lr_name)
        ax1.set_title('Fixed Learning Rates - Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Validation Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Learning rate schedules comparison
        ax2 = axes[0, 1]
        schedule_results = {
            'Step Decay': self.results['step_decay'],
            'Exponential Decay': self.results['exp_decay'],
            'Cosine Annealing': self.results['cosine_annealing'],
            'One Cycle': self.results['one_cycle']
        }
        
        for schedule_name, result in schedule_results.items():
            ax2.plot(result['history']['val_accuracies'], label=schedule_name)
        ax2.set_title('Learning Rate Schedules - Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Adaptive optimizers comparison
        ax3 = axes[0, 2]
        for opt_name, result in self.results['adaptive_optimizers'].items():
            ax3.plot(result['history']['val_accuracies'], label=opt_name)
        ax3.set_title('Adaptive Optimizers - Validation Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation Accuracy')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Learning rate curves
        ax4 = axes[1, 0]
        for lr_name, result in self.results['fixed_lr'].items():
            ax4.plot(result['history']['learning_rates'], label=lr_name)
        ax4.set_title('Learning Rate Curves (Fixed)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        ax4.grid(True)
        
        # Plot 5: Schedule learning rate curves
        ax5 = axes[1, 1]
        for schedule_name, result in schedule_results.items():
            ax5.plot(result['history']['learning_rates'], label=schedule_name)
        ax5.set_title('Learning Rate Curves (Schedules)')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Learning Rate')
        ax5.legend()
        ax5.grid(True)
        
        # Plot 6: Best accuracy comparison
        ax6 = axes[1, 2]
        all_techniques = []
        all_accuracies = []
        
        # Collect all techniques and their best accuracies
        for category, results in self.results.items():
            if category == 'fixed_lr':
                for technique, metrics in results.items():
                    all_techniques.append(technique)
                    all_accuracies.append(metrics['best_val_accuracy'])
            elif category == 'adaptive_optimizers':
                for technique, metrics in results.items():
                    all_techniques.append(technique)
                    all_accuracies.append(metrics['best_val_accuracy'])
            else:
                all_techniques.append(category)
                all_accuracies.append(results['best_val_accuracy'])
        
        bars = ax6.bar(range(len(all_techniques)), all_accuracies)
        ax6.set_title('Best Accuracy Comparison')
        ax6.set_ylabel('Best Validation Accuracy')
        ax6.set_xticks(range(len(all_techniques)))
        ax6.set_xticklabels(all_techniques, rotation=45, ha='right')
        ax6.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()

def main():
    """Run the learning rate optimization demo"""
    demo = LearningRateOptimizationDemo()
    results = demo.run_comprehensive_demo()
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    demo.plot_results()
    
    print("\n✅ Learning Rate Optimization Demo Complete!")
    print("\nKey Takeaways:")
    print("• Learning rate scheduling can significantly improve convergence speed")
    print("• One Cycle and Cosine Annealing often work well for deep learning")
    print("• Adaptive optimizers like Adam can reduce the need for manual tuning")
    print("• The optimal learning rate strategy depends on your specific problem")
    print("• Consider both final accuracy and convergence speed when choosing a strategy")

if __name__ == "__main__":
    main()
