"""
Simple ML Model Performance Optimization Demo
============================================

This is a simplified version of the optimization playground that focuses on
the core concepts without requiring all the advanced dependencies.

Perfect for getting started quickly!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_data(n_samples=1000, n_features=20):
    """Create synthetic dataset for demonstration"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y

def test_feature_scaling(X, y):
    """Test different feature scaling techniques"""
    print("🔧 Testing Feature Scaling Techniques")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scalers = {
        'No Scaling': None,
        'Standard Scaler': StandardScaler(),
        'MinMax Scaler': MinMaxScaler()
    }
    
    results = {}
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    for name, scaler in scalers.items():
        if scaler is None:
            X_train_scaled = X_train
            X_test_scaled = X_test
        else:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time
        
        results[name] = {
            'accuracy': accuracy,
            'training_time': training_time
        }
        
        print(f"{name:20} | Accuracy: {accuracy:.4f} | Time: {training_time:.4f}s")
    
    return results

def test_feature_selection(X, y):
    """Test feature selection techniques"""
    print("\n🎯 Testing Feature Selection Techniques")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the data first
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different numbers of features
    k_values = [5, 10, 15, 20]
    results = {}
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    
    for k in k_values:
        if k <= X_train_scaled.shape[1]:
            selector = SelectKBest(f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            start_time = time.time()
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            training_time = time.time() - start_time
            
            results[f'Top {k} Features'] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'n_features': k
            }
            
            print(f"Top {k:2} Features     | Accuracy: {accuracy:.4f} | Time: {training_time:.4f}s | Features: {k}")
    
    return results

def test_different_models(X, y):
    """Test different machine learning models"""
    print("\n🤖 Testing Different ML Models")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time
        
        results[name] = {
            'accuracy': accuracy,
            'training_time': training_time
        }
        
        print(f"{name:20} | Accuracy: {accuracy:.4f} | Time: {training_time:.4f}s")
    
    return results

def test_hyperparameter_tuning(X, y):
    """Test simple hyperparameter tuning"""
    print("\n⚙️ Testing Hyperparameter Tuning")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different hyperparameters for Random Forest
    n_estimators_list = [10, 50, 100, 200]
    max_depth_list = [5, 10, 15, None]
    
    results = {}
    best_accuracy = 0
    best_params = None
    
    for n_est in n_estimators_list:
        for max_d in max_depth_list:
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                random_state=42
            )
            
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            training_time = time.time() - start_time
            
            param_name = f"RF(n_est={n_est}, max_d={max_d})"
            results[param_name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'n_estimators': n_est,
                'max_depth': max_d
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = param_name
            
            print(f"{param_name:25} | Accuracy: {accuracy:.4f} | Time: {training_time:.4f}s")
    
    print(f"\n🏆 Best Configuration: {best_params} with accuracy {best_accuracy:.4f}")
    return results

def create_visualizations(scaling_results, selection_results, model_results, tuning_results):
    """Create visualizations of the results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ML Model Performance Optimization Results', fontsize=16)
    
    # Plot 1: Feature Scaling Results
    ax1 = axes[0, 0]
    methods = list(scaling_results.keys())
    accuracies = [scaling_results[method]['accuracy'] for method in methods]
    ax1.bar(methods, accuracies)
    ax1.set_title('Feature Scaling Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Feature Selection Results
    ax2 = axes[0, 1]
    methods = list(selection_results.keys())
    accuracies = [selection_results[method]['accuracy'] for method in methods]
    features = [selection_results[method]['n_features'] for method in methods]
    ax2.scatter(features, accuracies, s=100)
    ax2.set_title('Feature Selection: Accuracy vs Number of Features')
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('Accuracy')
    
    # Plot 3: Model Comparison
    ax3 = axes[1, 0]
    methods = list(model_results.keys())
    accuracies = [model_results[method]['accuracy'] for method in methods]
    times = [model_results[method]['training_time'] for method in methods]
    ax3.scatter(times, accuracies, s=100)
    for i, method in enumerate(methods):
        ax3.annotate(method, (times[i], accuracies[i]), xytext=(5, 5), textcoords='offset points')
    ax3.set_title('Model Comparison: Accuracy vs Training Time')
    ax3.set_xlabel('Training Time (s)')
    ax3.set_ylabel('Accuracy')
    
    # Plot 4: Hyperparameter Tuning Results
    ax4 = axes[1, 1]
    methods = list(tuning_results.keys())
    accuracies = [tuning_results[method]['accuracy'] for method in methods]
    n_estimators = [tuning_results[method]['n_estimators'] for method in methods]
    colors = [tuning_results[method]['max_depth'] if tuning_results[method]['max_depth'] else 20 for method in methods]
    scatter = ax4.scatter(n_estimators, accuracies, c=colors, cmap='viridis', s=100)
    ax4.set_title('Hyperparameter Tuning: Accuracy vs N_Estimators')
    ax4.set_xlabel('N_Estimators')
    ax4.set_ylabel('Accuracy')
    plt.colorbar(scatter, ax=ax4, label='Max Depth')
    
    plt.tight_layout()
    plt.show()

def print_summary(scaling_results, selection_results, model_results, tuning_results):
    """Print a summary of all results"""
    print("\n" + "="*80)
    print("📊 OPTIMIZATION SUMMARY")
    print("="*80)
    
    # Find best results
    all_results = []
    all_results.extend([(name, 'scaling', data) for name, data in scaling_results.items()])
    all_results.extend([(name, 'selection', data) for name, data in selection_results.items()])
    all_results.extend([(name, 'models', data) for name, data in model_results.items()])
    all_results.extend([(name, 'tuning', data) for name, data in tuning_results.items()])
    
    # Sort by accuracy
    all_results.sort(key=lambda x: x[2]['accuracy'], reverse=True)
    
    print("\n🏆 TOP 5 PERFORMING CONFIGURATIONS:")
    print("-" * 50)
    for i, (name, category, data) in enumerate(all_results[:5], 1):
        print(f"{i}. {name:25} ({category:10}) | Accuracy: {data['accuracy']:.4f} | Time: {data['training_time']:.4f}s")
    
    print("\n💡 KEY INSIGHTS:")
    print("-" * 20)
    print("• Feature scaling can significantly improve model performance")
    print("• Feature selection can reduce training time while maintaining accuracy")
    print("• Different models have different strengths and weaknesses")
    print("• Hyperparameter tuning can find the optimal configuration")
    print("• The best approach depends on your specific dataset and requirements")

def main():
    """Run the simple optimization demo"""
    print("🚀 ML Model Performance Optimization - Simple Demo")
    print("=" * 60)
    
    # Create synthetic data
    print("📊 Creating synthetic dataset...")
    X, y = create_synthetic_data(n_samples=1000, n_features=20)
    print(f"Dataset shape: {X.shape}, Classes: {len(np.unique(y))}")
    
    # Run different optimization tests
    scaling_results = test_feature_scaling(X, y)
    selection_results = test_feature_selection(X, y)
    model_results = test_different_models(X, y)
    tuning_results = test_hyperparameter_tuning(X, y)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_visualizations(scaling_results, selection_results, model_results, tuning_results)
    
    # Print summary
    print_summary(scaling_results, selection_results, model_results, tuning_results)
    
    print("\n✅ Demo Complete!")
    print("\nNext Steps:")
    print("• Try running individual demos in the other directories")
    print("• Experiment with your own datasets")
    print("• Explore more advanced optimization techniques")
    print("• Check out the full playground demos for deeper insights")

if __name__ == "__main__":
    main()

