"""
Feature Engineering for Performance Optimization
===============================================

This demo shows how different feature engineering techniques can significantly
improve model performance. We'll work with a synthetic dataset and compare
various feature engineering approaches.

Key Techniques Covered:
- Feature scaling and normalization
- Feature selection
- Polynomial features
- Feature interaction
- Dimensionality reduction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineeringDemo:
    def __init__(self):
        self.results = {}
        
    def create_synthetic_data(self, n_samples=1000, n_features=20, noise=0.1):
        """Create synthetic dataset for demonstration"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Add some noise to make it more realistic
        X += np.random.normal(0, noise, X.shape)
        
        return X, y
    
    def baseline_performance(self, X, y):
        """Test baseline performance without feature engineering"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Test with different models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        baseline_results = {}
        for name, model in models.items():
            start_time = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            training_time = time.time() - start_time
            
            baseline_results[name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'n_features': X.shape[1]
            }
        
        return baseline_results
    
    def scaling_comparison(self, X, y):
        """Compare different scaling techniques"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scalers = {
            'No Scaling': None,
            'Standard Scaler': StandardScaler(),
            'MinMax Scaler': MinMaxScaler(),
            'Robust Scaler': RobustScaler()
        }
        
        scaling_results = {}
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
            
            scaling_results[name] = {
                'accuracy': accuracy,
                'training_time': training_time
            }
        
        return scaling_results
    
    def feature_selection_demo(self, X, y):
        """Demonstrate feature selection techniques"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the data first
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Different feature selection methods
        selection_methods = {
            'All Features': None,
            'SelectKBest (f_classif)': SelectKBest(f_classif, k=10),
            'SelectKBest (mutual_info)': SelectKBest(mutual_info_classif, k=10),
            'PCA (10 components)': PCA(n_components=10)
        }
        
        selection_results = {}
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        
        for name, selector in selection_methods.items():
            if selector is None:
                X_train_selected = X_train_scaled
                X_test_selected = X_test_scaled
                n_features = X_train_scaled.shape[1]
            else:
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                X_test_selected = selector.transform(X_test_scaled)
                n_features = X_train_selected.shape[1]
            
            start_time = time.time()
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            training_time = time.time() - start_time
            
            selection_results[name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'n_features': n_features
            }
        
        return selection_results
    
    def polynomial_features_demo(self, X, y):
        """Demonstrate polynomial feature generation"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the data first
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Different polynomial degrees
        degrees = [1, 2, 3]
        poly_results = {}
        
        for degree in degrees:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(X_train_scaled)
            X_test_poly = poly.transform(X_test_scaled)
            
            # Use a simpler model for higher degree polynomials
            if degree > 2:
                model = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
            else:
                model = LogisticRegression(random_state=42, max_iter=1000)
            
            start_time = time.time()
            model.fit(X_train_poly, y_train)
            y_pred = model.predict(X_test_poly)
            accuracy = accuracy_score(y_test, y_pred)
            training_time = time.time() - start_time
            
            poly_results[f'Degree {degree}'] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'n_features': X_train_poly.shape[1]
            }
        
        return poly_results
    
    def run_comprehensive_demo(self):
        """Run all feature engineering experiments"""
        print("🚀 Starting Feature Engineering Performance Demo")
        print("=" * 60)
        
        # Create synthetic data
        print("📊 Creating synthetic dataset...")
        X, y = self.create_synthetic_data(n_samples=2000, n_features=30)
        print(f"Dataset shape: {X.shape}, Classes: {len(np.unique(y))}")
        
        # Baseline performance
        print("\n🔍 Testing baseline performance...")
        baseline_results = self.baseline_performance(X, y)
        self.results['baseline'] = baseline_results
        
        # Scaling comparison
        print("\n⚖️ Comparing scaling techniques...")
        scaling_results = self.scaling_comparison(X, y)
        self.results['scaling'] = scaling_results
        
        # Feature selection
        print("\n🎯 Testing feature selection...")
        selection_results = self.feature_selection_demo(X, y)
        self.results['selection'] = selection_results
        
        # Polynomial features
        print("\n📈 Testing polynomial features...")
        poly_results = self.polynomial_features_demo(X, y)
        self.results['polynomial'] = poly_results
        
        # Display results
        self.display_results()
        
        return self.results
    
    def display_results(self):
        """Display comprehensive results"""
        print("\n" + "=" * 80)
        print("📊 FEATURE ENGINEERING PERFORMANCE RESULTS")
        print("=" * 80)
        
        # Baseline results
        print("\n🔍 BASELINE PERFORMANCE (No Feature Engineering)")
        print("-" * 50)
        for model_name, metrics in self.results['baseline'].items():
            print(f"{model_name:20} | Accuracy: {metrics['accuracy']:.4f} | "
                  f"Time: {metrics['training_time']:.4f}s | Features: {metrics['n_features']}")
        
        # Scaling results
        print("\n⚖️ SCALING COMPARISON")
        print("-" * 50)
        for scaler_name, metrics in self.results['scaling'].items():
            print(f"{scaler_name:20} | Accuracy: {metrics['accuracy']:.4f} | "
                  f"Time: {metrics['training_time']:.4f}s")
        
        # Feature selection results
        print("\n🎯 FEATURE SELECTION COMPARISON")
        print("-" * 50)
        for method_name, metrics in self.results['selection'].items():
            print(f"{method_name:25} | Accuracy: {metrics['accuracy']:.4f} | "
                  f"Time: {metrics['training_time']:.4f}s | Features: {metrics['n_features']}")
        
        # Polynomial features results
        print("\n📈 POLYNOMIAL FEATURES COMPARISON")
        print("-" * 50)
        for degree_name, metrics in self.results['polynomial'].items():
            print(f"{degree_name:20} | Accuracy: {metrics['accuracy']:.4f} | "
                  f"Time: {metrics['training_time']:.4f}s | Features: {metrics['n_features']}")
        
        # Find best performing techniques
        self.find_best_techniques()
    
    def find_best_techniques(self):
        """Identify the best performing techniques"""
        print("\n🏆 BEST PERFORMING TECHNIQUES")
        print("-" * 50)
        
        all_results = []
        
        # Collect all results
        for category, results in self.results.items():
            for technique, metrics in results.items():
                all_results.append({
                    'category': category,
                    'technique': technique,
                    'accuracy': metrics['accuracy'],
                    'training_time': metrics['training_time'],
                    'n_features': metrics.get('n_features', 'N/A')
                })
        
        # Sort by accuracy
        all_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print("Top 5 by Accuracy:")
        for i, result in enumerate(all_results[:5], 1):
            print(f"{i}. {result['technique']:25} | "
                  f"Accuracy: {result['accuracy']:.4f} | "
                  f"Time: {result['training_time']:.4f}s")
        
        # Sort by speed (lowest training time)
        all_results.sort(key=lambda x: x['training_time'])
        
        print("\nTop 5 by Speed:")
        for i, result in enumerate(all_results[:5], 1):
            print(f"{i}. {result['technique']:25} | "
                  f"Accuracy: {result['accuracy']:.4f} | "
                  f"Time: {result['training_time']:.4f}s")
    
    def plot_results(self):
        """Create visualization of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Engineering Performance Analysis', fontsize=16)
        
        # Accuracy comparison
        ax1 = axes[0, 0]
        categories = list(self.results.keys())
        techniques = []
        accuracies = []
        
        for category, results in self.results.items():
            for technique, metrics in results.items():
                techniques.append(f"{category}\n{technique}")
                accuracies.append(metrics['accuracy'])
        
        ax1.bar(range(len(techniques)), accuracies)
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(techniques)))
        ax1.set_xticklabels(techniques, rotation=45, ha='right')
        
        # Training time comparison
        ax2 = axes[0, 1]
        times = [metrics['training_time'] for results in self.results.values() 
                for metrics in results.values()]
        ax2.bar(range(len(techniques)), times)
        ax2.set_title('Training Time Comparison')
        ax2.set_ylabel('Training Time (s)')
        ax2.set_xticks(range(len(techniques)))
        ax2.set_xticklabels(techniques, rotation=45, ha='right')
        
        # Accuracy vs Time scatter
        ax3 = axes[1, 0]
        ax3.scatter(times, accuracies, alpha=0.7)
        ax3.set_xlabel('Training Time (s)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy vs Training Time')
        
        # Feature count comparison (where available)
        ax4 = axes[1, 1]
        feature_counts = []
        feature_labels = []
        
        for category, results in self.results.items():
            for technique, metrics in results.items():
                if 'n_features' in metrics and metrics['n_features'] != 'N/A':
                    feature_counts.append(metrics['n_features'])
                    feature_labels.append(f"{category}\n{technique}")
        
        if feature_counts:
            ax4.bar(range(len(feature_labels)), feature_counts)
            ax4.set_title('Feature Count Comparison')
            ax4.set_ylabel('Number of Features')
            ax4.set_xticks(range(len(feature_labels)))
            ax4.set_xticklabels(feature_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()

def main():
    """Run the feature engineering demo"""
    demo = FeatureEngineeringDemo()
    results = demo.run_comprehensive_demo()
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    demo.plot_results()
    
    print("\n✅ Feature Engineering Demo Complete!")
    print("\nKey Takeaways:")
    print("• Feature scaling can significantly improve model performance")
    print("• Feature selection can reduce training time while maintaining accuracy")
    print("• Polynomial features can capture non-linear relationships")
    print("• The best technique depends on your specific dataset and requirements")

if __name__ == "__main__":
    main()
