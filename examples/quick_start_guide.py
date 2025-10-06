"""
Quick Start Guide for ML Model Performance Optimization
======================================================

This guide demonstrates how to use the playground to optimize ML model performance
with practical examples that you can run immediately.

What you'll learn:
1. How to set up and run optimization demos
2. How to interpret results and choose the best techniques
3. How to combine multiple optimization strategies
4. How to benchmark your optimizations
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# Add the playground modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_optimization.feature_engineering_demo import FeatureEngineeringDemo
from model_optimization.hyperparameter_tuning_demo import HyperparameterTuningDemo
from training_optimization.learning_rate_optimization import LearningRateOptimizationDemo
from inference_optimization.model_quantization_demo import ModelQuantizationDemo
from benchmarking.performance_profiler import PerformanceProfiler

class QuickStartGuide:
    def __init__(self):
        self.results = {}
        print("🚀 Welcome to the ML Model Performance Optimization Playground!")
        print("=" * 70)
    
    def step_1_data_optimization(self):
        """Step 1: Optimize your data preprocessing"""
        print("\n📊 STEP 1: DATA OPTIMIZATION")
        print("-" * 40)
        print("Let's start by optimizing how we preprocess and engineer features...")
        
        # Run feature engineering demo
        demo = FeatureEngineeringDemo()
        results = demo.run_comprehensive_demo()
        
        self.results['data_optimization'] = results
        
        print("\n✅ Data optimization complete!")
        print("Key insights:")
        print("• Feature scaling can improve model performance")
        print("• Feature selection can reduce training time")
        print("• The right preprocessing depends on your data")
        
        return results
    
    def step_2_model_optimization(self):
        """Step 2: Optimize your model architecture and hyperparameters"""
        print("\n🔧 STEP 2: MODEL OPTIMIZATION")
        print("-" * 40)
        print("Now let's find the best model architecture and hyperparameters...")
        
        # Run hyperparameter tuning demo
        demo = HyperparameterTuningDemo()
        results = demo.run_comprehensive_demo()
        
        self.results['model_optimization'] = results
        
        print("\n✅ Model optimization complete!")
        print("Key insights:")
        print("• Bayesian optimization often finds better solutions faster")
        print("• Different models benefit from different tuning strategies")
        print("• Consider both accuracy and training time when choosing")
        
        return results
    
    def step_3_training_optimization(self):
        """Step 3: Optimize your training process"""
        print("\n🏋️ STEP 3: TRAINING OPTIMIZATION")
        print("-" * 40)
        print("Let's optimize how we train our models...")
        
        # Run learning rate optimization demo
        demo = LearningRateOptimizationDemo()
        results = demo.run_comprehensive_demo()
        
        self.results['training_optimization'] = results
        
        print("\n✅ Training optimization complete!")
        print("Key insights:")
        print("• Learning rate scheduling can improve convergence")
        print("• Adaptive optimizers can reduce manual tuning")
        print("• One Cycle and Cosine Annealing often work well")
        
        return results
    
    def step_4_inference_optimization(self):
        """Step 4: Optimize your model for inference"""
        print("\n⚡ STEP 4: INFERENCE OPTIMIZATION")
        print("-" * 40)
        print("Finally, let's optimize our model for fast inference...")
        
        # Run model quantization demo
        demo = ModelQuantizationDemo()
        results = demo.run_comprehensive_demo()
        
        self.results['inference_optimization'] = results
        
        print("\n✅ Inference optimization complete!")
        print("Key insights:")
        print("• Quantization can significantly reduce model size")
        print("• Dynamic quantization provides good speedup with minimal accuracy loss")
        print("• Choose techniques based on your accuracy vs speed trade-offs")
        
        return results
    
    def step_5_benchmarking(self):
        """Step 5: Benchmark your optimizations"""
        print("\n📈 STEP 5: BENCHMARKING")
        print("-" * 40)
        print("Let's benchmark our optimizations to measure improvements...")
        
        # Create a simple example to benchmark
        def example_training():
            """Example training function to benchmark"""
            X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            return accuracy
        
        # Set up profiler
        profiler = PerformanceProfiler()
        
        # Benchmark baseline
        profiler.start_session("baseline", "Baseline model performance")
        baseline_result = profiler.measure_function(example_training)
        profiler.set_baseline("baseline", baseline_result['metrics'])
        profiler.end_session()
        
        # Benchmark optimized version
        def optimized_training():
            """Optimized training function"""
            X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Feature selection
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(f_classif, k=15)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Optimized hyperparameters
            model = RandomForestClassifier(
                n_estimators=50,  # Reduced for speed
                max_depth=10,     # Optimized
                min_samples_split=5,  # Optimized
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            return accuracy
        
        profiler.start_session("optimized", "Optimized model performance")
        optimized_result = profiler.measure_function(optimized_training)
        profiler.end_session()
        
        # Compare results
        comparison = profiler.compare_with_baseline(optimized_result['metrics'])
        
        print("\n📊 BENCHMARKING RESULTS:")
        print("-" * 30)
        print(f"Baseline accuracy: {baseline_result['result']:.4f}")
        print(f"Optimized accuracy: {optimized_result['result']:.4f}")
        print(f"Baseline time: {baseline_result['metrics']['execution_time']:.4f}s")
        print(f"Optimized time: {optimized_result['metrics']['execution_time']:.4f}s")
        
        if 'execution_time_improvement_percent' in comparison:
            improvement = comparison['execution_time_improvement_percent']
            print(f"Time improvement: {improvement:.2f}%")
        
        # Generate report
        report = profiler.generate_report()
        print(f"\n📋 DETAILED REPORT:\n{report}")
        
        self.results['benchmarking'] = {
            'baseline': baseline_result,
            'optimized': optimized_result,
            'comparison': comparison
        }
        
        return self.results['benchmarking']
    
    def step_6_combine_optimizations(self):
        """Step 6: Combine multiple optimization techniques"""
        print("\n🔗 STEP 6: COMBINING OPTIMIZATIONS")
        print("-" * 40)
        print("Let's combine multiple optimization techniques for maximum impact...")
        
        # Create a comprehensive optimization pipeline
        def comprehensive_optimization_pipeline():
            """Comprehensive optimization pipeline combining multiple techniques"""
            # 1. Data optimization
            X, y = make_classification(n_samples=2000, n_features=30, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Feature selection
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(f_classif, k=20)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # 2. Model optimization (optimized hyperparameters)
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            )
            
            # 3. Training optimization (early stopping simulation)
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # 4. Inference optimization (model size reduction)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                'accuracy': accuracy,
                'training_time': training_time,
                'n_features': X_train_selected.shape[1],
                'n_estimators': model.n_estimators
            }
        
        # Benchmark the comprehensive pipeline
        profiler = PerformanceProfiler()
        profiler.start_session("comprehensive", "Comprehensive optimization pipeline")
        result = profiler.measure_function(comprehensive_optimization_pipeline)
        profiler.end_session()
        
        print("\n🎯 COMPREHENSIVE OPTIMIZATION RESULTS:")
        print("-" * 45)
        print(f"Accuracy: {result['result']['accuracy']:.4f}")
        print(f"Training time: {result['result']['training_time']:.4f}s")
        print(f"Features used: {result['result']['n_features']}")
        print(f"Model complexity: {result['result']['n_estimators']} estimators")
        print(f"Memory usage: {result['metrics']['memory_delta_mb']:.2f}MB")
        print(f"CPU usage: {result['metrics']['cpu_usage_percent']:.1f}%")
        
        self.results['comprehensive'] = result
        
        return result
    
    def create_optimization_roadmap(self):
        """Create a personalized optimization roadmap"""
        print("\n🗺️ YOUR OPTIMIZATION ROADMAP")
        print("=" * 50)
        
        roadmap = [
            "1. 📊 DATA OPTIMIZATION",
            "   • Start with feature scaling and normalization",
            "   • Apply feature selection to reduce dimensionality",
            "   • Consider data augmentation for small datasets",
            "   • Handle missing values and outliers",
            "",
            "2. 🔧 MODEL OPTIMIZATION",
            "   • Use hyperparameter tuning (Bayesian optimization recommended)",
            "   • Try different model architectures",
            "   • Consider ensemble methods",
            "   • Use cross-validation for robust evaluation",
            "",
            "3. 🏋️ TRAINING OPTIMIZATION",
            "   • Implement learning rate scheduling",
            "   • Use adaptive optimizers (Adam, RMSprop)",
            "   • Consider mixed precision training",
            "   • Implement early stopping",
            "",
            "4. ⚡ INFERENCE OPTIMIZATION",
            "   • Apply model quantization (start with dynamic)",
            "   • Consider model pruning for size reduction",
            "   • Use ONNX for cross-platform deployment",
            "   • Implement batch processing for efficiency",
            "",
            "5. 📈 MONITORING & BENCHMARKING",
            "   • Set up performance monitoring",
            "   • Create automated benchmarks",
            "   • Track performance regressions",
            "   • Document optimization decisions",
            "",
            "6. 🔄 ITERATIVE IMPROVEMENT",
            "   • Continuously monitor performance",
            "   • A/B test different optimizations",
            "   • Update models with new data",
            "   • Stay updated with new techniques"
        ]
        
        for item in roadmap:
            print(item)
        
        print("\n💡 PRO TIPS:")
        print("• Start with the biggest impact, lowest effort optimizations")
        print("• Always measure before and after optimization")
        print("• Consider your specific use case and constraints")
        print("• Don't optimize prematurely - profile first!")
        print("• Document your optimization decisions")
    
    def run_complete_guide(self):
        """Run the complete quick start guide"""
        print("🎯 Running Complete ML Model Performance Optimization Guide")
        print("=" * 70)
        
        try:
            # Step 1: Data Optimization
            self.step_1_data_optimization()
            
            # Step 2: Model Optimization
            self.step_2_model_optimization()
            
            # Step 3: Training Optimization
            self.step_3_training_optimization()
            
            # Step 4: Inference Optimization
            self.step_4_inference_optimization()
            
            # Step 5: Benchmarking
            self.step_5_benchmarking()
            
            # Step 6: Combine Optimizations
            self.step_6_combine_optimizations()
            
            # Create roadmap
            self.create_optimization_roadmap()
            
            print("\n🎉 CONGRATULATIONS!")
            print("=" * 30)
            print("You've completed the ML Model Performance Optimization Playground!")
            print("\nNext steps:")
            print("• Explore the individual demo files in each directory")
            print("• Try the techniques on your own datasets")
            print("• Experiment with different combinations")
            print("• Share your results and learnings")
            print("\nHappy optimizing! 🚀")
            
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")
            print("Don't worry! You can run individual steps or check the specific demo files.")
            print("Each demo is designed to work independently.")

def main():
    """Run the quick start guide"""
    guide = QuickStartGuide()
    guide.run_complete_guide()

if __name__ == "__main__":
    main()
