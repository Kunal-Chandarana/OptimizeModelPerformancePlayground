"""
Hyperparameter Tuning for Performance Optimization
==================================================

This demo shows how different hyperparameter tuning techniques can significantly
improve model performance. We'll compare various optimization strategies and
their impact on both accuracy and training time.

Key Techniques Covered:
- Grid Search
- Random Search
- Bayesian Optimization (Optuna)
- Genetic Algorithms
- Successive Halving
- Multi-objective optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
import time
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuningDemo:
    def __init__(self):
        self.results = {}
        self.study_results = {}
        
    def create_synthetic_data(self, dataset_type='classification', n_samples=2000, n_features=20):
        """Create synthetic dataset for hyperparameter tuning demo"""
        if dataset_type == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=15,
                n_redundant=5,
                n_clusters_per_class=1,
                random_state=42
            )
        else:  # regression
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=0.1,
                random_state=42
            )
        
        return X, y
    
    def grid_search_demo(self, X, y, model_class, param_grid, cv=5):
        """Demonstrate grid search hyperparameter tuning"""
        print("🔍 Testing Grid Search...")
        
        from sklearn.model_selection import GridSearchCV
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform grid search
        start_time = time.time()
        grid_search = GridSearchCV(
            model_class(random_state=42),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_scaled, y)
        tuning_time = time.time() - start_time
        
        # Get results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        n_combinations = len(grid_search.cv_results_['params'])
        
        return {
            'method': 'Grid Search',
            'best_score': best_score,
            'best_params': best_params,
            'tuning_time': tuning_time,
            'n_combinations': n_combinations,
            'cv_results': grid_search.cv_results_
        }
    
    def random_search_demo(self, X, y, model_class, param_distributions, n_iter=50, cv=5):
        """Demonstrate random search hyperparameter tuning"""
        print("🎲 Testing Random Search...")
        
        from sklearn.model_selection import RandomizedSearchCV
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform random search
        start_time = time.time()
        random_search = RandomizedSearchCV(
            model_class(random_state=42),
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        random_search.fit(X_scaled, y)
        tuning_time = time.time() - start_time
        
        # Get results
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        return {
            'method': 'Random Search',
            'best_score': best_score,
            'best_params': best_params,
            'tuning_time': tuning_time,
            'n_combinations': n_iter,
            'cv_results': random_search.cv_results_
        }
    
    def optuna_bayesian_demo(self, X, y, model_class, param_space, n_trials=50, cv=5):
        """Demonstrate Bayesian optimization with Optuna"""
        print("🧠 Testing Bayesian Optimization (Optuna)...")
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'log_float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            
            # Create model with sampled parameters
            model = model_class(random_state=42, **params)
            
            # Cross-validation
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            return scores.mean()
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials)
        tuning_time = time.time() - start_time
        
        # Get results
        best_params = study.best_params
        best_score = study.best_value
        
        return {
            'method': 'Bayesian Optimization (Optuna)',
            'best_score': best_score,
            'best_params': best_params,
            'tuning_time': tuning_time,
            'n_combinations': n_trials,
            'study': study
        }
    
    def optuna_genetic_demo(self, X, y, model_class, param_space, n_trials=50, cv=5):
        """Demonstrate genetic algorithm optimization with Optuna"""
        print("🧬 Testing Genetic Algorithm (Optuna CMA-ES)...")
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'log_float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            
            # Create model with sampled parameters
            model = model_class(random_state=42, **params)
            
            # Cross-validation
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            return scores.mean()
        
        # Create study with CMA-ES sampler
        study = optuna.create_study(
            direction='maximize',
            sampler=CmaEsSampler(seed=42)
        )
        
        # Optimize
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials)
        tuning_time = time.time() - start_time
        
        # Get results
        best_params = study.best_params
        best_score = study.best_value
        
        return {
            'method': 'Genetic Algorithm (CMA-ES)',
            'best_score': best_score,
            'best_params': best_params,
            'tuning_time': tuning_time,
            'n_combinations': n_trials,
            'study': study
        }
    
    def multi_objective_demo(self, X, y, model_class, param_space, n_trials=50, cv=5):
        """Demonstrate multi-objective optimization"""
        print("🎯 Testing Multi-Objective Optimization...")
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'log_float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            
            # Create model with sampled parameters
            model = model_class(random_state=42, **params)
            
            # Cross-validation
            start_time = time.time()
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            training_time = time.time() - start_time
            
            # Return both accuracy and negative training time (for minimization)
            return scores.mean(), -training_time
        
        # Create multi-objective study
        study = optuna.create_study(
            directions=['maximize', 'maximize'],  # maximize accuracy, maximize negative training time
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials)
        tuning_time = time.time() - start_time
        
        # Get Pareto front
        pareto_front = study.best_trials
        
        return {
            'method': 'Multi-Objective Optimization',
            'pareto_front': pareto_front,
            'tuning_time': tuning_time,
            'n_combinations': n_trials,
            'study': study
        }
    
    def define_parameter_spaces(self):
        """Define parameter spaces for different models"""
        param_spaces = {
            'RandomForestClassifier': {
                'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]}
            },
            'LogisticRegression': {
                'C': {'type': 'log_float', 'low': 0.001, 'high': 100},
                'penalty': {'type': 'categorical', 'choices': ['l1', 'l2', 'elasticnet']},
                'solver': {'type': 'categorical', 'choices': ['liblinear', 'saga']},
                'max_iter': {'type': 'int', 'low': 100, 'high': 2000}
            },
            'SVC': {
                'C': {'type': 'log_float', 'low': 0.001, 'high': 100},
                'gamma': {'type': 'log_float', 'low': 0.0001, 'high': 1},
                'kernel': {'type': 'categorical', 'choices': ['rbf', 'poly', 'sigmoid']},
                'degree': {'type': 'int', 'low': 2, 'high': 5}
            },
            'MLPClassifier': {
                'hidden_layer_sizes': {'type': 'categorical', 'choices': [(50,), (100,), (50, 50), (100, 50), (100, 100)]},
                'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'logistic']},
                'alpha': {'type': 'log_float', 'low': 0.0001, 'high': 1},
                'learning_rate': {'type': 'categorical', 'choices': ['constant', 'adaptive']},
                'max_iter': {'type': 'int', 'low': 200, 'high': 1000}
            }
        }
        
        # Grid search parameter grids
        grid_params = {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVC': {
                'C': [0.1, 1, 10],
                'gamma': [0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            },
            'MLPClassifier': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        return param_spaces, grid_params
    
    def run_comprehensive_demo(self):
        """Run comprehensive hyperparameter tuning comparison"""
        print("🚀 Starting Hyperparameter Tuning Performance Demo")
        print("=" * 60)
        
        # Create synthetic data
        X, y = self.create_synthetic_data(n_samples=1500, n_features=25)
        print(f"Dataset shape: {X.shape}, Classes: {len(np.unique(y))}")
        
        # Define parameter spaces
        param_spaces, grid_params = self.define_parameter_spaces()
        
        # Test different models
        models_to_test = ['RandomForestClassifier', 'LogisticRegression', 'SVC', 'MLPClassifier']
        
        for model_name in models_to_test:
            print(f"\n🔧 Testing {model_name}")
            print("-" * 40)
            
            model_class = globals()[model_name]
            param_space = param_spaces[model_name]
            grid_param = grid_params[model_name]
            
            # Run different tuning methods
            methods_results = {}
            
            # Grid Search
            try:
                grid_result = self.grid_search_demo(X, y, model_class, grid_param)
                methods_results['grid_search'] = grid_result
            except Exception as e:
                print(f"Grid search failed: {e}")
            
            # Random Search
            try:
                random_result = self.random_search_demo(X, y, model_class, param_space, n_iter=30)
                methods_results['random_search'] = random_result
            except Exception as e:
                print(f"Random search failed: {e}")
            
            # Bayesian Optimization
            try:
                bayesian_result = self.optuna_bayesian_demo(X, y, model_class, param_space, n_trials=30)
                methods_results['bayesian'] = bayesian_result
            except Exception as e:
                print(f"Bayesian optimization failed: {e}")
            
            # Genetic Algorithm
            try:
                genetic_result = self.optuna_genetic_demo(X, y, model_class, param_space, n_trials=30)
                methods_results['genetic'] = genetic_result
            except Exception as e:
                print(f"Genetic algorithm failed: {e}")
            
            # Multi-objective (only for RandomForest)
            if model_name == 'RandomForestClassifier':
                try:
                    multi_result = self.multi_objective_demo(X, y, model_class, param_space, n_trials=30)
                    methods_results['multi_objective'] = multi_result
                except Exception as e:
                    print(f"Multi-objective optimization failed: {e}")
            
            self.results[model_name] = methods_results
        
        # Display results
        self.display_results()
        
        return self.results
    
    def display_results(self):
        """Display comprehensive results"""
        print("\n" + "=" * 80)
        print("📊 HYPERPARAMETER TUNING PERFORMANCE RESULTS")
        print("=" * 80)
        
        for model_name, methods_results in self.results.items():
            print(f"\n🔧 {model_name}")
            print("-" * 50)
            
            for method_name, result in methods_results.items():
                if method_name == 'multi_objective':
                    print(f"{method_name:20} | Pareto Solutions: {len(result['pareto_front'])} | "
                          f"Time: {result['tuning_time']:.4f}s")
                else:
                    print(f"{method_name:20} | Best Score: {result['best_score']:.4f} | "
                          f"Time: {result['tuning_time']:.4f}s | "
                          f"Combinations: {result['n_combinations']}")
        
        # Find best performing methods
        self.find_best_methods()
    
    def find_best_methods(self):
        """Identify the best performing methods"""
        print("\n🏆 BEST PERFORMING HYPERPARAMETER TUNING METHODS")
        print("-" * 60)
        
        all_results = []
        
        # Collect all results
        for model_name, methods_results in self.results.items():
            for method_name, result in methods_results.items():
                if method_name != 'multi_objective':
                    all_results.append({
                        'model': model_name,
                        'method': method_name,
                        'best_score': result['best_score'],
                        'tuning_time': result['tuning_time'],
                        'n_combinations': result['n_combinations']
                    })
        
        # Sort by best score
        all_results.sort(key=lambda x: x['best_score'], reverse=True)
        
        print("Top 10 by Best Score:")
        for i, result in enumerate(all_results[:10], 1):
            print(f"{i:2}. {result['method']:20} ({result['model']:20}) | "
                  f"Score: {result['best_score']:.4f} | "
                  f"Time: {result['tuning_time']:.4f}s")
        
        # Sort by efficiency (score per time)
        all_results.sort(key=lambda x: x['best_score'] / x['tuning_time'], reverse=True)
        
        print("\nTop 10 by Efficiency (Score/Time):")
        for i, result in enumerate(all_results[:10], 1):
            efficiency = result['best_score'] / result['tuning_time']
            print(f"{i:2}. {result['method']:20} ({result['model']:20}) | "
                  f"Efficiency: {efficiency:.4f} | "
                  f"Score: {result['best_score']:.4f}")
    
    def plot_results(self):
        """Create visualization of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hyperparameter Tuning Performance Analysis', fontsize=16)
        
        # Collect all results for plotting
        all_data = []
        for model_name, methods_results in self.results.items():
            for method_name, result in methods_results.items():
                if method_name != 'multi_objective':
                    all_data.append({
                        'model': model_name,
                        'method': method_name,
                        'score': result['best_score'],
                        'time': result['tuning_time'],
                        'combinations': result['n_combinations']
                    })
        
        if not all_data:
            print("No data to plot")
            return
        
        df = pd.DataFrame(all_data)
        
        # Score comparison by method
        ax1 = axes[0, 0]
        method_scores = df.groupby('method')['score'].mean().sort_values(ascending=False)
        method_scores.plot(kind='bar', ax=ax1)
        ax1.set_title('Average Score by Method')
        ax1.set_ylabel('Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Time comparison by method
        ax2 = axes[0, 1]
        method_times = df.groupby('method')['time'].mean().sort_values(ascending=False)
        method_times.plot(kind='bar', ax=ax2)
        ax2.set_title('Average Time by Method')
        ax2.set_ylabel('Time (s)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Score vs Time scatter
        ax3 = axes[1, 0]
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            ax3.scatter(method_data['time'], method_data['score'], 
                       label=method, alpha=0.7, s=60)
        ax3.set_xlabel('Tuning Time (s)')
        ax3.set_ylabel('Best Score')
        ax3.set_title('Score vs Tuning Time')
        ax3.legend()
        
        # Score by model
        ax4 = axes[1, 1]
        model_scores = df.groupby('model')['score'].mean().sort_values(ascending=False)
        model_scores.plot(kind='bar', ax=ax4)
        ax4.set_title('Average Score by Model')
        ax4.set_ylabel('Score')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def main():
    """Run the hyperparameter tuning demo"""
    demo = HyperparameterTuningDemo()
    results = demo.run_comprehensive_demo()
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    demo.plot_results()
    
    print("\n✅ Hyperparameter Tuning Demo Complete!")
    print("\nKey Takeaways:")
    print("• Bayesian optimization often finds better solutions faster than grid search")
    print("• Random search can be surprisingly effective with limited computational budget")
    print("• Multi-objective optimization helps balance accuracy and training time")
    print("• Different models benefit from different tuning strategies")
    print("• Consider both performance and computational cost when choosing a method")

if __name__ == "__main__":
    main()
