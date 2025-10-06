"""
A/B Testing Infrastructure for ML Models
========================================

This demo focuses on building A/B testing infrastructure for ML models,
covering experiment design, traffic splitting, and statistical analysis.

Key Topics:
- Experiment design and hypothesis testing
- Traffic splitting and routing
- Statistical significance testing
- Risk management and rollback strategies
- Experiment analysis and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class ABTestingDemo:
    def __init__(self):
        self.experiments = {}
        self.traffic_data = {}
        self.results = {}
        
    def create_models(self):
        """Create different model versions for A/B testing"""
        print("🤖 Creating model versions for A/B testing...")
        
        # Generate training data
        X, y = make_classification(
            n_samples=10000,
            n_features=50,
            n_informative=40,
            n_redundant=10,
            n_classes=3,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model A: Baseline (Random Forest)
        model_a = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        model_a.fit(X_train_scaled, y_train)
        
        # Model B: Optimized (Random Forest with different params)
        model_b = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=3,
            random_state=42
        )
        model_b.fit(X_train_scaled, y_train)
        
        # Model C: Different algorithm (Logistic Regression)
        model_c = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        model_c.fit(X_train_scaled, y_train)
        
        return {
            'A': model_a,
            'B': model_b,
            'C': model_c
        }, scaler, X_test_scaled, y_test
    
    def design_experiment(self, experiment_name, models, traffic_split, duration_days=7):
        """Design an A/B test experiment"""
        print(f"🧪 Designing experiment: {experiment_name}")
        
        experiment = {
            'name': experiment_name,
            'models': models,
            'traffic_split': traffic_split,
            'duration_days': duration_days,
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(days=duration_days),
            'status': 'active',
            'metrics': {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'latency': [],
                'requests': []
            }
        }
        
        self.experiments[experiment_name] = experiment
        return experiment
    
    def simulate_traffic_routing(self, experiment, scaler, test_data, test_labels, 
                               total_requests=10000):
        """Simulate traffic routing for A/B testing"""
        print(f"🚦 Simulating traffic routing for {experiment['name']}...")
        
        # Generate random traffic
        np.random.seed(42)
        traffic_assignments = np.random.choice(
            list(experiment['models'].keys()),
            size=total_requests,
            p=list(experiment['traffic_split'].values())
        )
        
        # Simulate requests
        results = {
            'assignments': traffic_assignments,
            'predictions': {},
            'actuals': {},
            'latencies': {},
            'timestamps': []
        }
        
        for model_name in experiment['models'].keys():
            results['predictions'][model_name] = []
            results['actuals'][model_name] = []
            results['latencies'][model_name] = []
        
        for i in range(total_requests):
            # Get random test sample
            sample_idx = np.random.randint(0, len(test_data))
            sample = test_data[sample_idx:sample_idx+1]
            actual = test_labels[sample_idx]
            
            # Route to assigned model
            assigned_model = traffic_assignments[i]
            model = experiment['models'][assigned_model]
            
            # Make prediction
            start_time = time.time()
            sample_scaled = scaler.transform(sample)
            prediction = model.predict(sample_scaled)[0]
            latency = time.time() - start_time
            
            # Store results
            results['predictions'][assigned_model].append(prediction)
            results['actuals'][assigned_model].append(actual)
            results['latencies'][assigned_model].append(latency)
            results['timestamps'].append(datetime.now() + timedelta(seconds=i))
        
        return results
    
    def calculate_experiment_metrics(self, experiment, traffic_results):
        """Calculate metrics for each model in the experiment"""
        print(f"📊 Calculating metrics for {experiment['name']}...")
        
        model_metrics = {}
        
        for model_name in experiment['models'].keys():
            predictions = traffic_results['predictions'][model_name]
            actuals = traffic_results['actuals'][model_name]
            latencies = traffic_results['latencies'][model_name]
            
            if len(predictions) > 0:
                metrics = {
                    'accuracy': accuracy_score(actuals, predictions),
                    'precision': precision_score(actuals, predictions, average='weighted'),
                    'recall': recall_score(actuals, predictions, average='weighted'),
                    'f1_score': f1_score(actuals, predictions, average='weighted'),
                    'avg_latency': np.mean(latencies),
                    'p95_latency': np.percentile(latencies, 95),
                    'p99_latency': np.percentile(latencies, 99),
                    'total_requests': len(predictions),
                    'error_rate': 1 - accuracy_score(actuals, predictions)
                }
                
                model_metrics[model_name] = metrics
            else:
                model_metrics[model_name] = {
                    'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0,
                    'avg_latency': 0, 'p95_latency': 0, 'p99_latency': 0,
                    'total_requests': 0, 'error_rate': 1
                }
        
        return model_metrics
    
    def statistical_significance_test(self, model_a_metrics, model_b_metrics, 
                                    metric='accuracy', alpha=0.05):
        """Perform statistical significance testing"""
        print(f"📈 Testing statistical significance for {metric}...")
        
        # Simulate confidence intervals (in real scenario, you'd have actual data)
        n_a = model_a_metrics['total_requests']
        n_b = model_b_metrics['total_requests']
        
        if n_a == 0 or n_b == 0:
            return {
                'significant': False,
                'p_value': 1.0,
                'confidence_interval': (0, 0),
                'effect_size': 0
            }
        
        # Simulate statistical test results
        metric_a = model_a_metrics[metric]
        metric_b = model_b_metrics[metric]
        
        # Calculate effect size
        effect_size = metric_b - metric_a
        
        # Simulate p-value calculation (in practice, use proper statistical tests)
        if abs(effect_size) < 0.01:  # Small effect
            p_value = np.random.uniform(0.1, 0.5)
        elif abs(effect_size) < 0.05:  # Medium effect
            p_value = np.random.uniform(0.01, 0.1)
        else:  # Large effect
            p_value = np.random.uniform(0.001, 0.01)
        
        # Calculate confidence interval (simplified)
        se = np.sqrt(metric_a * (1 - metric_a) / n_a + metric_b * (1 - metric_b) / n_b)
        margin_error = 1.96 * se
        ci_lower = effect_size - margin_error
        ci_upper = effect_size + margin_error
        
        return {
            'significant': p_value < alpha,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': effect_size,
            'alpha': alpha
        }
    
    def risk_assessment(self, experiment, model_metrics):
        """Assess risks of the experiment"""
        print(f"⚠️ Assessing risks for {experiment['name']}...")
        
        risks = []
        
        # Check for significant performance degradation
        baseline_model = list(experiment['models'].keys())[0]
        baseline_accuracy = model_metrics[baseline_model]['accuracy']
        
        for model_name, metrics in model_metrics.items():
            if model_name != baseline_model:
                accuracy_drop = baseline_accuracy - metrics['accuracy']
                if accuracy_drop > 0.05:  # 5% accuracy drop
                    risks.append({
                        'type': 'performance_degradation',
                        'model': model_name,
                        'severity': 'high',
                        'description': f'Accuracy dropped by {accuracy_drop:.3f}',
                        'value': accuracy_drop
                    })
                
                # Check for latency increase
                baseline_latency = model_metrics[baseline_model]['avg_latency']
                latency_increase = metrics['avg_latency'] - baseline_latency
                if latency_increase > 0.01:  # 10ms increase
                    risks.append({
                        'type': 'latency_increase',
                        'model': model_name,
                        'severity': 'medium',
                        'description': f'Latency increased by {latency_increase:.4f}s',
                        'value': latency_increase
                    })
                
                # Check for high error rate
                if metrics['error_rate'] > 0.1:  # 10% error rate
                    risks.append({
                        'type': 'high_error_rate',
                        'model': model_name,
                        'severity': 'high',
                        'description': f'Error rate is {metrics["error_rate"]:.3f}',
                        'value': metrics['error_rate']
                    })
        
        return risks
    
    def generate_experiment_report(self, experiment, model_metrics, traffic_results):
        """Generate comprehensive experiment report"""
        print(f"📋 Generating experiment report for {experiment['name']}...")
        
        report = {
            'experiment_name': experiment['name'],
            'duration': experiment['duration_days'],
            'traffic_split': experiment['traffic_split'],
            'model_metrics': model_metrics,
            'statistical_tests': {},
            'risks': [],
            'recommendations': []
        }
        
        # Statistical significance tests
        model_names = list(experiment['models'].keys())
        baseline_model = model_names[0]
        
        for model_name in model_names[1:]:
            stat_test = self.statistical_significance_test(
                model_metrics[baseline_model],
                model_metrics[model_name],
                'accuracy'
            )
            report['statistical_tests'][f'{baseline_model}_vs_{model_name}'] = stat_test
        
        # Risk assessment
        report['risks'] = self.risk_assessment(experiment, model_metrics)
        
        # Generate recommendations
        best_model = max(model_metrics.keys(), 
                        key=lambda x: model_metrics[x]['accuracy'])
        
        if best_model != baseline_model:
            improvement = (model_metrics[best_model]['accuracy'] - 
                          model_metrics[baseline_model]['accuracy'])
            report['recommendations'].append(
                f"Consider promoting {best_model} to production "
                f"(accuracy improvement: {improvement:.3f})"
            )
        else:
            report['recommendations'].append(
                "No significant improvement found. Consider extending experiment."
            )
        
        return report
    
    def run_comprehensive_demo(self):
        """Run comprehensive A/B testing demo"""
        print("🚀 Starting A/B Testing Infrastructure Demo")
        print("=" * 60)
        
        # Create models
        models, scaler, test_data, test_labels = self.create_models()
        
        # Design experiments
        experiments = [
            {
                'name': 'Model_Optimization_Test',
                'models': {'A': models['A'], 'B': models['B']},
                'traffic_split': {'A': 0.5, 'B': 0.5},
                'duration_days': 7
            },
            {
                'name': 'Algorithm_Comparison_Test',
                'models': {'A': models['A'], 'C': models['C']},
                'traffic_split': {'A': 0.7, 'C': 0.3},
                'duration_days': 14
            },
            {
                'name': 'Multi_Model_Test',
                'models': {'A': models['A'], 'B': models['B'], 'C': models['C']},
                'traffic_split': {'A': 0.4, 'B': 0.3, 'C': 0.3},
                'duration_days': 10
            }
        ]
        
        # Run experiments
        for exp_config in experiments:
            print(f"\n🧪 Running experiment: {exp_config['name']}")
            
            # Design experiment
            experiment = self.design_experiment(**exp_config)
            
            # Simulate traffic
            traffic_results = self.simulate_traffic_routing(
                experiment, scaler, test_data, test_labels, total_requests=5000
            )
            
            # Calculate metrics
            model_metrics = self.calculate_experiment_metrics(experiment, traffic_results)
            
            # Generate report
            report = self.generate_experiment_report(experiment, model_metrics, traffic_results)
            
            # Store results
            self.results[exp_config['name']] = {
                'experiment': experiment,
                'traffic_results': traffic_results,
                'model_metrics': model_metrics,
                'report': report
            }
        
        # Display results
        self.display_results()
        
        return self.results
    
    def display_results(self):
        """Display comprehensive results"""
        print("\n" + "=" * 80)
        print("📊 A/B TESTING INFRASTRUCTURE RESULTS")
        print("=" * 80)
        
        for exp_name, exp_data in self.results.items():
            print(f"\n🧪 EXPERIMENT: {exp_name}")
            print("-" * 50)
            
            model_metrics = exp_data['model_metrics']
            report = exp_data['report']
            
            # Model performance comparison
            print("Model Performance:")
            for model_name, metrics in model_metrics.items():
                print(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}, "
                      f"Latency={metrics['avg_latency']:.4f}s, "
                      f"Requests={metrics['total_requests']}")
            
            # Statistical significance
            print("\nStatistical Significance:")
            for test_name, test_result in report['statistical_tests'].items():
                significance = "✓" if test_result['significant'] else "✗"
                print(f"  {test_name}: {significance} "
                      f"(p={test_result['p_value']:.4f}, "
                      f"effect={test_result['effect_size']:.4f})")
            
            # Risks
            if report['risks']:
                print("\nRisks Identified:")
                for risk in report['risks']:
                    print(f"  {risk['severity'].upper()}: {risk['description']}")
            else:
                print("\nNo significant risks identified")
            
            # Recommendations
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        # Overall insights
        self.print_overall_insights()
    
    def print_overall_insights(self):
        """Print overall insights and recommendations"""
        print(f"\n💡 A/B TESTING INFRASTRUCTURE INSIGHTS")
        print("-" * 50)
        print("• Always start with small traffic splits for new models")
        print("• Monitor both performance and operational metrics")
        print("• Use statistical significance testing for decision making")
        print("• Implement automatic rollback for critical failures")
        print("• Set up comprehensive monitoring and alerting")
        print("• Document all experiments and their outcomes")
        print("• Consider business impact, not just technical metrics")
        print("• Use feature flags for gradual rollouts")
    
    def plot_experiment_analysis(self):
        """Create visualizations for experiment analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('A/B Testing Analysis Dashboard', fontsize=16)
        
        # Plot 1: Model performance comparison
        ax1 = axes[0, 0]
        exp_names = list(self.results.keys())
        model_names = []
        accuracies = []
        
        for exp_name, exp_data in self.results.items():
            for model_name, metrics in exp_data['model_metrics'].items():
                model_names.append(f"{exp_name}_{model_name}")
                accuracies.append(metrics['accuracy'])
        
        bars = ax1.bar(range(len(model_names)), accuracies)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Latency comparison
        ax2 = axes[0, 1]
        latencies = []
        for exp_name, exp_data in self.results.items():
            for model_name, metrics in exp_data['model_metrics'].items():
                latencies.append(metrics['avg_latency'])
        
        bars2 = ax2.bar(range(len(model_names)), latencies, color='orange')
        ax2.set_title('Model Latency Comparison')
        ax2.set_ylabel('Average Latency (s)')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Traffic distribution
        ax3 = axes[1, 0]
        traffic_splits = []
        for exp_name, exp_data in self.results.items():
            for model_name, split in exp_data['experiment']['traffic_split'].items():
                traffic_splits.append(split)
        
        ax3.pie(traffic_splits, labels=model_names, autopct='%1.1f%%')
        ax3.set_title('Traffic Distribution')
        
        # Plot 4: Statistical significance
        ax4 = axes[1, 1]
        p_values = []
        test_names = []
        for exp_name, exp_data in self.results.items():
            for test_name, test_result in exp_data['report']['statistical_tests'].items():
                p_values.append(test_result['p_value'])
                test_names.append(test_name)
        
        colors = ['green' if p < 0.05 else 'red' for p in p_values]
        bars3 = ax4.bar(range(len(test_names)), p_values, color=colors, alpha=0.7)
        ax4.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold')
        ax4.set_title('Statistical Significance (p-values)')
        ax4.set_ylabel('p-value')
        ax4.set_xticks(range(len(test_names)))
        ax4.set_xticklabels(test_names, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Run the A/B testing demo"""
    demo = ABTestingDemo()
    results = demo.run_comprehensive_demo()
    
    # Create visualizations
    print("\n📊 Creating experiment analysis...")
    demo.plot_experiment_analysis()
    
    print("\n✅ A/B Testing Infrastructure Demo Complete!")
    print("\nKey Takeaways for ML Infrastructure Engineers:")
    print("• A/B testing is essential for safe model deployments")
    print("• Statistical significance testing prevents false positives")
    print("• Risk assessment helps prevent production issues")
    print("• Traffic splitting allows gradual rollouts")
    print("• Comprehensive monitoring is crucial during experiments")

if __name__ == "__main__":
    main()
