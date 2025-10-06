"""
ML Observability and Monitoring Demo
====================================

This demo focuses on monitoring and observability for ML systems in production,
covering metrics collection, alerting, and operational concerns.

Key Topics:
- Model performance monitoring
- Data drift detection
- Resource utilization tracking
- Alerting and incident response
- SLA monitoring and compliance
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class MLObservabilityDemo:
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.model_performance = {}
        self.resource_metrics = {}
        
    def create_production_model(self):
        """Create a model for monitoring"""
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
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        return model, scaler, X_test_scaled, y_test
    
    def simulate_production_traffic(self, model, scaler, duration_hours=24):
        """Simulate production traffic over time"""
        print(f"📊 Simulating {duration_hours} hours of production traffic...")
        
        # Generate time series data
        timestamps = []
        predictions = []
        actuals = []
        latencies = []
        cpu_usage = []
        memory_usage = []
        
        # Simulate different traffic patterns throughout the day
        for hour in range(duration_hours):
            # Simulate traffic patterns (higher during business hours)
            if 9 <= hour % 24 <= 17:  # Business hours
                requests_per_hour = np.random.poisson(1000)
            else:  # Off hours
                requests_per_hour = np.random.poisson(200)
            
            for _ in range(requests_per_hour):
                # Generate request data
                X_new, y_new = make_classification(
                    n_samples=1,
                    n_features=50,
                    n_informative=40,
                    n_redundant=10,
                    n_classes=3,
                    random_state=42 + hour
                )
                
                # Simulate prediction
                start_time = time.time()
                X_scaled = scaler.transform(X_new)
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0].max()
                latency = time.time() - start_time
                
                # Simulate system metrics
                cpu = psutil.cpu_percent() + np.random.normal(0, 5)
                memory = psutil.virtual_memory().percent + np.random.normal(0, 2)
                
                timestamps.append(datetime.now() + timedelta(hours=hour))
                predictions.append(prediction)
                actuals.append(y_new[0])
                latencies.append(latency)
                cpu_usage.append(max(0, min(100, cpu)))
                memory_usage.append(max(0, min(100, memory)))
        
        return {
            'timestamps': timestamps,
            'predictions': predictions,
            'actuals': actuals,
            'latencies': latencies,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage
        }
    
    def calculate_model_metrics(self, predictions, actuals):
        """Calculate model performance metrics"""
        return {
            'accuracy': accuracy_score(actuals, predictions),
            'precision': precision_score(actuals, predictions, average='weighted'),
            'recall': recall_score(actuals, predictions, average='weighted'),
            'f1_score': f1_score(actuals, predictions, average='weighted')
        }
    
    def detect_data_drift(self, baseline_data, new_data, threshold=0.1):
        """Detect data drift using statistical tests"""
        print("🔍 Detecting data drift...")
        
        drift_detected = {}
        
        # Statistical drift detection (simplified)
        for feature in range(baseline_data.shape[1]):
            baseline_mean = np.mean(baseline_data[:, feature])
            baseline_std = np.std(baseline_data[:, feature])
            
            new_mean = np.mean(new_data[:, feature])
            new_std = np.std(new_data[:, feature])
            
            # Simple drift detection based on mean and std changes
            mean_change = abs(new_mean - baseline_mean) / baseline_mean if baseline_mean != 0 else 0
            std_change = abs(new_std - baseline_std) / baseline_std if baseline_std != 0 else 0
            
            drift_detected[f'feature_{feature}'] = {
                'mean_change': mean_change,
                'std_change': std_change,
                'drift_detected': mean_change > threshold or std_change > threshold
            }
        
        return drift_detected
    
    def monitor_model_performance(self, traffic_data):
        """Monitor model performance over time"""
        print("📈 Monitoring model performance...")
        
        # Calculate metrics for different time windows
        window_size = 100  # 100 requests per window
        performance_history = []
        
        for i in range(0, len(traffic_data['predictions']), window_size):
            window_predictions = traffic_data['predictions'][i:i+window_size]
            window_actuals = traffic_data['actuals'][i:i+window_size]
            window_latencies = traffic_data['latencies'][i:i+window_size]
            
            if len(window_predictions) > 0:
                metrics = self.calculate_model_metrics(window_predictions, window_actuals)
                metrics['avg_latency'] = np.mean(window_latencies)
                metrics['p95_latency'] = np.percentile(window_latencies, 95)
                metrics['window_start'] = traffic_data['timestamps'][i]
                
                performance_history.append(metrics)
        
        return performance_history
    
    def setup_alerting_rules(self):
        """Setup alerting rules for monitoring"""
        return {
            'accuracy_threshold': 0.85,  # Alert if accuracy drops below 85%
            'latency_threshold': 0.1,    # Alert if latency exceeds 100ms
            'cpu_threshold': 80,         # Alert if CPU usage exceeds 80%
            'memory_threshold': 85,      # Alert if memory usage exceeds 85%
            'error_rate_threshold': 0.05 # Alert if error rate exceeds 5%
        }
    
    def check_alerts(self, performance_history, resource_metrics, alert_rules):
        """Check for alert conditions"""
        print("🚨 Checking alert conditions...")
        
        alerts = []
        
        for i, metrics in enumerate(performance_history):
            timestamp = metrics['window_start']
            
            # Accuracy alert
            if metrics['accuracy'] < alert_rules['accuracy_threshold']:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'accuracy_low',
                    'severity': 'high',
                    'message': f"Accuracy dropped to {metrics['accuracy']:.3f}",
                    'value': metrics['accuracy'],
                    'threshold': alert_rules['accuracy_threshold']
                })
            
            # Latency alert
            if metrics['avg_latency'] > alert_rules['latency_threshold']:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'latency_high',
                    'severity': 'medium',
                    'message': f"Latency increased to {metrics['avg_latency']:.3f}s",
                    'value': metrics['avg_latency'],
                    'threshold': alert_rules['latency_threshold']
                })
            
            # CPU alert
            if i < len(resource_metrics['cpu_usage']):
                cpu_usage = resource_metrics['cpu_usage'][i]
                if cpu_usage > alert_rules['cpu_threshold']:
                    alerts.append({
                        'timestamp': timestamp,
                        'type': 'cpu_high',
                        'severity': 'medium',
                        'message': f"CPU usage increased to {cpu_usage:.1f}%",
                        'value': cpu_usage,
                        'threshold': alert_rules['cpu_threshold']
                    })
            
            # Memory alert
            if i < len(resource_metrics['memory_usage']):
                memory_usage = resource_metrics['memory_usage'][i]
                if memory_usage > alert_rules['memory_threshold']:
                    alerts.append({
                        'timestamp': timestamp,
                        'type': 'memory_high',
                        'severity': 'high',
                        'message': f"Memory usage increased to {memory_usage:.1f}%",
                        'value': memory_usage,
                        'threshold': alert_rules['memory_threshold']
                    })
        
        return alerts
    
    def calculate_sla_metrics(self, performance_history, sla_targets):
        """Calculate SLA compliance metrics"""
        print("📋 Calculating SLA compliance...")
        
        sla_metrics = {
            'availability': 0,
            'accuracy_sla': 0,
            'latency_sla': 0,
            'total_requests': len(performance_history)
        }
        
        if not performance_history:
            return sla_metrics
        
        # Calculate availability (simplified)
        sla_metrics['availability'] = 1.0  # Assume 100% availability for demo
        
        # Calculate accuracy SLA compliance
        accuracy_violations = sum(1 for metrics in performance_history 
                                if metrics['accuracy'] < sla_targets['accuracy'])
        sla_metrics['accuracy_sla'] = 1 - (accuracy_violations / len(performance_history))
        
        # Calculate latency SLA compliance
        latency_violations = sum(1 for metrics in performance_history 
                               if metrics['avg_latency'] > sla_targets['latency'])
        sla_metrics['latency_sla'] = 1 - (latency_violations / len(performance_history))
        
        return sla_metrics
    
    def run_comprehensive_demo(self):
        """Run comprehensive ML observability demo"""
        print("🚀 Starting ML Observability and Monitoring Demo")
        print("=" * 60)
        
        # Create production model
        print("🤖 Creating production model...")
        model, scaler, X_test, y_test = self.create_production_model()
        
        # Simulate production traffic
        traffic_data = self.simulate_production_traffic(model, scaler, duration_hours=12)
        
        # Monitor model performance
        performance_history = self.monitor_model_performance(traffic_data)
        
        # Setup alerting
        alert_rules = self.setup_alerting_rules()
        alerts = self.check_alerts(performance_history, traffic_data, alert_rules)
        
        # Calculate SLA metrics
        sla_targets = {
            'accuracy': 0.85,
            'latency': 0.1,
            'availability': 0.99
        }
        sla_metrics = self.calculate_sla_metrics(performance_history, sla_targets)
        
        # Store results
        self.results = {
            'traffic_data': traffic_data,
            'performance_history': performance_history,
            'alerts': alerts,
            'sla_metrics': sla_metrics,
            'alert_rules': alert_rules
        }
        
        # Display results
        self.display_results()
        
        return self.results
    
    def display_results(self):
        """Display comprehensive results"""
        print("\n" + "=" * 80)
        print("📊 ML OBSERVABILITY AND MONITORING RESULTS")
        print("=" * 80)
        
        # Performance summary
        if self.results['performance_history']:
            latest_metrics = self.results['performance_history'][-1]
            print(f"\n📈 CURRENT PERFORMANCE METRICS")
            print("-" * 50)
            print(f"Accuracy: {latest_metrics['accuracy']:.4f}")
            print(f"Precision: {latest_metrics['precision']:.4f}")
            print(f"Recall: {latest_metrics['recall']:.4f}")
            print(f"F1-Score: {latest_metrics['f1_score']:.4f}")
            print(f"Avg Latency: {latest_metrics['avg_latency']:.4f}s")
            print(f"P95 Latency: {latest_metrics['p95_latency']:.4f}s")
        
        # SLA compliance
        print(f"\n📋 SLA COMPLIANCE")
        print("-" * 50)
        sla = self.results['sla_metrics']
        print(f"Availability: {sla['availability']:.2%}")
        print(f"Accuracy SLA: {sla['accuracy_sla']:.2%}")
        print(f"Latency SLA: {sla['latency_sla']:.2%}")
        print(f"Total Requests: {sla['total_requests']}")
        
        # Alerts summary
        print(f"\n🚨 ALERTS SUMMARY")
        print("-" * 50)
        alerts = self.results['alerts']
        if alerts:
            alert_types = {}
            for alert in alerts:
                alert_type = alert['type']
                if alert_type not in alert_types:
                    alert_types[alert_type] = 0
                alert_types[alert_type] += 1
            
            for alert_type, count in alert_types.items():
                print(f"{alert_type}: {count} alerts")
        else:
            print("No alerts triggered")
        
        # Resource utilization
        traffic_data = self.results['traffic_data']
        print(f"\n💻 RESOURCE UTILIZATION")
        print("-" * 50)
        print(f"Avg CPU Usage: {np.mean(traffic_data['cpu_usage']):.1f}%")
        print(f"Max CPU Usage: {np.max(traffic_data['cpu_usage']):.1f}%")
        print(f"Avg Memory Usage: {np.mean(traffic_data['memory_usage']):.1f}%")
        print(f"Max Memory Usage: {np.max(traffic_data['memory_usage']):.1f}%")
        
        # Recommendations
        self.print_recommendations()
    
    def print_recommendations(self):
        """Print monitoring recommendations"""
        print(f"\n💡 MONITORING RECOMMENDATIONS")
        print("-" * 50)
        print("• Set up automated alerting for performance degradation")
        print("• Monitor both average and percentile latencies")
        print("• Track model accuracy trends over time")
        print("• Implement data drift detection")
        print("• Monitor resource utilization patterns")
        print("• Set up SLA dashboards for stakeholders")
        print("• Implement incident response procedures")
        print("• Regular model retraining based on performance metrics")
    
    def plot_monitoring_dashboard(self):
        """Create a monitoring dashboard visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ML System Monitoring Dashboard', fontsize=16)
        
        performance_history = self.results['performance_history']
        traffic_data = self.results['traffic_data']
        
        if not performance_history:
            print("No performance data to plot")
            return
        
        # Plot 1: Accuracy over time
        ax1 = axes[0, 0]
        timestamps = [metrics['window_start'] for metrics in performance_history]
        accuracies = [metrics['accuracy'] for metrics in performance_history]
        ax1.plot(timestamps, accuracies, marker='o', linewidth=2)
        ax1.axhline(y=0.85, color='r', linestyle='--', label='SLA Threshold')
        ax1.set_title('Model Accuracy Over Time')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Latency over time
        ax2 = axes[0, 1]
        latencies = [metrics['avg_latency'] for metrics in performance_history]
        ax2.plot(timestamps, latencies, marker='o', linewidth=2, color='orange')
        ax2.axhline(y=0.1, color='r', linestyle='--', label='SLA Threshold')
        ax2.set_title('Average Latency Over Time')
        ax2.set_ylabel('Latency (s)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Resource utilization
        ax3 = axes[0, 2]
        cpu_usage = traffic_data['cpu_usage']
        memory_usage = traffic_data['memory_usage']
        ax3.plot(cpu_usage, label='CPU Usage', alpha=0.7)
        ax3.plot(memory_usage, label='Memory Usage', alpha=0.7)
        ax3.axhline(y=80, color='r', linestyle='--', label='CPU Threshold')
        ax3.axhline(y=85, color='orange', linestyle='--', label='Memory Threshold')
        ax3.set_title('Resource Utilization')
        ax3.set_ylabel('Usage (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Request distribution
        ax4 = axes[1, 0]
        # Simulate request distribution by hour
        hourly_requests = np.random.poisson(100, 24)
        hours = range(24)
        ax4.bar(hours, hourly_requests, alpha=0.7)
        ax4.set_title('Request Distribution by Hour')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Number of Requests')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Error rate
        ax5 = axes[1, 1]
        # Simulate error rate over time
        error_rates = np.random.exponential(0.02, len(performance_history))
        ax5.plot(timestamps, error_rates, marker='o', linewidth=2, color='red')
        ax5.axhline(y=0.05, color='r', linestyle='--', label='Error Threshold')
        ax5.set_title('Error Rate Over Time')
        ax5.set_ylabel('Error Rate')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: SLA compliance
        ax6 = axes[1, 2]
        sla_metrics = self.results['sla_metrics']
        metrics = ['Availability', 'Accuracy SLA', 'Latency SLA']
        values = [sla_metrics['availability'], sla_metrics['accuracy_sla'], sla_metrics['latency_sla']]
        colors = ['green' if v >= 0.95 else 'orange' if v >= 0.9 else 'red' for v in values]
        
        bars = ax6.bar(metrics, values, color=colors, alpha=0.7)
        ax6.set_title('SLA Compliance')
        ax6.set_ylabel('Compliance Rate')
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

def main():
    """Run the ML observability demo"""
    demo = MLObservabilityDemo()
    results = demo.run_comprehensive_demo()
    
    # Create monitoring dashboard
    print("\n📊 Creating monitoring dashboard...")
    demo.plot_monitoring_dashboard()
    
    print("\n✅ ML Observability Demo Complete!")
    print("\nKey Takeaways for ML Infrastructure Engineers:")
    print("• Comprehensive monitoring is essential for ML systems")
    print("• Set up alerting for both performance and resource metrics")
    print("• Monitor SLA compliance continuously")
    print("• Implement data drift detection")
    print("• Use dashboards for operational visibility")

if __name__ == "__main__":
    main()
