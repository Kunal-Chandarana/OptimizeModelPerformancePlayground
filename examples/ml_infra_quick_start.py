"""
ML Infrastructure Engineer Quick Start Guide
===========================================

This guide is specifically designed for ML Infrastructure Engineers
who want to focus on production deployment, scalability, and operational concerns.

Run this to get started with ML infrastructure optimization techniques!
"""

import sys
import os
import subprocess

def main():
    """ML Infrastructure Engineer Quick Start"""
    print("🚀 ML Infrastructure Engineer Quick Start Guide")
    print("=" * 60)
    print("\nThis guide focuses on production ML infrastructure optimization.")
    print("Perfect for engineers working on ML systems at scale!")
    
    print("\n📚 Available ML Infrastructure Demos:")
    print("-" * 50)
    print("1. Model Serving Optimization")
    print("   - REST API serving strategies")
    print("   - Load balancing and auto-scaling")
    print("   - Resource utilization optimization")
    print("   - Latency vs throughput trade-offs")
    
    print("\n2. ML Observability & Monitoring")
    print("   - Model performance monitoring")
    print("   - Data drift detection")
    print("   - Resource utilization tracking")
    print("   - Alerting and SLA monitoring")
    
    print("\n3. A/B Testing Infrastructure")
    print("   - Experiment design and traffic splitting")
    print("   - Statistical significance testing")
    print("   - Risk management and rollback")
    print("   - Experiment analysis and reporting")
    
    print("\n4. Cost Optimization")
    print("   - Resource cost analysis")
    print("   - Auto-scaling strategies")
    print("   - Spot instances and reserved capacity")
    print("   - Cost monitoring and budgeting")
    
    print("\n5. Model Quantization & Compression")
    print("   - Model size reduction")
    print("   - Inference speed optimization")
    print("   - Production deployment optimization")
    
    print("\n🎯 Recommended Learning Path for ML Infra Engineers:")
    print("-" * 60)
    print("1. Start with Model Serving Optimization")
    print("2. Learn ML Observability & Monitoring")
    print("3. Master A/B Testing Infrastructure")
    print("4. Optimize for Cost Efficiency")
    print("5. Explore Model Compression Techniques")
    
    print("\n🚀 Ready to start? Choose a demo to run:")
    print("-" * 50)
    
    while True:
        print("\nSelect a demo to run:")
        print("1. Model Serving Optimization")
        print("2. ML Observability & Monitoring")
        print("3. A/B Testing Infrastructure")
        print("4. Cost Optimization")
        print("5. Model Quantization Demo")
        print("6. Run All Demos")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            print("\n🌐 Running Model Serving Optimization Demo...")
            try:
                subprocess.run([sys.executable, 'production_optimization/model_serving_demo.py'])
            except Exception as e:
                print(f"Error running demo: {e}")
        
        elif choice == '2':
            print("\n📊 Running ML Observability & Monitoring Demo...")
            try:
                subprocess.run([sys.executable, 'monitoring/ml_observability_demo.py'])
            except Exception as e:
                print(f"Error running demo: {e}")
        
        elif choice == '3':
            print("\n🧪 Running A/B Testing Infrastructure Demo...")
            try:
                subprocess.run([sys.executable, 'deployment/ab_testing_demo.py'])
            except Exception as e:
                print(f"Error running demo: {e}")
        
        elif choice == '4':
            print("\n💰 Running Cost Optimization Demo...")
            try:
                subprocess.run([sys.executable, 'infrastructure/cost_optimization_demo.py'])
            except Exception as e:
                print(f"Error running demo: {e}")
        
        elif choice == '5':
            print("\n⚡ Running Model Quantization Demo...")
            try:
                subprocess.run([sys.executable, 'inference_optimization/model_quantization_demo.py'])
            except Exception as e:
                print(f"Error running demo: {e}")
        
        elif choice == '6':
            print("\n🎯 Running All ML Infrastructure Demos...")
            demos = [
                ('Model Serving', 'production_optimization/model_serving_demo.py'),
                ('ML Observability', 'monitoring/ml_observability_demo.py'),
                ('A/B Testing', 'deployment/ab_testing_demo.py'),
                ('Cost Optimization', 'infrastructure/cost_optimization_demo.py'),
                ('Model Quantization', 'inference_optimization/model_quantization_demo.py')
            ]
            
            for demo_name, demo_path in demos:
                print(f"\n🚀 Running {demo_name} Demo...")
                try:
                    subprocess.run([sys.executable, demo_path])
                except Exception as e:
                    print(f"Error running {demo_name}: {e}")
                print(f"✅ {demo_name} Demo Complete!")
        
        elif choice == '7':
            print("\n👋 Thanks for using the ML Infrastructure Optimization Playground!")
            print("Happy optimizing! 🚀")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter a number between 1-7.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

