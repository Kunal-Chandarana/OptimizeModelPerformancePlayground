#!/usr/bin/env python3
"""
Setup and Run Script for ML Model Performance Optimization Playground
====================================================================

This script helps you get started quickly with the playground.
It will check dependencies and guide you through running the demos.
"""

import sys
import subprocess
import importlib
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("\n🔧 Installing missing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_simple.txt'])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_simple_demo():
    """Run the simple demo"""
    print("\n🚀 Running Simple Demo...")
    try:
        subprocess.run([sys.executable, 'examples/simple_demo.py'])
        return True
    except Exception as e:
        print(f"❌ Failed to run demo: {e}")
        return False

def show_menu():
    """Show the main menu"""
    print("\n" + "="*60)
    print("🎯 ML MODEL PERFORMANCE OPTIMIZATION PLAYGROUND")
    print("="*60)
    print("\nChoose what you'd like to do:")
    print("1. Run Simple Demo (recommended for beginners)")
    print("2. Run Feature Engineering Demo")
    print("3. Run Hyperparameter Tuning Demo")
    print("4. Run Learning Rate Optimization Demo")
    print("5. Run Model Quantization Demo")
    print("6. Run Performance Profiler Demo")
    print("7. Check Dependencies")
    print("8. Install Dependencies")
    print("9. Exit")
    print("\n" + "-"*60)

def main():
    """Main function"""
    print("Welcome to the ML Model Performance Optimization Playground! 🚀")
    
    while True:
        show_menu()
        choice = input("Enter your choice (1-9): ").strip()
        
        if choice == '1':
            print("\n🎯 Running Simple Demo...")
            if run_simple_demo():
                print("\n✅ Simple demo completed successfully!")
            else:
                print("\n❌ Simple demo failed. Please check dependencies.")
        
        elif choice == '2':
            print("\n🔧 Running Feature Engineering Demo...")
            try:
                subprocess.run([sys.executable, 'data_optimization/feature_engineering_demo.py'])
            except Exception as e:
                print(f"❌ Failed to run feature engineering demo: {e}")
        
        elif choice == '3':
            print("\n⚙️ Running Hyperparameter Tuning Demo...")
            try:
                subprocess.run([sys.executable, 'model_optimization/hyperparameter_tuning_demo.py'])
            except Exception as e:
                print(f"❌ Failed to run hyperparameter tuning demo: {e}")
        
        elif choice == '4':
            print("\n🏋️ Running Learning Rate Optimization Demo...")
            try:
                subprocess.run([sys.executable, 'training_optimization/learning_rate_optimization.py'])
            except Exception as e:
                print(f"❌ Failed to run learning rate optimization demo: {e}")
        
        elif choice == '5':
            print("\n⚡ Running Model Quantization Demo...")
            try:
                subprocess.run([sys.executable, 'inference_optimization/model_quantization_demo.py'])
            except Exception as e:
                print(f"❌ Failed to run model quantization demo: {e}")
        
        elif choice == '6':
            print("\n📊 Running Performance Profiler Demo...")
            try:
                subprocess.run([sys.executable, 'benchmarking/performance_profiler.py'])
            except Exception as e:
                print(f"❌ Failed to run performance profiler demo: {e}")
        
        elif choice == '7':
            print("\n🔍 Checking Dependencies...")
            missing = check_dependencies()
            if missing:
                print(f"\n❌ Missing packages: {', '.join(missing)}")
                print("Run option 8 to install them.")
            else:
                print("\n✅ All required dependencies are installed!")
        
        elif choice == '8':
            print("\n🔧 Installing Dependencies...")
            missing = check_dependencies()
            if missing:
                if install_dependencies():
                    print("\n✅ Dependencies installed successfully!")
                else:
                    print("\n❌ Failed to install dependencies. Please install manually.")
            else:
                print("\n✅ All dependencies are already installed!")
        
        elif choice == '9':
            print("\n👋 Thanks for using the ML Model Performance Optimization Playground!")
            print("Happy optimizing! 🚀")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter a number between 1-9.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

