#!/usr/bin/env python3
"""
Test execution script for the neuromorphic brain simulation project.

This script provides a convenient way to run the unified test suite with
various options and configurations.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description="Running command"):
    """Run a command and return the result."""
    print(f"🚀 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Success!")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ Failed with return code {result.returncode}")
        if result.stderr:
            print(f"Error: {result.stderr}")
        if result.stdout:
            print(f"Output: {result.stdout}")
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run neuromorphic brain simulation tests")
    
    # Test selection options
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--no-slow', action='store_true', help='Skip slow tests')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    
    # Specific module options
    parser.add_argument('--neuron', action='store_true', help='Test neuron_factory module only')
    parser.add_argument('--layers', action='store_true', help='Test layers_factory module only')
    parser.add_argument('--brain', action='store_true', help='Test brain module only')
    
    # Output options
    parser.add_argument('--coverage', action='store_true', help='Run with coverage reporting')
    parser.add_argument('--html-cov', action='store_true', help='Generate HTML coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    
    # Advanced options
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel (requires pytest-xdist)')
    parser.add_argument('--profile', action='store_true', help='Profile test execution')
    parser.add_argument('--pdb', action='store_true', help='Drop into debugger on failures')
    
    args = parser.parse_args()
    
    # Validate project structure
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"
    src_dir = project_root / "src"
    
    if not tests_dir.exists():
        print(f"❌ Tests directory not found: {tests_dir}")
        return 1
        
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return 1
    
    print("🧪 NEUROMORPHIC BRAIN SIMULATION - TEST RUNNER")
    print("=" * 60)
    print(f"📁 Project root: {project_root}")
    print(f"📁 Tests directory: {tests_dir}")
    print(f"📁 Source directory: {src_dir}")
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    test_files = []
    if args.neuron:
        test_files.append("tests/test_neuron_factory_unified.py")
    elif args.layers:
        test_files.append("tests/test_layers_factory_unified.py")
    elif args.brain:
        test_files.append("tests/test_brain_unified.py")
    else:
        test_files.append("tests/")
    
    cmd.extend(test_files)
    
    # Add marker filters
    markers = []
    if args.unit:
        markers.append("unit")
    elif args.integration:
        markers.append("integration")
    elif args.performance:
        markers.append("performance")
    
    if args.no_slow:
        markers.append("not slow")
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    # Add output options
    if args.verbose:
        cmd.append("-v")
    elif args.quiet:
        cmd.append("-q")
    else:
        cmd.append("-v")  # Default to verbose
    
    # Add coverage options
    if args.coverage or args.html_cov:
        cmd.extend(["--cov=src"])
        cmd.extend(["--cov-report=term-missing"])
        
        if args.html_cov:
            cmd.extend(["--cov-report=html"])
    
    # Add advanced options
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    if args.profile:
        cmd.append("--profile")
    
    if args.pdb:
        cmd.append("--pdb")
    
    # Add standard useful options
    cmd.extend([
        "--tb=short",           # Short traceback format
        "--durations=10",       # Show 10 slowest tests
        "--strict-markers",     # Enforce marker registration
        "--color=yes"          # Colored output
    ])
    
    print(f"\n🎯 TEST CONFIGURATION:")
    if args.unit:
        print("  ✅ Unit tests only")
    elif args.integration:
        print("  ✅ Integration tests only")
    elif args.performance:
        print("  ✅ Performance tests only")
    else:
        print("  ✅ All test types")
    
    if args.no_slow:
        print("  ⚡ Skipping slow tests")
    
    if args.coverage:
        print("  📊 Coverage reporting enabled")
    
    if args.parallel:
        print("  🚀 Parallel execution enabled")
    
    # Show which modules will be tested
    print(f"\n📋 MODULES TO TEST:")
    if args.neuron:
        print("  🧠 neuron_factory.py")
    elif args.layers:
        print("  🏗️  layers_factory.py")
    elif args.brain:
        print("  🧠 brain.py")
    else:
        print("  🧠 All modules (neuron_factory, layers_factory, brain)")
    
    # Run the tests
    print(f"\n🚀 EXECUTING TESTS:")
    success = run_command(cmd, "Running pytest")
    
    if success:
        print(f"\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        
        if args.html_cov:
            coverage_dir = project_root / "htmlcov"
            if coverage_dir.exists():
                print(f"📊 HTML coverage report generated: {coverage_dir / 'index.html'}")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())