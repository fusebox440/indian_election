"""
Pytest configuration and run script for the Twitter Sentiment Analysis project.
Author: Lakshya Khetan
"""

import sys
import pytest
from pathlib import Path


def run_tests():
    """Run the complete test suite with proper configuration."""
    
    # Add src to Python path
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))
    
    # Test configuration
    pytest_args = [
        # Test discovery
        str(project_root / "tests"),
        
        # Output configuration
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker usage
        
        # Coverage configuration
        "--cov=src",  # Coverage for src directory
        "--cov-report=html:htmlcov",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal coverage with missing lines
        "--cov-fail-under=80",  # Fail if coverage below 80%
        
        # Test markers
        "-m", "not slow",  # Skip slow tests by default
        
        # Warnings
        "--disable-warnings",  # Disable pytest warnings for cleaner output
        
        # Parallel execution (if pytest-xdist is installed)
        # "-n", "auto",  # Uncomment to run tests in parallel
    ]
    
    # Run pytest
    exit_code = pytest.main(pytest_args)
    return exit_code


def run_unit_tests():
    """Run only unit tests."""
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))
    
    pytest_args = [
        str(project_root / "tests" / "test_data"),
        str(project_root / "tests" / "test_models"),
        str(project_root / "tests" / "test_utils"),
        "-v",
        "-m", "unit",
        "--tb=short"
    ]
    
    return pytest.main(pytest_args)


def run_integration_tests():
    """Run only integration tests."""
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))
    
    pytest_args = [
        str(project_root / "tests" / "integration"),
        "-v",
        "-m", "integration",
        "--tb=long"
    ]
    
    return pytest.main(pytest_args)


def run_performance_tests():
    """Run performance tests."""
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))
    
    pytest_args = [
        str(project_root / "tests"),
        "-v",
        "-m", "slow",
        "--tb=short",
        "--durations=10"  # Show 10 slowest tests
    ]
    
    return pytest.main(pytest_args)


def run_coverage_report():
    """Generate and display coverage report."""
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))
    
    pytest_args = [
        str(project_root / "tests"),
        "--cov=src",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml:coverage.xml",
        "-m", "not slow"
    ]
    
    return pytest.main(pytest_args)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for Twitter Sentiment Analysis project")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "performance", "coverage"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    print(f"Running {args.type} tests...")
    
    if args.type == "all":
        exit_code = run_tests()
    elif args.type == "unit":
        exit_code = run_unit_tests()
    elif args.type == "integration":
        exit_code = run_integration_tests()
    elif args.type == "performance":
        exit_code = run_performance_tests()
    elif args.type == "coverage":
        exit_code = run_coverage_report()
    else:
        exit_code = run_tests()
    
    if exit_code == 0:
        print(f"\n✅ {args.type.capitalize()} tests completed successfully!")
    else:
        print(f"\n❌ {args.type.capitalize()} tests failed!")
    
    sys.exit(exit_code)