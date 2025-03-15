#!/usr/bin/env python
"""
Cleanup script for AFP implementation.

This script helps clean up unnecessary test files after organizing
the test structure. It identifies duplicate files and allows users
to remove them from the root directory.
"""

import os
import sys
import shutil

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80 + "\n")

def confirm(message):
    """Ask for user confirmation."""
    response = input(f"{message} (y/n): ").lower()
    return response == 'y' or response == 'yes'

def main():
    """Main cleanup function."""
    print_header("AFP Implementation Cleanup")
    
    # Check if we're in the right directory
    if not os.path.exists('tests') or not os.path.exists(os.path.join('tests', 'unit')):
        print("Error: This script must be run from the project root directory.")
        print("The 'tests' directory with subdirectories must exist.")
        sys.exit(1)
    
    # Files that have been moved to tests/unit
    unit_test_files = [
        'test_message.py',
        'test_subscription.py',
        'test_bus.py',
        'test_afp_components.py',
        'test_structure.py',
        'test_imports.py'
    ]
    
    # Files that have been moved to tests/performance
    performance_test_files = [
        'afp_performance_test.py',
        'baseline_test.py'
    ]
    
    # Files that have been moved to tests/integration
    integration_test_files = [
        'afp_ai_integration_test.py',
        'afp_example_app.py'
    ]
    
    # Files that have been moved to tests/results
    result_files = [
        'afp_comparison.json',
        'afp_results.json',
        'baseline_results.json',
        'afp_ai_integration_results.json'
    ]
    
    # Check if files exist in both root and test directories
    duplicates = []
    
    # Check unit test files
    for file in unit_test_files:
        if os.path.exists(file) and os.path.exists(os.path.join('tests', 'unit', file)):
            duplicates.append((file, os.path.join('tests', 'unit')))
    
    # Check performance test files
    for file in performance_test_files:
        if os.path.exists(file) and os.path.exists(os.path.join('tests', 'performance', file)):
            duplicates.append((file, os.path.join('tests', 'performance')))
    
    # Check integration test files
    for file in integration_test_files:
        if os.path.exists(file) and os.path.exists(os.path.join('tests', 'integration', file)):
            duplicates.append((file, os.path.join('tests', 'integration')))
    
    # Check result files
    for file in result_files:
        if os.path.exists(file) and os.path.exists(os.path.join('tests', 'results', file)):
            duplicates.append((file, os.path.join('tests', 'results')))
    
    # Report duplicates
    if duplicates:
        print("The following files exist both in the root directory and in test directories:")
        for file, directory in duplicates:
            print(f"  - {file} (also in {directory})")
        
        if confirm("Do you want to remove these duplicate files from the root directory?"):
            for file, _ in duplicates:
                try:
                    os.remove(file)
                    print(f"Removed: {file}")
                except Exception as e:
                    print(f"Error removing {file}: {e}")
    else:
        print("No duplicate files found.")
    
    print_header("Cleanup Complete")
    print("The test directory structure is now organized as follows:")
    print(f"  - {os.path.join('tests', 'unit')}: Unit tests for individual AFP components")
    print(f"  - {os.path.join('tests', 'performance')}: Performance comparison tests")
    print(f"  - {os.path.join('tests', 'integration')}: Integration and example tests")
    print(f"  - {os.path.join('tests', 'results')}: Test result data files")
    
    print("\nYou can now run tests using the commands in the README.md file.")

if __name__ == "__main__":
    main() 