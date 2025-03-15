import os

def main():
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
    
    print("Checking for duplicate files...")
    
    # Check unit test files
    for file in unit_test_files:
        root_exists = os.path.exists(file)
        test_exists = os.path.exists(os.path.join('tests', 'unit', file))
        print(f"{file}: Root={root_exists}, Test dir={test_exists}")
    
    # Check performance test files
    for file in performance_test_files:
        root_exists = os.path.exists(file)
        test_exists = os.path.exists(os.path.join('tests', 'performance', file))
        print(f"{file}: Root={root_exists}, Test dir={test_exists}")
    
    # Check integration test files
    for file in integration_test_files:
        root_exists = os.path.exists(file)
        test_exists = os.path.exists(os.path.join('tests', 'integration', file))
        print(f"{file}: Root={root_exists}, Test dir={test_exists}")
    
    # Check result files
    for file in result_files:
        root_exists = os.path.exists(file)
        test_exists = os.path.exists(os.path.join('tests', 'results', file))
        print(f"{file}: Root={root_exists}, Test dir={test_exists}")

if __name__ == "__main__":
    main() 