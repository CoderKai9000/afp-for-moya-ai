import json
import os
from typing import Dict, Any

def load_results(filename: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File {filename} not found. Returning empty data.")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: File {filename} contains invalid JSON. Returning empty data.")
        return {}

def calculate_improvements(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate performance improvements between baseline and optimized results."""
    improvements = {}
    
    for test_name in ["direct_communication", "multi_agent_orchestration", "complex_workflow"]:
        if test_name in baseline["tests"] and test_name in optimized["tests"]:
            baseline_test = baseline["tests"][test_name]
            optimized_test = optimized["tests"][test_name]
            
            # Latency improvement (lower is better)
            baseline_latency = baseline_test["avg_latency"] * 1000  # Convert to ms
            optimized_latency = optimized_test["avg_latency"] * 1000  # Convert to ms
            latency_improvement = ((baseline_latency - optimized_latency) / baseline_latency) * 100
            
            # Throughput improvement (higher is better)
            baseline_throughput = baseline_test["throughput"]
            optimized_throughput = optimized_test["throughput"]
            throughput_improvement = ((optimized_throughput - baseline_throughput) / baseline_throughput) * 100
            
            # Store improvements
            improvements[test_name] = {
                "baseline_latency_ms": baseline_latency,
                "optimized_latency_ms": optimized_latency,
                "latency_improvement_percent": latency_improvement,
                "baseline_throughput": baseline_throughput,
                "optimized_throughput": optimized_throughput,
                "throughput_improvement_percent": throughput_improvement
            }
            
            # Add workflow times if available
            if test_name == "complex_workflow" and "workflow_completion_time" in baseline_test and "workflow_completion_time" in optimized_test:
                baseline_workflow_times = baseline_test["workflow_completion_time"]
                optimized_workflow_times = optimized_test["workflow_completion_time"]
                
                avg_baseline_workflow_time = sum(baseline_workflow_times) / len(baseline_workflow_times) * 1000  # to ms
                avg_optimized_workflow_time = sum(optimized_workflow_times) / len(optimized_workflow_times) * 1000  # to ms
                
                workflow_improvement = ((avg_baseline_workflow_time - avg_optimized_workflow_time) / avg_baseline_workflow_time) * 100
                
                improvements[test_name]["avg_baseline_workflow_time_ms"] = avg_baseline_workflow_time
                improvements[test_name]["avg_optimized_workflow_time_ms"] = avg_optimized_workflow_time
                improvements[test_name]["workflow_time_improvement_percent"] = workflow_improvement
    
    return improvements

def print_comparison_table(improvements: Dict[str, Any]) -> None:
    """Print a formatted comparison table."""
    print("\n=== Performance Comparison: Baseline vs. Optimized AFP ===\n")
    
    # Print latency comparison
    print("LATENCY COMPARISON (ms, lower is better)")
    print("+--------------------------+------------+----------------+----------------+")
    print("| Test                     | Baseline   | Optimized AFP  | Improvement    |")
    print("+--------------------------+------------+----------------+----------------+")
    
    for test_name, data in improvements.items():
        test_label = test_name.replace("_", " ").title()
        baseline = data["baseline_latency_ms"]
        optimized = data["optimized_latency_ms"]
        improvement = data["latency_improvement_percent"]
        
        print(f"| {test_label:<24} | {baseline:>10.2f} | {optimized:>14.2f} | {improvement:>14.2f}% |")
    
    print("+--------------------------+------------+----------------+----------------+")
    
    # Print throughput comparison
    print("\nTHROUGHPUT COMPARISON (messages/second, higher is better)")
    print("+--------------------------+------------+----------------+----------------+")
    print("| Test                     | Baseline   | Optimized AFP  | Improvement    |")
    print("+--------------------------+------------+----------------+----------------+")
    
    for test_name, data in improvements.items():
        test_label = test_name.replace("_", " ").title()
        baseline = data["baseline_throughput"]
        optimized = data["optimized_throughput"]
        improvement = data["throughput_improvement_percent"]
        
        print(f"| {test_label:<24} | {baseline:>10.2f} | {optimized:>14.2f} | {improvement:>14.2f}% |")
    
    print("+--------------------------+------------+----------------+----------------+")
    
    # Print workflow time comparison if available
    if "complex_workflow" in improvements and "avg_baseline_workflow_time_ms" in improvements["complex_workflow"]:
        print("\nWORKFLOW COMPLETION TIME (ms, lower is better)")
        print("+--------------------------+------------+----------------+----------------+")
        print("| Test                     | Baseline   | Optimized AFP  | Improvement    |")
        print("+--------------------------+------------+----------------+----------------+")
        
        data = improvements["complex_workflow"]
        baseline = data["avg_baseline_workflow_time_ms"]
        optimized = data["avg_optimized_workflow_time_ms"]
        improvement = data["workflow_time_improvement_percent"]
        
        print(f"| Complex Workflow         | {baseline:>10.2f} | {optimized:>14.2f} | {improvement:>14.2f}% |")
        print("+--------------------------+------------+----------------+----------------+")
    
    # Print summary
    avg_latency_improvement = sum(data["latency_improvement_percent"] for data in improvements.values()) / len(improvements)
    avg_throughput_improvement = sum(data["throughput_improvement_percent"] for data in improvements.values()) / len(improvements)
    
    print("\nSUMMARY")
    print(f"Average latency reduction: {avg_latency_improvement:.2f}%")
    print(f"Average throughput increase: {avg_throughput_improvement:.2f}%")
    
    # Find best improvements
    best_latency = max(improvements.items(), key=lambda x: x[1]["latency_improvement_percent"])
    best_throughput = max(improvements.items(), key=lambda x: x[1]["throughput_improvement_percent"])
    
    print(f"Best latency improvement: {best_latency[0].replace('_', ' ').title()} ({best_latency[1]['latency_improvement_percent']:.2f}%)")
    print(f"Best throughput improvement: {best_throughput[0].replace('_', ' ').title()} ({best_throughput[1]['throughput_improvement_percent']:.2f}%)")

def main():
    """Load results, calculate improvements, and display comparison."""
    baseline = load_results("baseline_results.json")
    optimized = load_results("afp_optimized_results.json")
    
    if not baseline or not optimized:
        print("Failed to load result files.")
        return
    
    improvements = calculate_improvements(baseline, optimized)
    print_comparison_table(improvements)
    
    # Save improvements to file
    with open("direct_comparison_results.json", "w") as f:
        json.dump(improvements, f, indent=2)
    
    print("\nDetailed comparison saved to 'direct_comparison_results.json'")

if __name__ == "__main__":
    main() 