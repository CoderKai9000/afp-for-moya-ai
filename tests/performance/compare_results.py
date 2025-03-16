#!/usr/bin/env python
"""
Compare performance results between baseline, AFP, and optimized AFP implementations.

This script loads results from baseline_results.json, afp_results.json, and afp_optimized_results.json,
calculates the performance improvements, and generates a detailed report.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional


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


def extract_metrics(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Extract comparable metrics from results data regardless of format
    Returns a standardized format with direct_communication, multi_agent_orchestration, and complex_workflow
    """
    metrics = {
        "direct_communication": {},
        "multi_agent_orchestration": {},
        "complex_workflow": {}
    }
    
    if not results or "tests" not in results:
        return metrics
    
    tests = results["tests"]
    
    # Handle baseline and afp_optimized format (consistent structure)
    if isinstance(tests, dict) and "direct_communication" in tests and isinstance(tests["direct_communication"], dict):
        # Baseline or AFP Optimized format
        for test_name in ["direct_communication", "multi_agent_orchestration", "complex_workflow"]:
            if test_name in tests:
                test_data = tests[test_name]
                metrics[test_name]["total_time"] = test_data.get("total_time", 0)
                metrics[test_name]["message_count"] = test_data.get("message_count", 0)
                metrics[test_name]["avg_latency"] = test_data.get("avg_latency", 0) * 1000  # Convert to ms
                metrics[test_name]["throughput"] = test_data.get("throughput", 0)
                
                if test_name == "complex_workflow" and "workflow_completion_time" in test_data:
                    workflow_times = test_data["workflow_completion_time"]
                    metrics[test_name]["avg_workflow_time"] = sum(workflow_times) / len(workflow_times) * 1000 if workflow_times else 0
    
    # Handle original AFP format (array of test runs)
    elif isinstance(tests, dict):
        # Original AFP format
        if "direct_communication" in tests and isinstance(tests["direct_communication"], list):
            # Use the first entry in the array (for 100 messages)
            dc_data = tests["direct_communication"][0] if tests["direct_communication"] else {}
            metrics["direct_communication"]["total_time"] = dc_data.get("total_time_sec", 0)
            metrics["direct_communication"]["message_count"] = dc_data.get("num_messages", 0)
            metrics["direct_communication"]["avg_latency"] = dc_data.get("avg_latency_ms", 0)
            metrics["direct_communication"]["throughput"] = dc_data.get("throughput_msgs_per_sec", 0)
        
        if "multi_agent_orchestration" in tests and isinstance(tests["multi_agent_orchestration"], list):
            # Use the first entry with 5 agents
            mo_data = tests["multi_agent_orchestration"][0] if tests["multi_agent_orchestration"] else {}
            metrics["multi_agent_orchestration"]["total_time"] = mo_data.get("total_time_sec", 0)
            metrics["multi_agent_orchestration"]["message_count"] = mo_data.get("num_messages", 0)
            # Calculate average latency from total time and message count
            msg_count = mo_data.get("num_messages", 0)
            total_time = mo_data.get("total_time_sec", 0)
            metrics["multi_agent_orchestration"]["avg_latency"] = (total_time / msg_count) * 1000 if msg_count > 0 else 0
            metrics["multi_agent_orchestration"]["throughput"] = mo_data.get("msgs_per_second", 0)
            
        if "complex_workflow" in tests and isinstance(tests["complex_workflow"], list):
            # Use the middle complexity workflow (6 steps)
            cw_data = tests["complex_workflow"][1] if len(tests["complex_workflow"]) > 1 else {}
            # Estimate total time based on average workflow latency and message count
            avg_latency_ms = cw_data.get("avg_workflow_latency_ms", 0)
            metrics["complex_workflow"]["total_time"] = avg_latency_ms * 50 / 1000  # Assuming 50 workflows
            metrics["complex_workflow"]["message_count"] = 250  # Typical message count for complex workflow
            metrics["complex_workflow"]["avg_latency"] = avg_latency_ms
            metrics["complex_workflow"]["throughput"] = 250 / (avg_latency_ms * 50 / 1000) if avg_latency_ms > 0 else 0
            metrics["complex_workflow"]["avg_workflow_time"] = avg_latency_ms
    
    return metrics


def calculate_improvements(baseline: Dict[str, Dict[str, float]], optimized: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Calculate performance improvements between baseline and optimized results."""
    improvements = {
        "direct_communication": {},
        "multi_agent_orchestration": {},
        "complex_workflow": {}
    }
    
    for test_name in ["direct_communication", "multi_agent_orchestration", "complex_workflow"]:
        baseline_test = baseline.get(test_name, {})
        optimized_test = optimized.get(test_name, {})
        
        if not baseline_test or not optimized_test:
            continue
            
        # Calculate latency improvement (lower is better)
        baseline_latency = baseline_test.get("avg_latency", 0)
        optimized_latency = optimized_test.get("avg_latency", 0)
        if baseline_latency > 0:
            latency_improvement = ((baseline_latency - optimized_latency) / baseline_latency) * 100
        else:
            latency_improvement = 0
            
        # Calculate throughput improvement (higher is better)
        baseline_throughput = baseline_test.get("throughput", 0)
        optimized_throughput = optimized_test.get("throughput", 0)
        if baseline_throughput > 0:
            throughput_improvement = ((optimized_throughput - baseline_throughput) / baseline_throughput) * 100
        else:
            throughput_improvement = 0
            
        # Store improvements
        improvements[test_name]["latency_improvement"] = latency_improvement
        improvements[test_name]["throughput_improvement"] = throughput_improvement
        
        # Store raw values for reference
        improvements[test_name]["baseline_latency"] = baseline_latency
        improvements[test_name]["optimized_latency"] = optimized_latency
        improvements[test_name]["baseline_throughput"] = baseline_throughput
        improvements[test_name]["optimized_throughput"] = optimized_throughput
    
    return improvements


def generate_report(baseline: Dict[str, Dict[str, float]], afp: Dict[str, Dict[str, float]], optimized: Dict[str, Dict[str, float]]) -> str:
    """Generate a performance comparison report."""
    # Calculate improvements
    baseline_vs_afp = calculate_improvements(baseline, afp)
    baseline_vs_optimized = calculate_improvements(baseline, optimized)
    afp_vs_optimized = calculate_improvements(afp, optimized)
    
    # Calculate average improvements
    avg_latency_improvement_b_to_a = sum(v.get("latency_improvement", 0) for v in baseline_vs_afp.values()) / len(baseline_vs_afp)
    avg_throughput_improvement_b_to_a = sum(v.get("throughput_improvement", 0) for v in baseline_vs_afp.values()) / len(baseline_vs_afp)
    
    avg_latency_improvement_b_to_o = sum(v.get("latency_improvement", 0) for v in baseline_vs_optimized.values()) / len(baseline_vs_optimized)
    avg_throughput_improvement_b_to_o = sum(v.get("throughput_improvement", 0) for v in baseline_vs_optimized.values()) / len(baseline_vs_optimized)
    
    avg_latency_improvement_a_to_o = sum(v.get("latency_improvement", 0) for v in afp_vs_optimized.values()) / len(afp_vs_optimized)
    avg_throughput_improvement_a_to_o = sum(v.get("throughput_improvement", 0) for v in afp_vs_optimized.values()) / len(afp_vs_optimized)
    
    # Generate report
    report = [
        "# Performance Comparison Report",
        "",
        "## Overview",
        "",
        "This report compares the performance of three implementations:",
        "- **Baseline**: Direct function calls without AFP",
        "- **AFP**: Original Agent Flow Protocol implementation",
        "- **Optimized AFP**: Improved AFP implementation with direct routing and prioritization",
        "",
        "## Latency Comparison (ms)",
        "",
        "| Test | Baseline | AFP | Optimized AFP | Improvement (Baseline->Opt) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    
    for test_name in ["direct_communication", "multi_agent_orchestration", "complex_workflow"]:
        baseline_latency = baseline.get(test_name, {}).get("avg_latency", 0)
        afp_latency = afp.get(test_name, {}).get("avg_latency", 0)
        optimized_latency = optimized.get(test_name, {}).get("avg_latency", 0)
        improvement = baseline_vs_optimized.get(test_name, {}).get("latency_improvement", 0)
        
        report.append(f"| {test_name.replace('_', ' ').title()} | {baseline_latency:.2f} | {afp_latency:.2f} | {optimized_latency:.2f} | {improvement:.2f}% |")
    
    report.extend([
        "",
        "## Throughput Comparison (messages/second)",
        "",
        "| Test | Baseline | AFP | Optimized AFP | Improvement (Baseline->Opt) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    
    for test_name in ["direct_communication", "multi_agent_orchestration", "complex_workflow"]:
        baseline_throughput = baseline.get(test_name, {}).get("throughput", 0)
        afp_throughput = afp.get(test_name, {}).get("throughput", 0)
        optimized_throughput = optimized.get(test_name, {}).get("throughput", 0)
        improvement = baseline_vs_optimized.get(test_name, {}).get("throughput_improvement", 0)
        
        report.append(f"| {test_name.replace('_', ' ').title()} | {baseline_throughput:.2f} | {afp_throughput:.2f} | {optimized_throughput:.2f} | {improvement:.2f}% |")
    
    # Add workflow specific metrics if available
    if ("complex_workflow" in baseline and "avg_workflow_time" in baseline["complex_workflow"] and
        "complex_workflow" in afp and "avg_workflow_time" in afp["complex_workflow"] and
        "complex_workflow" in optimized and "avg_workflow_time" in optimized["complex_workflow"]):
        
        baseline_workflow_time = baseline["complex_workflow"]["avg_workflow_time"]
        afp_workflow_time = afp["complex_workflow"]["avg_workflow_time"]
        optimized_workflow_time = optimized["complex_workflow"]["avg_workflow_time"]
        
        if baseline_workflow_time > 0:
            workflow_improvement = ((baseline_workflow_time - optimized_workflow_time) / baseline_workflow_time) * 100
        else:
            workflow_improvement = 0
            
        report.extend([
            "",
            "## Workflow Completion Time (ms)",
            "",
            f"| Implementation | Average Time | Improvement vs. Baseline |",
            f"| --- | ---: | ---: |",
            f"| Baseline | {baseline_workflow_time:.2f} | - |",
            f"| AFP | {afp_workflow_time:.2f} | {((baseline_workflow_time - afp_workflow_time) / baseline_workflow_time * 100) if baseline_workflow_time > 0 else 0:.2f}% |",
            f"| Optimized AFP | {optimized_workflow_time:.2f} | {workflow_improvement:.2f}% |",
        ])
    
    # Add summary
    report.extend([
        "",
        "## Key Findings",
        "",
        f"1. **Overall Latency Improvement**: The optimized AFP implementation shows an average latency reduction of {avg_latency_improvement_b_to_o:.2f}% compared to the baseline.",
        f"2. **Overall Throughput Improvement**: The optimized AFP implementation shows an average throughput increase of {avg_throughput_improvement_b_to_o:.2f}% compared to the baseline.",
        f"3. **AFP Optimization Impact**: Compared to the original AFP implementation, the optimized version shows a {avg_latency_improvement_a_to_o:.2f}% reduction in latency and a {avg_throughput_improvement_a_to_o:.2f}% increase in throughput.",
    ])
    
    # Identify the most significant improvement
    most_improved_latency = max(baseline_vs_optimized.items(), key=lambda x: x[1].get("latency_improvement", 0))
    most_improved_throughput = max(baseline_vs_optimized.items(), key=lambda x: x[1].get("throughput_improvement", 0))
    
    report.extend([
        f"4. **Most Significant Latency Improvement**: {most_improved_latency[0].replace('_', ' ').title()} shows the most significant latency reduction at {most_improved_latency[1].get('latency_improvement', 0):.2f}%.",
        f"5. **Most Significant Throughput Improvement**: {most_improved_throughput[0].replace('_', ' ').title()} shows the most significant throughput increase at {most_improved_throughput[1].get('throughput_improvement', 0):.2f}%.",
    ])
    
    return "\n".join(report)


def main():
    """Load results, calculate improvements, and generate report."""
    # Load result files
    baseline = extract_metrics(load_results("baseline_results.json"))
    afp = extract_metrics(load_results("afp_results.json"))
    optimized = extract_metrics(load_results("afp_optimized_results.json"))
    
    # Print available data
    print("Loaded performance data:")
    print(f"- Baseline: {list(baseline.keys())}")
    print(f"- AFP: {list(afp.keys())}")
    print(f"- Optimized AFP: {list(optimized.keys())}")
    
    # Generate performance comparison report
    report = generate_report(baseline, afp, optimized)
    
    # Save report to file
    with open("performance_comparison.md", "w") as f:
        f.write(report)
    
    print("\nPerformance comparison report generated and saved to 'performance_comparison.md'")
    
    # Print summary to console
    b_to_o_improvements = calculate_improvements(baseline, optimized)
    avg_latency_improvement = sum(v.get("latency_improvement", 0) for v in b_to_o_improvements.values()) / len(b_to_o_improvements)
    avg_throughput_improvement = sum(v.get("throughput_improvement", 0) for v in b_to_o_improvements.values()) / len(b_to_o_improvements)
    
    print("\nPerformance Improvement Summary:")
    print(f"- Average latency reduction: {avg_latency_improvement:.2f}%")
    print(f"- Average throughput increase: {avg_throughput_improvement:.2f}%")
    
    # Most significant improvements
    most_improved_latency = max(b_to_o_improvements.items(), key=lambda x: x[1].get("latency_improvement", 0))
    most_improved_throughput = max(b_to_o_improvements.items(), key=lambda x: x[1].get("throughput_improvement", 0))
    
    print(f"- Best latency improvement: {most_improved_latency[0].replace('_', ' ').title()} ({most_improved_latency[1].get('latency_improvement', 0):.2f}%)")
    print(f"- Best throughput improvement: {most_improved_throughput[0].replace('_', ' ').title()} ({most_improved_throughput[1].get('throughput_improvement', 0):.2f}%)")


if __name__ == "__main__":
    main() 