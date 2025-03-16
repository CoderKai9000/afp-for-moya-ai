#!/usr/bin/env python
"""
Performance comparison between non-AFP and AFP-based multi-agent implementations.

This script measures and compares the performance of two multi-agent implementations:
1. simple_multiagent.py - Standard implementation without AFP
2. afp_multiagent.py - Implementation using the Agent Flow Protocol (AFP)

Metrics measured include:
- Message processing time
- Agent communication overhead
- Total conversation processing time
"""

import os
import time
import json
import importlib.util
import statistics
from typing import Dict, List, Any, Optional, Callable
import platform
import gc
import threading
from concurrent.futures import ThreadPoolExecutor

# Function to import modules from file path
def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PerformanceTest:
    """Class to run performance tests on multi-agent systems."""
    
    def __init__(self):
        """Initialize performance test."""
        self.test_cases = [
            "Tell me a joke about programming",
            "What's the weather like today?",
            "Tell me about machine learning",
            "Translate 'Hello, how are you?' to Spanish",
            "Háblame sobre inteligencia artificial",
            "Tell me a joke about AI"
        ]
        
        # Performance metrics storage
        self.simple_metrics = {
            "response_times": [],
            "api_call_times": [],  # New field for tracking API call times
            "framework_overhead_times": [],  # New field for tracking framework overhead
            "total_time": 0,
            "total_api_time": 0  # New field for total API time
        }
        
        self.afp_metrics = {
            "response_times": [],
            "api_call_times": [],  # New field for tracking API call times
            "framework_overhead_times": [],  # New field for tracking framework overhead
            "total_time": 0,
            "total_api_time": 0  # New field for total API time
        }
    
    def run_simple_multiagent_test(self):
        """Run performance test on the simple multiagent implementation."""
        print("\n" + "="*50)
        print("Testing Simple Multi-Agent Implementation")
        print("="*50)
        
        # Import the simple_multiagent module
        simple_module = import_module_from_file("simple_multiagent", "simple_multiagent.py")
        
        # Patch the AzureOpenAIAgent.handle_message method to track API time
        original_handle_message = simple_module.AzureOpenAIAgent.handle_message
        
        # Track API call times
        api_call_times = []
        
        def patched_handle_message(self, message, **kwargs):
            # Start API timing
            api_start_time = time.time()
            
            # Call the original method
            result = original_handle_message(self, message, **kwargs)
            
            # End API timing
            api_end_time = time.time()
            api_call_time = api_end_time - api_start_time
            api_call_times.append(api_call_time)
            
            return result
        
        # Apply the patch
        simple_module.AzureOpenAIAgent.handle_message = patched_handle_message
        
        # Setup the orchestrator
        orchestrator = simple_module.setup_orchestrator()
        thread_id = "simple_test_thread"
        
        # Initialize memory
        simple_module.EphemeralMemory.store_message(thread_id=thread_id, sender="system", content=f"thread ID: {thread_id}")
        
        # Run the test cases
        start_total = time.time()
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\nTest Case {i+1}: {test_case}")
            
            # Reset API call times for this test
            api_call_times.clear()
            
            # Store the user message
            simple_module.EphemeralMemory.store_message(thread_id=thread_id, sender="user", content=test_case)
            
            # Record the time to process this message
            start_time = time.time()
            
            # Use a silent callback to avoid printing during tests
            def silent_callback(chunk):
                pass
            
            # Get response - capture it but don't print for benchmark
            response = orchestrator.orchestrate(
                thread_id=thread_id,
                user_message=test_case,
                stream_callback=silent_callback
            )
            
            # Measure response time
            end_time = time.time()
            response_time = end_time - start_time
            
            # Calculate API time for this test (might be multiple API calls)
            api_time = sum(api_call_times)
            
            # Calculate framework overhead (total time minus API time)
            framework_time = response_time - api_time
            
            # Store the metrics
            self.simple_metrics["response_times"].append(response_time)
            self.simple_metrics["api_call_times"].append(api_time)
            self.simple_metrics["framework_overhead_times"].append(framework_time)
            
            # Print detailed timing information
            print(f"Total response time: {response_time:.4f} seconds")
            print(f"API call time: {api_time:.4f} seconds")
            print(f"Framework overhead: {framework_time:.4f} seconds")
        
        # Calculate total time
        end_total = time.time()
        self.simple_metrics["total_time"] = end_total - start_total
        self.simple_metrics["total_api_time"] = sum(self.simple_metrics["api_call_times"])
        
        # Restore the original method
        simple_module.AzureOpenAIAgent.handle_message = original_handle_message
        
        # Just run garbage collection to clean up resources
        gc.collect()
    
    def run_afp_multiagent_test(self):
        """Run performance test on the AFP-based multiagent implementation."""
        print("\n" + "="*50)
        print("Testing AFP Multi-Agent Implementation")
        print("="*50)
        
        # Import the afp_multiagent module
        afp_module = import_module_from_file("afp_multiagent", "afp_multiagent.py")
        
        # Patch the AzureOpenAIAgent.handle_message method to track API time
        original_handle_message = afp_module.AzureOpenAIAgent.handle_message
        
        # Track API call times
        api_call_times = []
        
        def patched_handle_message(self, message, **kwargs):
            # Start API timing
            api_start_time = time.time()
            
            # Call the original method
            result = original_handle_message(self, message, **kwargs)
            
            # End API timing
            api_end_time = time.time()
            api_call_time = api_end_time - api_start_time
            api_call_times.append(api_call_time)
            
            return result
        
        # Apply the patch
        afp_module.AzureOpenAIAgent.handle_message = patched_handle_message
        
        # Setup the orchestrator
        orchestrator = afp_module.setup_afp_orchestrator()
        thread_id = "afp_test_thread"
        
        # Initialize memory
        afp_module.EphemeralMemory.store_message(thread_id=thread_id, sender="system", content=f"thread ID: {thread_id}")
        
        # Run the test cases
        start_total = time.time()
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\nTest Case {i+1}: {test_case}")
            
            # Reset API call times for this test
            api_call_times.clear()
            
            # Store the user message
            afp_module.EphemeralMemory.store_message(thread_id=thread_id, sender="user", content=test_case)
            
            # Record the time to process this message
            start_time = time.time()
            
            # Use a silent callback to avoid printing during tests
            def silent_callback(chunk):
                pass
            
            # Get response - capture it but don't print for benchmark
            response = orchestrator.orchestrate(
                thread_id=thread_id,
                user_message=test_case,
                stream_callback=None
            )
            
            # Measure response time
            end_time = time.time()
            response_time = end_time - start_time
            
            # Calculate API time for this test (might be multiple API calls)
            api_time = sum(api_call_times)
            
            # Calculate framework overhead (total time minus API time)
            framework_time = response_time - api_time
            
            # Store the metrics
            self.afp_metrics["response_times"].append(response_time)
            self.afp_metrics["api_call_times"].append(api_time)
            self.afp_metrics["framework_overhead_times"].append(framework_time)
            
            # Print detailed timing information
            print(f"Total response time: {response_time:.4f} seconds")
            print(f"API call time: {api_time:.4f} seconds")
            print(f"Framework overhead: {framework_time:.4f} seconds")
            
            # Also print the performance message from the AFP orchestrator
            print(f"[Performance] Message processed in {response_time:.4f} seconds by agent")
        
        # Calculate total time
        end_total = time.time()
        self.afp_metrics["total_time"] = end_total - start_total
        self.afp_metrics["total_api_time"] = sum(self.afp_metrics["api_call_times"])
        
        # Restore the original method
        afp_module.AzureOpenAIAgent.handle_message = original_handle_message
        
        # Just run garbage collection to clean up resources
        gc.collect()
    
    def run_concurrent_load_test(self, num_concurrent=3, num_messages=2):
        """
        Run concurrent load tests to measure system performance under load.
        
        Args:
            num_concurrent: Number of concurrent users/threads
            num_messages: Number of messages per user/thread
        """
        print("\n" + "="*50)
        print(f"Running Concurrent Load Test ({num_concurrent} users, {num_messages} messages each)")
        print("="*50)
        
        # Import the modules
        simple_module = import_module_from_file("simple_multiagent", "simple_multiagent.py")
        afp_module = import_module_from_file("afp_multiagent", "afp_multiagent.py")
        
        # Setup orchestrators for both implementations
        simple_orchestrator = simple_module.setup_orchestrator()
        afp_orchestrator = afp_module.setup_afp_orchestrator()
        
        # Metrics for concurrent tests
        simple_concurrent_times = []
        afp_concurrent_times = []
        
        # Create test cases for concurrent testing - ensure we have enough test cases
        total_messages_needed = num_concurrent * num_messages
        concurrent_tests = []
        
        # Generate enough test cases by repeating and extending the basic test cases
        while len(concurrent_tests) < total_messages_needed:
            for test in self.test_cases:
                concurrent_tests.append(test)
                if len(concurrent_tests) >= total_messages_needed:
                    break
                    
        # Truncate to exactly the number we need
        concurrent_tests = concurrent_tests[:total_messages_needed]
        
        # Function to process a message using the simple implementation
        def process_simple_message(thread_id, message):
            simple_module.EphemeralMemory.store_message(thread_id=thread_id, sender="system", content=f"thread ID: {thread_id}")
            simple_module.EphemeralMemory.store_message(thread_id=thread_id, sender="user", content=message)
            
            start_time = time.time()
            
            def silent_callback(chunk):
                pass
            
            response = simple_orchestrator.orchestrate(
                thread_id=thread_id,
                user_message=message,
                stream_callback=silent_callback
            )
            
            end_time = time.time()
            return end_time - start_time
        
        # Function to process a message using the AFP implementation
        def process_afp_message(thread_id, message):
            afp_module.EphemeralMemory.store_message(thread_id=thread_id, sender="system", content=f"thread ID: {thread_id}")
            afp_module.EphemeralMemory.store_message(thread_id=thread_id, sender="user", content=message)
            
            start_time = time.time()
            
            def silent_callback(chunk):
                pass
            
            response = afp_orchestrator.orchestrate(
                thread_id=thread_id,
                user_message=message,
                stream_callback=silent_callback
            )
            
            end_time = time.time()
            return end_time - start_time
        
        # Test simple implementation with concurrent users
        print("\nTesting Simple Multi-Agent with concurrent users...")
        
        start_total = time.time()
        
        # Create thread pool
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            # Submit tasks for each user/message
            futures = []
            msg_index = 0
            
            for user_id in range(num_concurrent):
                for msg_idx in range(num_messages):
                    thread_id = f"simple_user_{user_id}_{msg_idx}"
                    message = concurrent_tests[msg_index]
                    msg_index += 1
                    
                    future = executor.submit(process_simple_message, thread_id, message)
                    futures.append(future)
            
            # Collect results
            for future in futures:
                simple_concurrent_times.append(future.result())
        
        end_total = time.time()
        simple_concurrent_total = end_total - start_total
        
        # Just run garbage collection
        gc.collect()
        
        # Test AFP implementation with concurrent users
        print("\nTesting AFP Multi-Agent with concurrent users...")
        
        start_total = time.time()
        
        # Create thread pool
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            # Submit tasks for each user/message
            futures = []
            msg_index = 0
            
            for user_id in range(num_concurrent):
                for msg_idx in range(num_messages):
                    thread_id = f"afp_user_{user_id}_{msg_idx}"
                    message = concurrent_tests[msg_index]
                    msg_index += 1
                    
                    future = executor.submit(process_afp_message, thread_id, message)
                    futures.append(future)
            
            # Collect results
            for future in futures:
                afp_concurrent_times.append(future.result())
        
        end_total = time.time()
        afp_concurrent_total = end_total - start_total
        
        # Just run garbage collection
        gc.collect()
        
        # Return concurrent test results
        return {
            "simple": {
                "individual_times": simple_concurrent_times,
                "avg_response_time": statistics.mean(simple_concurrent_times),
                "total_time": simple_concurrent_total,
                "throughput": (num_concurrent * num_messages) / simple_concurrent_total
            },
            "afp": {
                "individual_times": afp_concurrent_times,
                "avg_response_time": statistics.mean(afp_concurrent_times),
                "total_time": afp_concurrent_total,
                "throughput": (num_concurrent * num_messages) / afp_concurrent_total
            }
        }
    
    def generate_report(self, concurrent_results=None):
        """Generate a performance comparison report."""
        print("\n" + "="*50)
        print("Performance Comparison Report")
        print("="*50)
        
        # Calculate metrics with API time included
        simple_avg_response = statistics.mean(self.simple_metrics["response_times"])
        afp_avg_response = statistics.mean(self.afp_metrics["response_times"])
        
        simple_throughput = len(self.test_cases) / self.simple_metrics["total_time"]
        afp_throughput = len(self.test_cases) / self.afp_metrics["total_time"]
        
        # Calculate metrics without API time (framework overhead only)
        simple_avg_overhead = statistics.mean(self.simple_metrics["framework_overhead_times"])
        afp_avg_overhead = statistics.mean(self.afp_metrics["framework_overhead_times"])
        
        simple_overhead_throughput = len(self.test_cases) / (self.simple_metrics["total_time"] - self.simple_metrics["total_api_time"])
        afp_overhead_throughput = len(self.test_cases) / (self.afp_metrics["total_time"] - self.afp_metrics["total_api_time"])
        
        # Calculate improvements with API time included
        response_improvement = ((simple_avg_response - afp_avg_response) / simple_avg_response) * 100
        throughput_improvement = ((afp_throughput - simple_throughput) / simple_throughput) * 100
        
        # Calculate improvements without API time (framework overhead only)
        overhead_improvement = ((simple_avg_overhead - afp_avg_overhead) / simple_avg_overhead) * 100
        overhead_throughput_improvement = ((afp_overhead_throughput - simple_overhead_throughput) / simple_overhead_throughput) * 100
        
        # Print sequential test results with API time included
        print("\nSequential Test Results (INCLUDING API Time):")
        print(f"Simple MultiAgent - Avg Response Time: {simple_avg_response:.4f} seconds")
        print(f"AFP MultiAgent - Avg Response Time: {afp_avg_response:.4f} seconds")
        print(f"Response Time Improvement: {response_improvement:.2f}%")
        
        print(f"\nSimple MultiAgent - Throughput: {simple_throughput:.2f} messages/second")
        print(f"AFP MultiAgent - Throughput: {afp_throughput:.2f} messages/second")
        print(f"Throughput Improvement: {throughput_improvement:.2f}%")
        
        # Print sequential test results without API time (framework overhead only)
        print("\nSequential Test Results (EXCLUDING API Time - Framework Overhead Only):")
        print(f"Simple MultiAgent - Avg Framework Overhead: {simple_avg_overhead:.4f} seconds")
        print(f"AFP MultiAgent - Avg Framework Overhead: {afp_avg_overhead:.4f} seconds")
        print(f"Framework Overhead Improvement: {overhead_improvement:.2f}%")
        
        print(f"\nSimple MultiAgent - Framework Throughput: {simple_overhead_throughput:.2f} messages/second")
        print(f"AFP MultiAgent - Framework Throughput: {afp_overhead_throughput:.2f} messages/second")
        print(f"Framework Throughput Improvement: {overhead_throughput_improvement:.2f}%")
        
        # Print concurrent test results if available
        if concurrent_results:
            simple_concurrent = concurrent_results["simple"]
            afp_concurrent = concurrent_results["afp"]
            
            concurrent_response_improvement = ((simple_concurrent["avg_response_time"] - afp_concurrent["avg_response_time"]) / simple_concurrent["avg_response_time"]) * 100
            concurrent_throughput_improvement = ((afp_concurrent["throughput"] - simple_concurrent["throughput"]) / simple_concurrent["throughput"]) * 100
            
            print("\nConcurrent Test Results:")
            print(f"Simple MultiAgent - Avg Response Time: {simple_concurrent['avg_response_time']:.4f} seconds")
            print(f"AFP MultiAgent - Avg Response Time: {afp_concurrent['avg_response_time']:.4f} seconds")
            print(f"Response Time Improvement: {concurrent_response_improvement:.2f}%")
            
            print(f"\nSimple MultiAgent - Throughput: {simple_concurrent['throughput']:.2f} messages/second")
            print(f"AFP MultiAgent - Throughput: {afp_concurrent['throughput']:.2f} messages/second")
            print(f"Throughput Improvement: {concurrent_throughput_improvement:.2f}%")
        
        # Save results to file
        results = {
            "system_info": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "sequential_tests": {
                "with_api_time": {
                    "simple": {
                        "response_times": self.simple_metrics["response_times"],
                        "avg_response_time": simple_avg_response,
                        "total_time": self.simple_metrics["total_time"],
                        "throughput": simple_throughput
                    },
                    "afp": {
                        "response_times": self.afp_metrics["response_times"],
                        "avg_response_time": afp_avg_response,
                        "total_time": self.afp_metrics["total_time"],
                        "throughput": afp_throughput
                    },
                    "improvements": {
                        "response_time_percent": response_improvement,
                        "throughput_percent": throughput_improvement
                    }
                },
                "framework_overhead_only": {
                    "simple": {
                        "overhead_times": self.simple_metrics["framework_overhead_times"],
                        "avg_overhead_time": simple_avg_overhead,
                        "total_overhead_time": self.simple_metrics["total_time"] - self.simple_metrics["total_api_time"],
                        "overhead_throughput": simple_overhead_throughput
                    },
                    "afp": {
                        "overhead_times": self.afp_metrics["framework_overhead_times"],
                        "avg_overhead_time": afp_avg_overhead,
                        "total_overhead_time": self.afp_metrics["total_time"] - self.afp_metrics["total_api_time"],
                        "overhead_throughput": afp_overhead_throughput
                    },
                    "improvements": {
                        "overhead_time_percent": overhead_improvement,
                        "overhead_throughput_percent": overhead_throughput_improvement
                    }
                }
            }
        }
        
        if concurrent_results:
            results["concurrent_tests"] = concurrent_results
            results["concurrent_tests"]["improvements"] = {
                "response_time_percent": concurrent_response_improvement,
                "throughput_percent": concurrent_throughput_improvement
            }
        
        # Save to JSON file
        with open("multiagent_performance_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to 'multiagent_performance_results.json'")


def main():
    """Run the performance comparison tests."""
    # Create performance test instance
    tester = PerformanceTest()
    
    # Run sequential tests
    tester.run_simple_multiagent_test()
    tester.run_afp_multiagent_test()
    
    # Skip concurrent load test for now
    # concurrent_results = tester.run_concurrent_load_test(num_concurrent=3, num_messages=2)
    
    # Generate and print report without concurrent results
    tester.generate_report(concurrent_results=None)


if __name__ == "__main__":
    main() 