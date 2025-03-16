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
- Framework overhead
- API call times
"""

import os
import time
import json
import importlib.util
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
import platform
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from datetime import datetime
import pathlib
import tempfile
import shutil

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
            "What are the differences between classical and quantum computing?",
            "Explain the concept of machine learning in simple terms",
            "What are the top 5 programming languages in 2023 and why?",
            "How do neural networks function in deep learning?",
            "Explain the principles of blockchain technology",
            "What are the environmental implications of cryptocurrency mining?",
            "Summarize the history and impact of the internet",
            "What are the ethical considerations in artificial intelligence?",
            "How has remote work changed during and after the pandemic?",
            "Describe the process of photosynthesis in plants",
            "What advances have been made in renewable energy in the last decade?",
            "Compare and contrast different database management systems",
            "Explain how GPS navigation works",
            "What are the key differences between REST and GraphQL APIs?",
            "How do self-driving cars perceive and navigate their environment?"
        ]
        
        # Performance metrics storage
        self.simple_metrics = {
            "response_times": [],
            "api_call_times": [],  # Track API call times
            "framework_overhead_times": [],  # Track framework overhead
            "total_time": 0,
            "total_api_time": 0,  # Total API time
            "agent_interactions": [],  # Track which agents were involved in each response
            "message_counts": [],  # Number of messages exchanged for each query
            "agent_processing_times": {},  # Processing time per agent
            "classification_times": []  # Time spent on classification
        }
        
        self.afp_metrics = {
            "response_times": [],
            "api_call_times": [],  # Track API call times
            "framework_overhead_times": [],  # Track framework overhead
            "total_time": 0,
            "total_api_time": 0,  # Total API time
            "agent_interactions": [],  # Track which agents were involved in each response
            "message_counts": [],  # Number of messages exchanged for each query
            "agent_processing_times": {},  # Processing time per agent
            "classification_times": [],  # Time spent on classification
            "bus_overhead_times": []  # Time spent on AFP message bus communication
        }
    
    def run_simple_multiagent_test(self):
        """Run performance test on the simple multiagent implementation."""
        print("\n" + "="*50)
        print("Testing Simple Multi-Agent Implementation")
        print("="*50)
        
        # Import the simple_multiagent module
        simple_module = import_module_from_file("simple_multiagent", "simple_multiagent.py")
        
        # Track agent interactions and message counts
        agent_interactions = []
        message_counts = []
        classification_times = []
        agent_processing_times = {}
        
        # Patch the orchestrator's orchestrate method to track agent interactions
        original_orchestrate = simple_module.MultiAgentOrchestrator.orchestrate
        
        def patched_orchestrate(self, thread_id, user_message, stream_callback=None):
            # Clear the current interactions for this request
            current_interactions = []
            agent_name = None
            
            # Start classification timing
            classification_start = time.time()
            
            # Use the classifier to determine which agent to use
            if self.classifier:
                agent_name = self.classifier.classify(user_message)
            
            classification_end = time.time()
            classification_times.append(classification_end - classification_start)
            
            # Get the selected agent
            if agent_name and agent_name in self.agent_registry.agents:
                agent = self.agent_registry.get_agent(agent_name)
                current_interactions.append(agent_name)
            elif self.default_agent_name and self.default_agent_name in self.agent_registry.agents:
                agent = self.agent_registry.get_agent(self.default_agent_name)
                current_interactions.append(self.default_agent_name)
            else:
                # Get the first available agent
                available_agents = self.agent_registry.list_agents()
                if available_agents:
                    agent = available_agents[0]
                    current_interactions.append(agent.name)
                else:
                    raise ValueError("No agents available")
            
            # Add to interactions list
            agent_interactions.append(current_interactions)
            
            # Add message count (for simple orchestrator, it's always 1 message)
            message_counts.append(1)
            
            # Call the original orchestrate method
            return original_orchestrate(self, thread_id, user_message, stream_callback)
        
        # Apply the patch
        simple_module.MultiAgentOrchestrator.orchestrate = patched_orchestrate
        
        # Patch the AzureOpenAIAgent.handle_message method to track API time and agent processing time
        original_handle_message = simple_module.AzureOpenAIAgent.handle_message
        
        # Track API call times
        api_call_times = []
        
        def patched_handle_message(self, message, **kwargs):
            # Get agent name safely
            if hasattr(self, 'config') and isinstance(self.config, dict) and 'agent_name' in self.config:
                agent_name = self.config['agent_name']
            elif hasattr(self, 'config') and hasattr(self.config, 'agent_name'):
                agent_name = self.config.agent_name
            elif hasattr(self, 'name'):
                agent_name = self.name
            else:
                agent_name = "unknown"
                
            if agent_name not in agent_processing_times:
                agent_processing_times[agent_name] = []
            
            # Start agent timing
            agent_start_time = time.time()
            
            # Start API timing
            api_start_time = time.time()
            
            # Call the original method
            result = original_handle_message(self, message, **kwargs)
            
            # End API timing
            api_end_time = time.time()
            api_call_time = api_end_time - api_start_time
            api_call_times.append(api_call_time)
            
            # End agent timing
            agent_end_time = time.time()
            agent_processing_time = agent_end_time - agent_start_time
            agent_processing_times[agent_name].append(agent_processing_time)
            
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
            print(f"Agent used: {agent_interactions[-1][0] if agent_interactions else 'unknown'}")
        
        # Calculate total time
        end_total = time.time()
        self.simple_metrics["total_time"] = end_total - start_total
        self.simple_metrics["total_api_time"] = sum(self.simple_metrics["api_call_times"])
        
        # Store collected metrics
        self.simple_metrics["agent_interactions"] = agent_interactions
        self.simple_metrics["message_counts"] = message_counts
        self.simple_metrics["classification_times"] = classification_times
        self.simple_metrics["agent_processing_times"] = agent_processing_times
        
        # Restore the original methods
        simple_module.AzureOpenAIAgent.handle_message = original_handle_message
        simple_module.MultiAgentOrchestrator.orchestrate = original_orchestrate
        
        # Just run garbage collection to clean up resources
        gc.collect()
    
    def run_afp_multiagent_test(self):
        """Run performance test on the AFP-based multiagent implementation."""
        print("\n" + "="*50)
        print("Testing AFP Multi-Agent Implementation")
        print("="*50)
        
        # Import the afp_multiagent module
        afp_module = import_module_from_file("afp_multiagent", "afp_multiagent.py")
        
        # Track agent interactions, message counts, and classification times
        agent_interactions = []
        message_counts = []
        classification_times = []
        agent_processing_times = {}
        bus_overhead_times = []
        
        # Patch the AFPCommunicationBus.send_message method to track message flow
        original_send_message = afp_module.AFPCommunicationBus.send_message
        
        def patched_send_message(self, message):
            # Track the time spent on message bus communication
            bus_start_time = time.time()
            result = original_send_message(self, message)
            bus_end_time = time.time()
            bus_overhead_times.append(bus_end_time - bus_start_time)
            return result
            
        # Apply the patch
        afp_module.AFPCommunicationBus.send_message = patched_send_message
        
        # Patch the orchestrate method to track metrics
        original_orchestrate = afp_module.AFPOrchestrator.orchestrate
        
        def patched_orchestrate(self, thread_id, user_message, stream_callback=None):
            # Track metrics for this test case
            current_interactions = []
            current_message_count = 0
            
            # Start classification timing
            classification_start = time.time()
            
            # Before the call, we'll setup a callback to collect agent interactions
            def message_callback(message):
                nonlocal current_message_count, current_interactions
                current_message_count += 1
                
                # Add sender to interactions if not already included
                sender = message.sender
                if sender not in current_interactions and sender != "orchestrator" and sender != "classifier":
                    current_interactions.append(sender)
                    
                # Also track recipients that are agents
                for recipient in message.recipients:
                    if recipient not in current_interactions and recipient != "orchestrator" and recipient != "classifier":
                        current_interactions.append(recipient)
            
            # Register the message_tracer agent with the bus
            try:
                self.bus.register_agent("message_tracer")
            except Exception as e:
                print(f"Warning: Could not register message_tracer agent: {e}")
            
            # Register the callback for message tracing
            try:
                callback_id = self.bus.subscribe(
                    subscriber="message_tracer",
                    callback=message_callback,
                    pattern={}  # Match all messages
                )
            except Exception as e:
                print(f"Warning: Could not subscribe message_tracer: {e}")
                callback_id = None
            
            # Call classification
            if self.classifier:
                agent_name = self.classifier.classify(user_message, thread_id)
                
            classification_end = time.time()
            classification_times.append(classification_end - classification_start)
            
            # Call the original orchestrate method
            result = original_orchestrate(self, thread_id, user_message, stream_callback)
            
            # Unsubscribe the callback
            if callback_id:
                self.bus.unsubscribe(callback_id)
            
            # Add to metrics
            agent_interactions.append(current_interactions)
            message_counts.append(current_message_count)
            
            return result
            
        # Apply the patch
        afp_module.AFPOrchestrator.orchestrate = patched_orchestrate
        
        # Patch the AzureOpenAIAgent.handle_message method to track API time and agent processing
        original_handle_message = afp_module.AzureOpenAIAgent.handle_message
        
        # Track API call times
        api_call_times = []
        
        def patched_handle_message(self, message, **kwargs):
            # Get agent name safely
            if hasattr(self, 'config') and isinstance(self.config, dict) and 'agent_name' in self.config:
                agent_name = self.config['agent_name']
            elif hasattr(self, 'config') and hasattr(self.config, 'agent_name'):
                agent_name = self.config.agent_name
            elif hasattr(self, 'name'):
                agent_name = self.name
            else:
                agent_name = "unknown"
                
            if agent_name not in agent_processing_times:
                agent_processing_times[agent_name] = []
            
            # Start agent timing
            agent_start_time = time.time()
            
            # Start API timing
            api_start_time = time.time()
            
            # Call the original method
            result = original_handle_message(self, message, **kwargs)
            
            # End API timing
            api_end_time = time.time()
            api_call_time = api_end_time - api_start_time
            api_call_times.append(api_call_time)
            
            # End agent timing
            agent_end_time = time.time()
            agent_processing_time = agent_end_time - agent_start_time
            agent_processing_times[agent_name].append(agent_processing_time)
            
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
            print(f"Message bus overhead: {sum(bus_overhead_times):.4f} seconds")
            print(f"Agents involved: {', '.join(agent_interactions[-1])}")
            print(f"Messages exchanged: {message_counts[-1]}")
            
            # Clear the bus overhead times for the next test
            bus_overhead_times.clear()
        
        # Calculate total time
        end_total = time.time()
        self.afp_metrics["total_time"] = end_total - start_total
        self.afp_metrics["total_api_time"] = sum(self.afp_metrics["api_call_times"])
        
        # Store collected metrics
        self.afp_metrics["agent_interactions"] = agent_interactions
        self.afp_metrics["message_counts"] = message_counts
        self.afp_metrics["classification_times"] = classification_times
        self.afp_metrics["agent_processing_times"] = agent_processing_times
        self.afp_metrics["bus_overhead_times"] = bus_overhead_times
        
        # Restore the original methods
        afp_module.AzureOpenAIAgent.handle_message = original_handle_message
        afp_module.AFPOrchestrator.orchestrate = original_orchestrate
        afp_module.AFPCommunicationBus.send_message = original_send_message
        
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

    def generate_visualization(self, output_dir="visualizations"):
        """Generate visualizations for the performance metrics."""
        # Create a temporary directory for visualizations
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory for visualizations: {temp_dir}")
        
        # Create visualizations in the temporary directory
        self._create_response_time_comparison(temp_dir)
        self._create_framework_overhead_comparison(temp_dir)
        self._create_message_count_comparison(temp_dir)
        self._create_agent_processing_time_comparison(temp_dir)
        self._create_classification_time_comparison(temp_dir)
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy files from temporary directory to output directory
        for filename in os.listdir(temp_dir):
            src_file = os.path.join(temp_dir, filename)
            dst_file = os.path.join(output_dir, filename)
            try:
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"Warning: Could not copy {filename} to {output_dir}: {e}")
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temporary directory {temp_dir}: {e}")
        
        print(f"\nVisualizations created at {output_dir}")
    
    def _create_response_time_comparison(self, output_dir):
        """Create a bar chart comparing response times."""
        # Extract data
        simple_times = []
        afp_times = []
        
        # Check if metrics are available and in the expected format
        if hasattr(self, 'response_times_simple') and isinstance(self.response_times_simple, list):
            simple_times = self.response_times_simple
        elif hasattr(self, 'simple_metrics') and isinstance(self.simple_metrics, dict) and "response_times" in self.simple_metrics:
            simple_times = self.simple_metrics["response_times"]
        
        if hasattr(self, 'response_times_afp') and isinstance(self.response_times_afp, list):
            afp_times = self.response_times_afp
        elif hasattr(self, 'afp_metrics') and isinstance(self.afp_metrics, dict) and "response_times" in self.afp_metrics:
            afp_times = self.afp_metrics["response_times"]
        
        # Ensure we have data to plot
        if not simple_times and not afp_times:
            print("Warning: No response time data available for visualization")
            return
        
        # Ensure both lists have the same length
        max_len = max(len(simple_times), len(afp_times))
        simple_times = simple_times + [0] * (max_len - len(simple_times))
        afp_times = afp_times + [0] * (max_len - len(afp_times))
        
        # Create x-axis labels
        x = np.arange(len(simple_times))
        width = 0.35
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bars
        ax.bar(x - width/2, simple_times, width, label='Standard MultiAgent')
        ax.bar(x + width/2, afp_times, width, label='AFP MultiAgent')
        
        # Add labels and title
        ax.set_xlabel('Test Case')
        ax.set_ylabel('Response Time (seconds)')
        ax.set_title('Response Time Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Case {i+1}' for i in range(len(simple_times))])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'response_time_comparison.png'))
        plt.close()
    
    def _create_framework_overhead_comparison(self, output_dir):
        """Create a bar chart comparing framework overhead times."""
        # Extract data
        simple_times = []
        afp_times = []
        
        # Check if metrics are available and in the expected format
        if hasattr(self, 'framework_overhead_simple') and isinstance(self.framework_overhead_simple, list):
            simple_times = self.framework_overhead_simple
        elif hasattr(self, 'simple_metrics') and isinstance(self.simple_metrics, dict) and "framework_overhead_times" in self.simple_metrics:
            simple_times = self.simple_metrics["framework_overhead_times"]
        
        if hasattr(self, 'framework_overhead_afp') and isinstance(self.framework_overhead_afp, list):
            afp_times = self.framework_overhead_afp
        elif hasattr(self, 'afp_metrics') and isinstance(self.afp_metrics, dict) and "framework_overhead_times" in self.afp_metrics:
            afp_times = self.afp_metrics["framework_overhead_times"]
        
        # Ensure we have data to plot
        if not simple_times and not afp_times:
            print("Warning: No framework overhead data available for visualization")
            return
        
        # Ensure both lists have the same length
        max_len = max(len(simple_times), len(afp_times))
        simple_times = simple_times + [0] * (max_len - len(simple_times))
        afp_times = afp_times + [0] * (max_len - len(afp_times))
        
        # Create x-axis labels
        x = np.arange(len(simple_times))
        width = 0.35
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bars
        ax.bar(x - width/2, simple_times, width, label='Standard MultiAgent')
        ax.bar(x + width/2, afp_times, width, label='AFP MultiAgent')
        
        # Add labels and title
        ax.set_xlabel('Test Case')
        ax.set_ylabel('Framework Overhead (seconds)')
        ax.set_title('Framework Overhead Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Case {i+1}' for i in range(len(simple_times))])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'framework_overhead_comparison.png'))
        plt.close()
    
    def _create_message_count_comparison(self, output_dir):
        """Create a bar chart comparing message counts."""
        # Extract data
        simple_counts = []
        afp_counts = []
        
        # Check if metrics are available and in the expected format
        if hasattr(self, 'message_counts_simple') and isinstance(self.message_counts_simple, list):
            simple_counts = self.message_counts_simple
        elif hasattr(self, 'simple_metrics') and isinstance(self.simple_metrics, dict) and "message_counts" in self.simple_metrics:
            simple_counts = self.simple_metrics["message_counts"]
        
        if hasattr(self, 'message_counts_afp') and isinstance(self.message_counts_afp, list):
            afp_counts = self.message_counts_afp
        elif hasattr(self, 'afp_metrics') and isinstance(self.afp_metrics, dict) and "message_counts" in self.afp_metrics:
            afp_counts = self.afp_metrics["message_counts"]
        
        # Ensure we have data to plot
        if not simple_counts and not afp_counts:
            print("Warning: No message count data available for visualization")
            return
        
        # Ensure both lists have the same length
        max_len = max(len(simple_counts), len(afp_counts))
        simple_counts = simple_counts + [0] * (max_len - len(simple_counts))
        afp_counts = afp_counts + [0] * (max_len - len(afp_counts))
        
        # Create x-axis labels
        x = np.arange(len(simple_counts))
        width = 0.35
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bars
        ax.bar(x - width/2, simple_counts, width, label='Standard MultiAgent')
        ax.bar(x + width/2, afp_counts, width, label='AFP MultiAgent')
        
        # Add labels and title
        ax.set_xlabel('Test Case')
        ax.set_ylabel('Number of Messages')
        ax.set_title('Message Count Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Case {i+1}' for i in range(len(simple_counts))])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'message_count_comparison.png'))
        plt.close()
    
    def _create_agent_processing_time_comparison(self, output_dir):
        """Create a bar chart comparing agent processing times."""
        # Extract data
        simple_agents_data = {}
        afp_agents_data = {}
        
        # Check if metrics are available and in the expected format
        if hasattr(self, 'agent_processing_times_simple') and isinstance(self.agent_processing_times_simple, dict):
            simple_agents_data = self.agent_processing_times_simple
        elif hasattr(self, 'simple_metrics') and isinstance(self.simple_metrics, dict) and "agent_processing_times" in self.simple_metrics:
            simple_agents_data = self.simple_metrics["agent_processing_times"]
        
        if hasattr(self, 'agent_processing_times_afp') and isinstance(self.agent_processing_times_afp, dict):
            afp_agents_data = self.agent_processing_times_afp
        elif hasattr(self, 'afp_metrics') and isinstance(self.afp_metrics, dict) and "agent_processing_times" in self.afp_metrics:
            afp_agents_data = self.afp_metrics["agent_processing_times"]
        
        # Ensure we have data to plot
        if not simple_agents_data and not afp_agents_data:
            print("Warning: No agent processing time data available for visualization")
            return
        
        # Collect agent names from both implementations
        simple_agents = list(simple_agents_data.keys())
        afp_agents = list(afp_agents_data.keys())
        all_agents = sorted(list(set(simple_agents + afp_agents)))
        
        if not all_agents:
            print("Warning: No agents found for processing time comparison")
            return
        
        # Calculate average processing time for each agent
        simple_times = []
        afp_times = []
        
        for agent in all_agents:
            simple_times.append(np.mean(simple_agents_data.get(agent, [0])))
            afp_times.append(np.mean(afp_agents_data.get(agent, [0])))
        
        # Create x-axis labels
        x = np.arange(len(all_agents))
        width = 0.35
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bars
        ax.bar(x - width/2, simple_times, width, label='Standard MultiAgent')
        ax.bar(x + width/2, afp_times, width, label='AFP MultiAgent')
        
        # Add labels and title
        ax.set_xlabel('Agent')
        ax.set_ylabel('Average Processing Time (seconds)')
        ax.set_title('Agent Processing Time Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(all_agents)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'agent_processing_time_comparison.png'))
        plt.close()
    
    def _create_classification_time_comparison(self, output_dir):
        """Create a bar chart comparing classification times."""
        # Extract data
        simple_times = []
        afp_times = []
        
        # Check if metrics are available and in the expected format
        if hasattr(self, 'classification_times_simple') and isinstance(self.classification_times_simple, list):
            simple_times = self.classification_times_simple
        elif hasattr(self, 'simple_metrics') and isinstance(self.simple_metrics, dict) and "classification_times" in self.simple_metrics:
            simple_times = self.simple_metrics["classification_times"]
        
        if hasattr(self, 'classification_times_afp') and isinstance(self.classification_times_afp, list):
            afp_times = self.classification_times_afp
        elif hasattr(self, 'afp_metrics') and isinstance(self.afp_metrics, dict) and "classification_times" in self.afp_metrics:
            afp_times = self.afp_metrics["classification_times"]
        
        # Ensure we have data to plot
        if not simple_times and not afp_times:
            print("Warning: No classification time data available for visualization")
            return
        
        # Ensure both lists have the same length
        max_len = max(len(simple_times), len(afp_times))
        simple_times = simple_times + [0] * (max_len - len(simple_times))
        afp_times = afp_times + [0] * (max_len - len(afp_times))
        
        # Create x-axis labels
        x = np.arange(len(simple_times))
        width = 0.35
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bars
        ax.bar(x - width/2, simple_times, width, label='Standard MultiAgent')
        ax.bar(x + width/2, afp_times, width, label='AFP MultiAgent')
        
        # Add labels and title
        ax.set_xlabel('Test Case')
        ax.set_ylabel('Classification Time (seconds)')
        ax.set_title('Classification Time Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Case {i+1}' for i in range(len(simple_times))])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'classification_time_comparison.png'))
        plt.close()


def main():
    """Run the performance comparison tests."""
    # Create a performance tester
    tester = PerformanceTest()
    
    # Run the tests
    tester.run_simple_multiagent_test()
    tester.run_afp_multiagent_test()
    # tester.run_concurrent_load_test()  # Uncomment to run concurrent load test
    
    # Generate visualizations
    tester.generate_visualization(output_dir="visualizations")
    
    # Create summary report
    create_summary_report(results_file="multiagent_performance_results.json", output_dir="visualizations")


def create_summary_report(results_file="multiagent_performance_results.json", output_dir="visualizations"):
    """Create a summary report from test results including visualizations."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if results file exists
    if not os.path.exists(results_file):
        print(f"Warning: Results file {results_file} not found. Creating a basic report.")
        
        # Create a basic text summary
        text_summary = """
        MultiAgent Performance Comparison Summary
        =========================================
        
        No performance results file found. Please run the tests to generate results.
        """
        
        # Save the text summary
        with open(os.path.join(output_dir, "performance_summary.txt"), "w") as f:
            f.write(text_summary)
        
        # Create a basic HTML report
        html_report = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MultiAgent Performance Comparison Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .section { margin-bottom: 30px; }
                .visualization { margin: 20px 0; text-align: center; }
                img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>MultiAgent Performance Comparison Report</h1>
            
            <div class="section">
                <h2>No Performance Results Available</h2>
                <p>No performance results file was found. Please run the tests to generate results.</p>
            </div>
            
            <div class="section">
                <h2>Performance Visualizations</h2>
                
                <div class="visualization">
                    <h3>Response Time Comparison</h3>
                    <img src="response_time_comparison.png" alt="Response Time Comparison">
                </div>
                
                <div class="visualization">
                    <h3>Framework Overhead Comparison</h3>
                    <img src="framework_overhead_comparison.png" alt="Framework Overhead Comparison">
                </div>
                
                <div class="visualization">
                    <h3>Message Count Comparison</h3>
                    <img src="message_count_comparison.png" alt="Message Count Comparison">
                </div>
                
                <div class="visualization">
                    <h3>Agent Processing Time Comparison</h3>
                    <img src="agent_processing_time_comparison.png" alt="Agent Processing Time Comparison">
                </div>
                
                <div class="visualization">
                    <h3>Classification Time Comparison</h3>
                    <img src="classification_time_comparison.png" alt="Classification Time Comparison">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save the HTML report
        with open(os.path.join(output_dir, "performance_report.html"), "w") as f:
            f.write(html_report)
            
        print(f"\nSummary report created at {os.path.join(output_dir, 'performance_summary.txt')}")
        print(f"HTML report created at {os.path.join(output_dir, 'performance_report.html')}")
        return
    
    try:
        # Load results from file
        with open(results_file, "r") as f:
            results = json.load(f)
        
        # Extract key metrics with error handling
        sequential = results.get("sequential_tests", {})
        
        # With API time metrics
        with_api = sequential.get("with_api_time", {})
        simple_avg_response = with_api.get("simple", {}).get("avg_response_time", 0)
        afp_avg_response = with_api.get("afp", {}).get("avg_response_time", 0)
        simple_throughput = with_api.get("simple", {}).get("throughput", 0)
        afp_throughput = with_api.get("afp", {}).get("throughput", 0)
        response_improvement = with_api.get("improvements", {}).get("response_time_percent", 0)
        throughput_improvement = with_api.get("improvements", {}).get("throughput_percent", 0)
        
        # Framework overhead metrics
        framework = sequential.get("framework_overhead_only", {})
        simple_avg_overhead = framework.get("simple", {}).get("avg_overhead_time", 0)
        afp_avg_overhead = framework.get("afp", {}).get("avg_overhead_time", 0)
        simple_overhead_throughput = framework.get("simple", {}).get("overhead_throughput", 0)
        afp_overhead_throughput = framework.get("afp", {}).get("overhead_throughput", 0)
        overhead_improvement = framework.get("improvements", {}).get("overhead_time_percent", 0)
        overhead_throughput_improvement = framework.get("improvements", {}).get("overhead_throughput_percent", 0)
        
        # Create a text summary
        text_summary = f"""
        MultiAgent Performance Comparison Summary
        =========================================
        
        Sequential Test Results (INCLUDING API Time):
        Standard MultiAgent - Avg Response Time: {simple_avg_response:.4f} seconds
        AFP MultiAgent - Avg Response Time: {afp_avg_response:.4f} seconds
        Response Time Improvement: {response_improvement:.2f}%
        
        Standard MultiAgent - Throughput: {simple_throughput:.2f} messages/second
        AFP MultiAgent - Throughput: {afp_throughput:.2f} messages/second
        Throughput Improvement: {throughput_improvement:.2f}%
        
        Framework Overhead Only (EXCLUDING API Time):
        Standard MultiAgent - Avg Framework Overhead: {simple_avg_overhead:.4f} seconds
        AFP MultiAgent - Avg Framework Overhead: {afp_avg_overhead:.4f} seconds
        Framework Overhead Improvement: {overhead_improvement:.2f}%
        
        Standard MultiAgent - Framework Throughput: {simple_overhead_throughput:.2f} messages/second
        AFP MultiAgent - Framework Throughput: {afp_overhead_throughput:.2f} messages/second
        Framework Throughput Improvement: {overhead_throughput_improvement:.2f}%
        """
        
        # Save the text summary
        with open(os.path.join(output_dir, "performance_summary.txt"), "w") as f:
            f.write(text_summary)
            
        # Create HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MultiAgent Performance Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .improvement {{ color: green; font-weight: bold; }}
                .section {{ margin-bottom: 30px; }}
                .visualization {{ margin: 20px 0; text-align: center; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>MultiAgent Performance Comparison Report</h1>
            
            <div class="section">
                <h2>System Information</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Test Date</td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                    <tr><td>Number of Test Cases</td><td>{len(results.get("test_cases", []))}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Sequential Test Results (INCLUDING API Time)</h2>
                <table>
                    <tr>
                        <th>Implementation</th>
                        <th>Avg Response Time (s)</th>
                        <th>Throughput (msg/s)</th>
                    </tr>
                    <tr>
                        <td>Standard MultiAgent</td>
                        <td>{simple_avg_response:.4f}</td>
                        <td>{simple_throughput:.2f}</td>
                    </tr>
                    <tr>
                        <td>AFP MultiAgent</td>
                        <td>{afp_avg_response:.4f}</td>
                        <td>{afp_throughput:.2f}</td>
                    </tr>
                    <tr>
                        <td>Improvement</td>
                        <td class="improvement">{response_improvement:.2f}%</td>
                        <td class="improvement">{throughput_improvement:.2f}%</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Framework Overhead (EXCLUDING API Time)</h2>
                <table>
                    <tr>
                        <th>Implementation</th>
                        <th>Avg Framework Overhead (s)</th>
                        <th>Framework Throughput (msg/s)</th>
                    </tr>
                    <tr>
                        <td>Standard MultiAgent</td>
                        <td>{simple_avg_overhead:.4f}</td>
                        <td>{simple_overhead_throughput:.2f}</td>
                    </tr>
                    <tr>
                        <td>AFP MultiAgent</td>
                        <td>{afp_avg_overhead:.4f}</td>
                        <td>{afp_overhead_throughput:.2f}</td>
                    </tr>
                    <tr>
                        <td>Improvement</td>
                        <td class="improvement">{overhead_improvement:.2f}%</td>
                        <td class="improvement">{overhead_throughput_improvement:.2f}%</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Performance Visualizations</h2>
                
                <div class="visualization">
                    <h3>Response Time Comparison</h3>
                    <img src="response_time_comparison.png" alt="Response Time Comparison">
                </div>
                
                <div class="visualization">
                    <h3>Framework Overhead Comparison</h3>
                    <img src="framework_overhead_comparison.png" alt="Framework Overhead Comparison">
                </div>
                
                <div class="visualization">
                    <h3>Message Count Comparison</h3>
                    <img src="message_count_comparison.png" alt="Message Count Comparison">
                </div>
                
                <div class="visualization">
                    <h3>Agent Processing Time Comparison</h3>
                    <img src="agent_processing_time_comparison.png" alt="Agent Processing Time Comparison">
                </div>
                
                <div class="visualization">
                    <h3>Classification Time Comparison</h3>
                    <img src="classification_time_comparison.png" alt="Classification Time Comparison">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save the HTML report
        with open(os.path.join(output_dir, "performance_report.html"), "w") as f:
            f.write(html_report)
            
        print(f"\nSummary report created at {os.path.join(output_dir, 'performance_summary.txt')}")
        print(f"HTML report created at {os.path.join(output_dir, 'performance_report.html')}")
    
    except Exception as e:
        print(f"Error creating summary report: {e}")
        # Create a basic error report
        with open(os.path.join(output_dir, "performance_summary.txt"), "w") as f:
            f.write(f"Error creating summary report: {e}")


if __name__ == "__main__":
    main() 