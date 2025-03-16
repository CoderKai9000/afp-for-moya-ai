#!/usr/bin/env python
"""
Baseline performance test script.

This script measures baseline performance without using AFP for the same operations
as in the optimized test, to provide a fair comparison.
"""

import time
import json
import threading
import queue
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable


class BaselineAgent:
    """Simple agent without AFP messaging."""
    
    def __init__(self, id):
        """Initialize a simple agent."""
        self.id = id
        self.messages_received = 0
        self.messages_sent = 0
    
    def process_message(self, sender, content):
        """Process a message from another agent."""
        self.messages_received += 1
        # Simulate message processing time
        time.sleep(0.001)
        # Print message for debugging
        print(f"Agent {self.id} received: {content}")
        return True


class BaselineOrchestrator:
    """Simple orchestrator without AFP messaging."""
    
    def __init__(self):
        """Initialize a simple orchestrator."""
        self.agents = {}  # Dictionary of registered agents
    
    def register_agent(self, agent_id, agent_instance):
        """Register an agent with the orchestrator."""
        self.agents[agent_id] = agent_instance
        return True
    
    def route_message(self, sender, recipients, content):
        """Route a message from sender to recipients."""
        # Validate sender and recipients
        if sender not in self.agents:
            raise ValueError(f"Sender {sender} is not registered")
        
        success = True
        for recipient in recipients:
            if recipient not in self.agents:
                raise ValueError(f"Recipient {recipient} is not registered")
            
            # Deliver message to recipient
            if not self.agents[recipient].process_message(sender, content):
                success = False
        
        return success


def test_direct_communication(iterations=100):
    """Test direct message sending between two agents."""
    metrics = {
        "total_time": 0,
        "message_count": iterations,
        "avg_latency": 0,
        "throughput": 0
    }
    
    # Setup agents
    agent1 = BaselineAgent("agent1")
    agent2 = BaselineAgent("agent2")
    
    # Time the sending and receiving of messages
    start_time = time.time()
    
    for i in range(iterations):
        content = f"Test message {i}"
        agent2.process_message("agent1", content)
        
        # Wait for message to be processed
        time.sleep(0.001)  # Small sleep to ensure message processing
    
    end_time = time.time()
    total_time = end_time - start_time
    
    metrics["total_time"] = total_time
    metrics["avg_latency"] = total_time / iterations
    metrics["throughput"] = iterations / total_time
    
    return metrics


def test_multi_agent_orchestration(iterations=100):
    """Test orchestrator routing messages between multiple agents."""
    metrics = {
        "total_time": 0,
        "message_count": iterations,
        "avg_latency": 0,
        "throughput": 0
    }
    
    # Setup orchestrator
    orchestrator = BaselineOrchestrator()
    
    # Setup agents
    agent1 = BaselineAgent("agent1")
    agent2 = BaselineAgent("agent2")
    agent3 = BaselineAgent("agent3")
    
    # Register agents with orchestrator
    orchestrator.register_agent("agent1", agent1)
    orchestrator.register_agent("agent2", agent2)
    orchestrator.register_agent("agent3", agent3)
    
    # Time the sending and receiving of messages
    start_time = time.time()
    
    for i in range(iterations):
        content = f"Test message {i}"
        orchestrator.route_message("agent1", ["agent2", "agent3"], content)
        
        # Wait for message to be processed
        time.sleep(0.001)  # Small sleep to ensure message processing
    
    end_time = time.time()
    total_time = end_time - start_time
    
    metrics["total_time"] = total_time
    metrics["avg_latency"] = total_time / iterations
    metrics["throughput"] = iterations / total_time
    
    return metrics


def test_complex_workflow(iterations=50):
    """Test a complex workflow with multiple message hops."""
    metrics = {
        "total_time": 0,
        "message_count": iterations,
        "avg_latency": 0,
        "throughput": 0,
        "workflow_completion_time": []
    }
    
    # Setup orchestrator
    orchestrator = BaselineOrchestrator()
    
    # Setup workflow agents
    agent_router = BaselineAgent("router")
    agent_validator = BaselineAgent("validator")
    agent_processor = BaselineAgent("processor")
    agent_approver = BaselineAgent("approver")
    agent_notifier = BaselineAgent("notifier")
    
    # Register agents with orchestrator
    orchestrator.register_agent("router", agent_router)
    orchestrator.register_agent("validator", agent_validator)
    orchestrator.register_agent("processor", agent_processor)
    orchestrator.register_agent("approver", agent_approver)
    orchestrator.register_agent("notifier", agent_notifier)
    
    # Time the execution of workflows
    start_time = time.time()
    
    for i in range(iterations):
        workflow_id = f"workflow-{i}"
        workflow_start = time.time()  # Start timing this specific workflow
        
        # Initiate workflow
        orchestrator.route_message("router", ["validator"], f"Validation request {i}")
        
        # Ensure message processing before moving to next step
        time.sleep(0.002)
        
        # Simulate validation step
        orchestrator.route_message("validator", ["processor"], f"Validated request {i}")
        
        # Ensure message processing before moving to next step
        time.sleep(0.002)
        
        # Simulate processing step
        orchestrator.route_message("processor", ["approver"], f"Processed request {i}")
        
        # Ensure message processing before moving to next step
        time.sleep(0.002)
        
        # Simulate approval step
        orchestrator.route_message("approver", ["notifier"], f"Approved request {i}")
        
        # Ensure message processing before moving to next step
        time.sleep(0.002)
        
        # Simulate notification step (completion)
        orchestrator.route_message("notifier", ["router"], f"Workflow {i} completed")
        
        # Ensure final message processing is complete
        time.sleep(0.002)
        
        # Record workflow completion time
        workflow_end = time.time()
        workflow_duration = workflow_end - workflow_start
        metrics["workflow_completion_time"].append(workflow_duration)
        
        # Small sleep to separate workflows
        time.sleep(0.005)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Each workflow has 5 messages (validation, processing, approval, notification, completion)
    total_messages = iterations * 5
    
    metrics["total_time"] = total_time
    metrics["message_count"] = total_messages
    metrics["avg_latency"] = sum(metrics["workflow_completion_time"]) / iterations
    metrics["throughput"] = total_messages / total_time
    
    return metrics


def run_benchmarks():
    """Run all baseline performance benchmarks and print results."""
    print("Running baseline direct communication tests...")
    direct_results = test_direct_communication(iterations=100)
    
    print("\nRunning baseline multi-agent orchestration tests...")
    orchestration_results = test_multi_agent_orchestration(iterations=100)
    
    print("\nRunning baseline complex workflow tests...")
    workflow_results = test_complex_workflow(iterations=50)
    
    # Print results
    print("\n===== Baseline Performance Results =====")
    
    print("\nDirect Communication:")
    print(f"Total time: {direct_results['total_time']:.4f} seconds")
    print(f"Message count: {direct_results['message_count']}")
    print(f"Average latency: {direct_results['avg_latency'] * 1000:.4f} ms")
    print(f"Throughput: {direct_results['throughput']:.2f} messages/second")
    
    print("\nMulti-Agent Orchestration:")
    print(f"Total time: {orchestration_results['total_time']:.4f} seconds")
    print(f"Message count: {orchestration_results['message_count']}")
    print(f"Average latency: {orchestration_results['avg_latency'] * 1000:.4f} ms")
    print(f"Throughput: {orchestration_results['throughput']:.2f} messages/second")
    
    print("\nComplex Workflow:")
    print(f"Total time: {workflow_results['total_time']:.4f} seconds")
    print(f"Message count: {workflow_results['message_count']}")
    print(f"Average latency: {workflow_results['avg_latency'] * 1000:.4f} ms")
    print(f"Throughput: {workflow_results['throughput']:.2f} messages/second")
    
    # Save results to file
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {
            "direct_communication": direct_results,
            "multi_agent_orchestration": orchestration_results,
            "complex_workflow": workflow_results
        }
    }
    
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBaseline results saved to 'baseline_results.json'")


if __name__ == "__main__":
    run_benchmarks() 