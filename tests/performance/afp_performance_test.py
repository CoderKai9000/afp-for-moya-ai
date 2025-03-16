#!/usr/bin/env python
"""
Performance test script for Agent Flow Protocol (AFP).

This script measures the performance of AFP for various communication patterns,
including direct communication, multi-agent orchestration, and complex workflows.
Results can be compared with the baseline measurements from baseline_test.py.
"""

import time
import random
import statistics
import json
import os
from datetime import datetime
import threading
from typing import Dict, List, Any, Optional, Callable

# Import AFP components
from moya.communication.afp.message import AFPMessage, ContentType
from moya.communication.afp.bus import AFPCommunicationBus
from moya.communication.afp.monitoring.metrics import AFPMetricsCollector


class AFPAgent:
    """Agent implementation using AFP for communication."""
    
    def __init__(self, agent_id: str, bus: AFPCommunicationBus):
        """
        Initialize an agent with AFP communication capabilities.
        
        Args:
            agent_id: Unique identifier for the agent
            bus: AFP communication bus
        """
        self.id = agent_id
        self.bus = bus
        self.messages_received = 0
        self.messages_sent = 0
        self.response_callbacks = {}
        
        # Register with the communication bus
        self.bus.register_agent(self.id)
        
        # Subscribe to messages addressed to this agent
        self.bus.subscribe(
            subscriber=self.id,
            callback=self._handle_message,
            pattern={"recipients": [self.id]}
        )
    
    def _handle_message(self, message: AFPMessage):
        """Handle incoming AFP messages."""
        self.messages_received += 1
        
        # Simulate processing time
        time.sleep(0.001)
        
        # Check if this is a synchronous request that needs a response
        if message.metadata.get("requires_response", False):
            # Create and send a response
            response = message.create_response(
                content={"status": "received", "agent": self.id},
                content_type=ContentType.JSON,
                metadata={"is_response": True}
            )
            self.bus.send_message(response)
    
    def send_message(self, recipient: str, content: Any, sync: bool = False, timeout: float = 5.0):
        """
        Send a message to another agent using AFP.
        
        Args:
            recipient: ID of the recipient agent
            content: Message content
            sync: Whether to wait for a response
            timeout: Timeout for sync requests in seconds
            
        Returns:
            Response message if sync=True, otherwise None
        """
        message = AFPMessage(
            sender=self.id,
            recipients=[recipient],
            content_type=ContentType.JSON,
            content=content,
            metadata={"requires_response": sync}
        )
        
        self.messages_sent += 1
        
        if sync:
            return self.bus.send_message_sync(message, timeout=timeout)
        else:
            self.bus.send_message(message)
            return None


class AFPOrchestrator:
    """Orchestrator implementation using AFP for agent coordination."""
    
    def __init__(self, bus: AFPCommunicationBus):
        """
        Initialize an orchestrator with AFP communication capabilities.
        
        Args:
            bus: AFP communication bus
        """
        self.bus = bus
        self.messages_processed = 0
        self.metrics = AFPMetricsCollector()
    
    def route_message(self, sender: str, recipient: str, content: Any, workflow_id: str = None) -> Dict:
        """
        Route a message from sender to recipient using AFP.
        
        Args:
            sender: ID of the sending agent
            recipient: ID of the recipient agent
            content: Message content
            workflow_id: ID of the workflow
            
        Returns:
            Response message content
        """
        # Create a message
        message = AFPMessage(
            sender=sender,
            recipients=[recipient],
            content_type=ContentType.JSON,
            content=content,
            metadata={"requires_response": True, "routed_by": "orchestrator", "workflow_id": workflow_id}
        )
        
        # Start a timer for tracking message processing
        timer_id = self.metrics.start_message_processing_timer("orchestrator")
        
        # Simulate orchestrator processing overhead
        time.sleep(0.002)
        
        # Send the message and wait for response
        response = self.bus.send_message_sync(message, timeout=5.0)
        
        # Stop the timer
        self.metrics.stop_message_processing_timer(timer_id, "orchestrator")
        
        self.messages_processed += 1
        
        return response.content if response else {"status": "error", "message": "No response received"}
    
    def broadcast_message(self, sender: str, content: Any) -> List[Dict]:
        """
        Broadcast a message from sender to all agents except sender.
        
        Args:
            sender: ID of the sending agent
            content: Message content
            
        Returns:
            List of response messages
        """
        # Create a message with all agents as recipients
        message = AFPMessage(
            sender=sender,
            recipients=["*"],  # Broadcast to all
            content_type=ContentType.JSON,
            content=content,
            metadata={"broadcast": True, "sent_by": sender}
        )
        
        # Send the broadcast message (no responses expected)
        self.bus.send_message(message)
        
        # In a real implementation, we would collect responses
        # For this test, we'll simulate responses
        results = []
        
        # For simplicity, we'll assume the orchestrator knows about 
        # all agents by tracking the message delivery count
        self.messages_processed += self.bus.get_agent_count() - 1
        
        return results


def test_direct_communication(iterations=100):
    """Test direct message sending between two agents."""
    metrics = {
        "total_time": 0,
        "message_count": iterations,
        "avg_latency": 0,
        "throughput": 0
    }
    
    # Create common communication bus
    bus = AFPCommunicationBus()
    
    # Setup agents
    agent1 = AFPAgent("agent1", bus)
    agent2 = AFPAgent("agent2", bus)
    
    # Subscribe agent2 to messages
    agent2.subscribe(lambda msg: msg.sender == "agent1")
    
    # Time the sending and receiving of messages
    start_time = time.time()
    
    for i in range(iterations):
        message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content=f"Test message {i}",
            priority=5  # Set a medium priority
        )
        
        agent1.send_message(message)
        
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
    
    # Create common communication bus
    bus = AFPCommunicationBus()
    
    # Setup orchestrator
    orchestrator = AFPOrchestrator(bus)
    
    # Setup agents
    agent1 = AFPAgent("agent1", bus)
    agent2 = AFPAgent("agent2", bus)
    agent3 = AFPAgent("agent3", bus)
    
    # Register agents with orchestrator
    orchestrator.register_agent("agent1")
    orchestrator.register_agent("agent2")
    orchestrator.register_agent("agent3")
    
    # Time the sending and receiving of messages
    start_time = time.time()
    
    for i in range(iterations):
        message = AFPMessage(
            sender="agent1",
            recipients=["agent2", "agent3"],
            content_type=ContentType.TEXT,
            content=f"Test message {i}",
            priority=5  # Set a medium priority
        )
        
        orchestrator.route_message(message)
        
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
    
    # Create common communication bus with optimization features enabled
    bus = AFPCommunicationBus(cache_size=100)
    
    # Setup orchestrator
    orchestrator = AFPOrchestrator(bus)
    
    # Setup workflow agents
    agent_router = AFPAgent("router", bus)
    agent_validator = AFPAgent("validator", bus)
    agent_processor = AFPAgent("processor", bus)
    agent_approver = AFPAgent("approver", bus)
    agent_notifier = AFPAgent("notifier", bus)
    
    # Register agents with orchestrator
    orchestrator.register_agent("router")
    orchestrator.register_agent("validator")
    orchestrator.register_agent("processor")
    orchestrator.register_agent("approver")
    orchestrator.register_agent("notifier")
    
    # Create direct routes for workflow agents
    bus.add_direct_route("router", "validator")
    bus.add_direct_route("validator", "processor")
    bus.add_direct_route("processor", "approver")
    bus.add_direct_route("approver", "notifier")
    bus.add_direct_route("notifier", "router")  # Complete the cycle
    
    # Time the execution of workflows
    start_time = time.time()
    
    for i in range(iterations):
        workflow_id = f"workflow-{i}"
        workflow_start = time.time()
        
        # Initiate workflow with a high priority
        initial_message = AFPMessage(
            sender="router",
            recipients=["validator"],
            content_type=ContentType.TEXT,
            content=f"Validation request {i}",
            metadata={"workflow_id": workflow_id, "step": "validation"},
            priority=10  # High priority for workflow messages
        )
        
        # Route through each step of the workflow
        orchestrator.route_message(initial_message, workflow_id=workflow_id)
        
        # Simulate validation step
        validation_message = AFPMessage(
            sender="validator",
            recipients=["processor"],
            content_type=ContentType.TEXT,
            content=f"Validated request {i}",
            metadata={"workflow_id": workflow_id, "step": "processing"},
            priority=10
        )
        orchestrator.route_message(validation_message, workflow_id=workflow_id)
        
        # Simulate processing step
        processing_message = AFPMessage(
            sender="processor",
            recipients=["approver"],
            content_type=ContentType.TEXT,
            content=f"Processed request {i}",
            metadata={"workflow_id": workflow_id, "step": "approval"},
            priority=10
        )
        orchestrator.route_message(processing_message, workflow_id=workflow_id)
        
        # Simulate approval step
        approval_message = AFPMessage(
            sender="approver",
            recipients=["notifier"],
            content_type=ContentType.TEXT,
            content=f"Approved request {i}",
            metadata={"workflow_id": workflow_id, "step": "notification"},
            priority=10
        )
        orchestrator.route_message(approval_message, workflow_id=workflow_id)
        
        # Simulate notification step (completion)
        completion_message = AFPMessage(
            sender="notifier",
            recipients=["router"],
            content_type=ContentType.TEXT,
            content=f"Workflow {i} completed",
            metadata={"workflow_id": workflow_id, "step": "complete"},
            priority=10
        )
        orchestrator.route_message(completion_message, workflow_id=workflow_id)
        
        # Record workflow completion time
        workflow_end = time.time()
        metrics["workflow_completion_time"].append(workflow_end - workflow_start)
        
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


def compare_with_baseline(afp_results, baseline_file="baseline_results.json"):
    """
    Compare AFP results with baseline results.
    
    Args:
        afp_results: Results from AFP performance tests
        baseline_file: File containing baseline results
        
    Returns:
        Comparison results
    """
    try:
        with open(baseline_file, "r") as f:
            baseline = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"error": f"Could not load baseline results from {baseline_file}"}
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Compare direct communication tests
    if "direct_communication" in baseline["tests"] and "direct_communication" in afp_results["tests"]:
        comparison["tests"]["direct_communication"] = []
        
        for baseline_test, afp_test in zip(baseline["tests"]["direct_communication"], afp_results["tests"]["direct_communication"]):
            if baseline_test["message_size_bytes"] == afp_test["message_size_bytes"] and baseline_test["num_messages"] == afp_test["num_messages"]:
                latency_change = (afp_test["avg_latency_ms"] - baseline_test["avg_latency_ms"]) / baseline_test["avg_latency_ms"] * 100
                throughput_change = (afp_test["throughput_msgs_per_sec"] - baseline_test["throughput_msgs_per_sec"]) / baseline_test["throughput_msgs_per_sec"] * 100
                
                comparison["tests"]["direct_communication"].append({
                    "message_size_bytes": baseline_test["message_size_bytes"],
                    "num_messages": baseline_test["num_messages"],
                    "baseline_latency_ms": baseline_test["avg_latency_ms"],
                    "afp_latency_ms": afp_test["avg_latency_ms"],
                    "latency_change_percent": latency_change,
                    "baseline_throughput": baseline_test["throughput_msgs_per_sec"],
                    "afp_throughput": afp_test["throughput_msgs_per_sec"],
                    "throughput_change_percent": throughput_change
                })
    
    # Compare multi-agent orchestration tests
    if "multi_agent_orchestration" in baseline["tests"] and "multi_agent_orchestration" in afp_results["tests"]:
        comparison["tests"]["multi_agent_orchestration"] = []
        
        for baseline_test, afp_test in zip(baseline["tests"]["multi_agent_orchestration"], afp_results["tests"]["multi_agent_orchestration"]):
            if baseline_test["num_agents"] == afp_test["num_agents"] and baseline_test["num_messages"] == afp_test["num_messages"]:
                throughput_change = (afp_test["msgs_per_second"] - baseline_test["msgs_per_second"]) / baseline_test["msgs_per_second"] * 100
                
                comparison["tests"]["multi_agent_orchestration"].append({
                    "num_agents": baseline_test["num_agents"],
                    "num_messages": baseline_test["num_messages"],
                    "baseline_throughput": baseline_test["msgs_per_second"],
                    "afp_throughput": afp_test["msgs_per_second"],
                    "throughput_change_percent": throughput_change
                })
    
    # Compare complex workflow tests
    if "complex_workflow" in baseline["tests"] and "complex_workflow" in afp_results["tests"]:
        comparison["tests"]["complex_workflow"] = []
        
        for baseline_test, afp_test in zip(baseline["tests"]["complex_workflow"], afp_results["tests"]["complex_workflow"]):
            if baseline_test["num_agents"] == afp_test["num_agents"] and baseline_test["workflow_steps"] == afp_test["workflow_steps"]:
                latency_change = (afp_test["avg_workflow_latency_ms"] - baseline_test["avg_workflow_latency_ms"]) / baseline_test["avg_workflow_latency_ms"] * 100
                
                comparison["tests"]["complex_workflow"].append({
                    "num_agents": baseline_test["num_agents"],
                    "workflow_steps": baseline_test["workflow_steps"],
                    "baseline_latency_ms": baseline_test["avg_workflow_latency_ms"],
                    "afp_latency_ms": afp_test["avg_workflow_latency_ms"],
                    "latency_change_percent": latency_change
                })
    
    return comparison


def run_benchmarks():
    """Run all AFP benchmarks and save results to a file."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    print("Running AFP direct communication tests...")
    direct_results = []
    for msg_size in [128, 1024, 10240]:
        for num_msgs in [100, 500]:
            test_result = test_direct_communication(iterations=num_msgs)
            test_result["message_size_bytes"] = msg_size
            test_result["num_messages"] = num_msgs
            direct_results.append(test_result)
            print(f"  Message size: {msg_size} bytes, Messages: {num_msgs}")
            print(f"  Throughput: {test_result['throughput_msgs_per_sec']:.2f} msgs/sec")
            print(f"  Avg Latency: {test_result['avg_latency_ms']:.2f} ms")
            print()
    
    results["tests"]["direct_communication"] = direct_results
    
    print("Running AFP multi-agent orchestration tests...")
    multi_agent_results = []
    for num_agents in [5, 10, 25, 50]:
        test_result = test_multi_agent_orchestration(num_agents=num_agents, num_messages=50)
        multi_agent_results.append(test_result)
        print(f"  Agents: {num_agents}")
        print(f"  Throughput: {test_result['msgs_per_second']:.2f} msgs/sec")
        print()
    
    results["tests"]["multi_agent_orchestration"] = multi_agent_results
    
    print("Running AFP complex workflow tests...")
    workflow_results = []
    for agents in [10, 25]:
        for steps in [3, 6, 10]:
            test_result = test_complex_workflow(num_agents=agents, workflow_steps=steps)
            workflow_results.append(test_result)
            print(f"  Agents: {agents}, Workflow steps: {steps}")
            print(f"  Avg Workflow Latency: {test_result['avg_workflow_latency_ms']:.2f} ms")
            print()
    
    results["tests"]["complex_workflow"] = workflow_results
    
    # Save results to file
    with open("afp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"AFP results saved to 'afp_results.json'")
    
    # Compare with baseline
    if os.path.exists("baseline_results.json"):
        print("Comparing with baseline results...")
        comparison = compare_with_baseline(results)
        
        # Save comparison to file
        with open("afp_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Comparison results saved to 'afp_comparison.json'")
        
        # Print summary of comparison
        print("\nPerformance Comparison Summary:")
        
        if "direct_communication" in comparison["tests"]:
            avg_latency_change = sum(test["latency_change_percent"] for test in comparison["tests"]["direct_communication"]) / len(comparison["tests"]["direct_communication"])
            avg_throughput_change = sum(test["throughput_change_percent"] for test in comparison["tests"]["direct_communication"]) / len(comparison["tests"]["direct_communication"])
            
            print(f"Direct Communication:")
            print(f"  Average Latency Change: {avg_latency_change:.2f}% ({'higher' if avg_latency_change > 0 else 'lower'})")
            print(f"  Average Throughput Change: {avg_throughput_change:.2f}% ({'higher' if avg_throughput_change > 0 else 'lower'})")
        
        if "multi_agent_orchestration" in comparison["tests"]:
            avg_throughput_change = sum(test["throughput_change_percent"] for test in comparison["tests"]["multi_agent_orchestration"]) / len(comparison["tests"]["multi_agent_orchestration"])
            
            print(f"Multi-Agent Orchestration:")
            print(f"  Average Throughput Change: {avg_throughput_change:.2f}% ({'higher' if avg_throughput_change > 0 else 'lower'})")
        
        if "complex_workflow" in comparison["tests"]:
            avg_latency_change = sum(test["latency_change_percent"] for test in comparison["tests"]["complex_workflow"]) / len(comparison["tests"]["complex_workflow"])
            
            print(f"Complex Workflow:")
            print(f"  Average Latency Change: {avg_latency_change:.2f}% ({'higher' if avg_latency_change > 0 else 'lower'})")


if __name__ == "__main__":
    run_benchmarks() 