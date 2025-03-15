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
    
    def route_message(self, sender: str, recipient: str, content: Any) -> Dict:
        """
        Route a message from sender to recipient using AFP.
        
        Args:
            sender: ID of the sending agent
            recipient: ID of the recipient agent
            content: Message content
            
        Returns:
            Response message content
        """
        # Create a message
        message = AFPMessage(
            sender=sender,
            recipients=[recipient],
            content_type=ContentType.JSON,
            content=content,
            metadata={"requires_response": True, "routed_by": "orchestrator"}
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


def test_direct_communication(num_messages=100, message_size=1024):
    """Test performance of direct AFP communication."""
    
    # Create communication bus
    bus = AFPCommunicationBus()
    
    # Create agents
    agent1 = AFPAgent("agent1", bus)
    agent2 = AFPAgent("agent2", bus)
    
    # Create test message content (adjust size as needed)
    message_content = {"data": "x" * message_size}
    
    # Measure AFP-based communication
    latencies = []
    start_total = time.time()
    
    for i in range(num_messages):
        # Time a request-response cycle
        start = time.time()
        
        # Send message from agent1 to agent2 directly with AFP
        response = agent1.send_message(
            recipient=agent2.id,
            content=message_content,
            sync=True
        )
        
        end = time.time()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    end_total = time.time()
    
    # Calculate metrics
    avg_latency = statistics.mean(latencies)
    p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else avg_latency
    p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else avg_latency
    throughput = num_messages / (end_total - start_total)
    
    # Clean up
    bus.shutdown()
    
    return {
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "throughput_msgs_per_sec": throughput,
        "total_time_sec": end_total - start_total
    }


def test_multi_agent_orchestration(num_agents=10, num_messages=100):
    """Test AFP orchestration with multiple agents."""
    
    # Create communication bus
    bus = AFPCommunicationBus()
    
    # Create agents
    agents = [AFPAgent(f"agent{i}", bus) for i in range(num_agents)]
    
    # Create orchestrator
    orchestrator = AFPOrchestrator(bus)
    
    # Measure time to distribute messages to all agents
    start = time.time()
    
    for i in range(num_messages):
        # Broadcast from first agent to all others
        orchestrator.broadcast_message(
            sender=agents[0].id,
            content={"message": f"Broadcast message {i}"}
        )
    
    end = time.time()
    
    # Clean up
    bus.shutdown()
    
    return {
        "num_agents": num_agents,
        "num_messages": num_messages,
        "total_time_sec": end - start,
        "msgs_per_second": num_messages / (end - start)
    }


def test_complex_workflow(num_agents=20, workflow_steps=5, runs=10):
    """
    Test a more complex workflow with multiple message hops using AFP.
    
    In this test, a workflow moves through multiple agents in sequence,
    simulating a more complex business process.
    """
    # Create communication bus
    bus = AFPCommunicationBus()
    
    # Create agents
    agents = [AFPAgent(f"agent{i}", bus) for i in range(num_agents)]
    
    # Create orchestrator
    orchestrator = AFPOrchestrator(bus)
    
    latencies = []
    
    for _ in range(runs):
        # Select random sequence of agents to form workflow
        workflow_agents = random.sample(agents, workflow_steps)
        
        start = time.time()
        
        # Pass a message through the workflow sequence
        message = {"workflow_id": random.randint(1000, 9999), "data": {"value": 100}}
        current_agent = workflow_agents[0]
        
        for next_agent in workflow_agents[1:]:
            # Agent processes and passes to next agent in workflow
            orchestrator.route_message(
                sender=current_agent.id,
                recipient=next_agent.id,
                content=message
            )
            
            # Update message for next hop
            message["previous_agent"] = current_agent.id
            current_agent = next_agent
        
        end = time.time()
        latencies.append((end - start) * 1000)  # ms
    
    # Clean up
    bus.shutdown()
    
    return {
        "num_agents": num_agents,
        "workflow_steps": workflow_steps,
        "avg_workflow_latency_ms": statistics.mean(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies)
    }


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
            test_result = test_direct_communication(num_messages=num_msgs, message_size=msg_size)
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