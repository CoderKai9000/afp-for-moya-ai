#!/usr/bin/env python
"""
Optimized performance test script for Agent Flow Protocol (AFP).

This script measures the performance of the optimized AFP implementation
for various communication patterns, including direct communication,
multi-agent orchestration, and complex workflows.
"""

import time
import json
import threading
import queue
import heapq
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

# Import AFP components
from moya.communication.afp.message import AFPMessage, ContentType
from moya.communication.afp.bus import AFPCommunicationBus


class AFPAgent:
    """Agent implementation using AFP for communication."""
    
    def __init__(self, id, bus):
        """
        Initialize an agent with AFP communication capabilities.
        
        Args:
            id: Unique identifier for the agent
            bus: AFP communication bus
        """
        self.id = id
        self.bus = bus
        self.messages_received = 0
        self.messages_sent = 0
        
        # Register with the bus
        self.bus.register_agent(self.id)
        
        # Subscribe to messages addressed to this agent
        self.subscription_id = None
    
    def subscribe(self, filter_fn=None):
        """
        Subscribe to messages with an optional filter function.
        
        Args:
            filter_fn: Function that takes a message and returns True if interested
        """
        # Create a pattern-based subscription
        pattern = None
        
        if filter_fn is None:
            # Default pattern: accept messages addressed to this agent
            pattern = {"recipients": [self.id]}
        
        # Define a callback function that will be called when a message is received
        def message_callback(message):
            # Only process if filter function passes (if provided)
            if filter_fn is None or filter_fn(message):
                self.messages_received += 1
                # Process the message (in a real agent, this would do something with the content)
                print(f"Agent {self.id} received: {message.content}")
        
        # Subscribe to the bus
        self.subscription_id = self.bus.subscribe(
            subscriber=self.id,
            callback=message_callback,
            pattern=pattern
        )
        
        return self.subscription_id
    
    def send_message(self, message):
        """
        Send a message using the AFP bus.
        
        Args:
            message: AFPMessage to send
            
        Returns:
            True if message was sent successfully
        """
        self.messages_sent += 1
        return self.bus.send_message(message)


class AFPOrchestrator:
    """Orchestrator for AFP agents."""
    
    def __init__(self, id, bus):
        """
        Initialize a new AFPOrchestrator.
        
        Args:
            id: Unique identifier for this orchestrator
            bus: AFPCommunicationBus instance for messaging
        """
        self.id = id
        self.bus = bus
        self.agents = {}  # Dictionary of registered agents: {agent_id: agent_info}
        
        # Register with the bus
        self.bus.register_agent(self.id)
        
        # Subscribe to messages intended for the orchestrator
        self.bus.subscribe(
            subscriber=self.id,
            callback=self._handle_message,
            pattern={"recipients": [self.id]}
        )
    
    def _handle_message(self, message):
        """
        Handle messages sent to the orchestrator.
        
        Args:
            message: AFPMessage to handle
        """
        # Process orchestrator-specific messages
        print(f"Orchestrator {self.id} received: {message.content}")
    
    def register_agent(self, agent_id, capabilities=None):
        """
        Register an agent with the orchestrator.
        
        Args:
            agent_id: ID of the agent to register
            capabilities: Optional dict describing agent capabilities
        """
        self.agents[agent_id] = {
            "id": agent_id,
            "capabilities": capabilities or {},
            "registered_at": time.time()
        }
        return True
    
    def unregister_agent(self, agent_id):
        """
        Unregister an agent from the orchestrator.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False
    
    def route_message(self, message, workflow_id=None):
        """
        Route a message to its recipients.
        
        Args:
            message: AFPMessage to route
            workflow_id: Optional workflow ID for optimized routing
        
        Returns:
            True if message was routed successfully
        """
        # Add workflow_id to metadata if provided
        if workflow_id and isinstance(message.metadata, dict):
            message.metadata["workflow_id"] = workflow_id
        
        # Validate sender and recipients
        if message.sender not in self.agents:
            raise ValueError(f"Sender {message.sender} is not registered")
        
        for recipient in message.recipients:
            if recipient not in self.agents and recipient != self.id:
                raise ValueError(f"Recipient {recipient} is not registered")
        
        # Create copies of the message for each recipient
        success = True
        
        for recipient in message.recipients:
            # Create a new message targeted at the specific recipient
            recipient_msg = AFPMessage(
                sender=message.sender,
                recipients=[recipient],
                content_type=message.content_type,
                content=message.content,
                metadata=message.metadata.copy() if message.metadata else {},
                parent_message_id=message.message_id,
                priority=message.priority  # Preserve message priority
            )
            
            # Add routing information to metadata
            if recipient_msg.metadata is None:
                recipient_msg.metadata = {}
            recipient_msg.metadata["routed_by"] = self.id
            if workflow_id:
                recipient_msg.metadata["workflow_id"] = workflow_id
            
            # Use send_message for delivery
            if not self.bus.send_message(recipient_msg):
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
    
    # Create common communication bus
    bus = AFPCommunicationBus()
    
    # Setup agents
    agent1 = AFPAgent("agent1", bus)
    agent2 = AFPAgent("agent2", bus)
    
    # Define a custom filter function
    def agent1_filter(message):
        return message.sender == "agent1"
    
    # Subscribe agent2 to messages from agent1
    agent2.subscribe(agent1_filter)
    
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
    orchestrator = AFPOrchestrator("orchestrator", bus)
    
    # Setup agents
    agent1 = AFPAgent("agent1", bus)
    agent2 = AFPAgent("agent2", bus)
    agent3 = AFPAgent("agent3", bus)
    
    # Register agents with orchestrator
    orchestrator.register_agent("agent1")
    orchestrator.register_agent("agent2")
    orchestrator.register_agent("agent3")
    
    # Subscribe agents to messages
    agent2.subscribe()
    agent3.subscribe()
    
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
    bus = AFPCommunicationBus()
    
    # Setup orchestrator
    orchestrator = AFPOrchestrator("orchestrator", bus)
    
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
    
    # Subscribe agents to messages
    agent_validator.subscribe()
    agent_processor.subscribe()
    agent_approver.subscribe()
    agent_notifier.subscribe()
    agent_router.subscribe()
    
    # Create direct routes for workflow agents
    bus.create_direct_route("router", "validator")
    bus.create_direct_route("validator", "processor")
    bus.create_direct_route("processor", "approver")
    bus.create_direct_route("approver", "notifier")
    bus.create_direct_route("notifier", "router")  # Complete the cycle
    
    # Time the execution of workflows
    start_time = time.time()
    
    for i in range(iterations):
        workflow_id = f"workflow-{i}"
        workflow_start = time.time()  # Start timing this specific workflow
        
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
        
        # Ensure message processing before moving to next step
        time.sleep(0.002)
        
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
        
        # Ensure message processing before moving to next step
        time.sleep(0.002)
        
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
        
        # Ensure message processing before moving to next step
        time.sleep(0.002)
        
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
        
        # Ensure message processing before moving to next step
        time.sleep(0.002)
        
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
    """Run all AFP performance benchmarks and print results."""
    print("Running AFP direct communication tests...")
    direct_results = test_direct_communication(iterations=100)
    
    print("\nRunning AFP multi-agent orchestration tests...")
    orchestration_results = test_multi_agent_orchestration(iterations=100)
    
    print("\nRunning AFP complex workflow tests...")
    workflow_results = test_complex_workflow(iterations=50)
    
    # Print results
    print("\n===== AFP Performance Results =====")
    
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
    
    with open("afp_optimized_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAFP results saved to 'afp_optimized_results.json'")


if __name__ == "__main__":
    run_benchmarks() 