import time
import random
import statistics
import json
import os
from datetime import datetime

# Mock Moya agent classes for baseline testing
class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.messages_received = 0
        self.messages_sent = 0
    
    def receive_message(self, sender, content):
        self.messages_received += 1
        # Simulate processing time
        time.sleep(0.001)
        return {"status": "received", "agent": self.id}
    
    def send_message(self, recipient, content):
        self.messages_sent += 1
        return {"status": "sent", "from": self.id, "to": recipient}


class SimpleOrchestrator:
    def __init__(self, agents):
        self.agents = {agent.id: agent for agent in agents}
        self.messages_processed = 0
    
    def route_message(self, sender, recipient, content):
        """Route a message from sender to recipient."""
        self.messages_processed += 1
        
        # Simulate orchestrator processing overhead
        time.sleep(0.002)
        
        if recipient in self.agents:
            return self.agents[recipient].receive_message(sender, content)
        else:
            return {"status": "error", "message": f"Recipient {recipient} not found"}
    
    def broadcast_message(self, sender, content):
        """Broadcast a message from sender to all agents."""
        results = []
        for agent_id, agent in self.agents.items():
            if agent_id != sender:
                results.append(agent.receive_message(sender, content))
        
        self.messages_processed += len(self.agents) - 1
        return results


def test_direct_communication(num_messages=100, message_size=1024):
    """Test performance of direct orchestrator-based communication."""
    
    # Create agents
    agent1 = Agent("agent1")
    agent2 = Agent("agent2")
    
    # Create orchestrator
    orchestrator = SimpleOrchestrator([agent1, agent2])
    
    # Create test message (adjust size as needed)
    message_content = "x" * message_size
    
    # Measure orchestrator-based communication
    latencies = []
    start_total = time.time()
    
    for i in range(num_messages):
        # Time a request-response cycle
        start = time.time()
        
        # Send message from agent1 to agent2 through orchestrator
        response = orchestrator.route_message(
            sender=agent1.id,
            recipient=agent2.id,
            content=message_content
        )
        
        end = time.time()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    end_total = time.time()
    
    # Calculate metrics
    avg_latency = statistics.mean(latencies)
    p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else avg_latency
    p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else avg_latency
    throughput = num_messages / (end_total - start_total)
    
    return {
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "throughput_msgs_per_sec": throughput,
        "total_time_sec": end_total - start_total
    }

def test_multi_agent_orchestration(num_agents=10, num_messages=100):
    """Test orchestration with multiple agents."""
    
    # Create agents
    agents = [Agent(f"agent{i}") for i in range(num_agents)]
    
    # Create orchestrator
    orchestrator = SimpleOrchestrator(agents)
    
    # Measure time to distribute messages to all agents
    start = time.time()
    
    for i in range(num_messages):
        # Broadcast from first agent to all others
        orchestrator.broadcast_message(
            sender=agents[0].id,
            content=f"Broadcast message {i}"
        )
    
    end = time.time()
    
    return {
        "num_agents": num_agents,
        "num_messages": num_messages,
        "total_time_sec": end - start,
        "msgs_per_second": num_messages / (end - start)
    }

def test_complex_workflow(num_agents=20, workflow_steps=5, runs=10):
    """
    Test a more complex workflow with multiple message hops.
    
    In this test, a workflow moves through multiple agents in sequence,
    simulating a more complex business process.
    """
    agents = [Agent(f"agent{i}") for i in range(num_agents)]
    orchestrator = SimpleOrchestrator(agents)
    
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
    
    return {
        "num_agents": num_agents,
        "workflow_steps": workflow_steps,
        "avg_workflow_latency_ms": statistics.mean(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies)
    }

def run_benchmarks():
    """Run all benchmarks and save results to a file."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    print("Running direct communication tests...")
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
    
    print("Running multi-agent orchestration tests...")
    multi_agent_results = []
    for num_agents in [5, 10, 25, 50]:
        test_result = test_multi_agent_orchestration(num_agents=num_agents, num_messages=50)
        multi_agent_results.append(test_result)
        print(f"  Agents: {num_agents}")
        print(f"  Throughput: {test_result['msgs_per_second']:.2f} msgs/sec")
        print()
    
    results["tests"]["multi_agent_orchestration"] = multi_agent_results
    
    print("Running complex workflow tests...")
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
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Baseline results saved to 'baseline_results.json'")

if __name__ == "__main__":
    run_benchmarks() 