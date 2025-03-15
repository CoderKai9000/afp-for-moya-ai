#!/usr/bin/env python
"""
Real-world integration test comparing performance before and after AFP implementation.

This script tests a real-world scenario where multiple agents communicate with Azure OpenAI
to process and analyze data, comparing direct API calls vs. AFP-mediated communication.
"""

import os
import time
import json
import statistics
import uuid
import threading
import asyncio
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
import requests

# Import AFP components
from moya.communication.afp.message import AFPMessage, ContentType
from moya.communication.afp.bus import AFPCommunicationBus
from moya.communication.afp.monitoring.metrics import AFPMetricsCollector
from moya.communication.afp.security.auth import HMACAuthenticator

# Azure OpenAI API credentials
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = "gpt-35-turbo"  # Default deployment
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"  # Default API version


def check_azure_credentials():
    """Check if Azure OpenAI API credentials are properly set."""
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        print("Error: Azure OpenAI API credentials not found.")
        print("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.")
        return False
    return True


def call_azure_openai(prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
    """
    Call Azure OpenAI API directly.
    
    Args:
        prompt: The prompt to send to the API
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        API response
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }
    
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }
    
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    
    start_time = time.time()
    response = requests.post(url, headers=headers, json=body)
    end_time = time.time()
    
    response_time = end_time - start_time
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return {"error": response.text, "response_time": response_time}
    
    # Parse the response
    result = response.json()
    result["response_time"] = response_time
    
    return result


class AIAgent:
    """Agent for direct API calls (without AFP)."""
    
    def __init__(self, agent_id: str):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        self.id = agent_id
        self.messages_sent = 0
        self.messages_received = 0
        self.response_times = []
    
    def process_query(self, query: str, max_tokens: int = 100) -> Dict[str, Any]:
        """
        Process a query using Azure OpenAI.
        
        Args:
            query: The query to process
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Processing result
        """
        self.messages_sent += 1
        
        # Call Azure OpenAI API
        result = call_azure_openai(query, max_tokens)
        
        self.messages_received += 1
        
        if "response_time" in result:
            self.response_times.append(result["response_time"])
        
        return result


class AFPAIAgent:
    """Agent using AFP for communication with Azure OpenAI."""
    
    def __init__(self, agent_id: str, bus: AFPCommunicationBus, authenticator: HMACAuthenticator,
                 ai_agent_id: str = "ai_processor"):
        """
        Initialize an agent with AFP communication.
        
        Args:
            agent_id: Unique identifier for the agent
            bus: AFP communication bus
            authenticator: Authentication provider
            ai_agent_id: ID of the AI processor agent
        """
        self.id = agent_id
        self.bus = bus
        self.authenticator = authenticator
        self.ai_agent_id = ai_agent_id
        self.metrics = AFPMetricsCollector()
        self.response_times = []
        self.responses = {}
        self.secret_key = authenticator.register_agent(agent_id)
        
        # Register with communication bus
        self.bus.register_agent(agent_id)
        
        # Subscribe to messages addressed to this agent
        self.bus.subscribe(
            subscriber=agent_id,
            callback=self._handle_message,
            pattern={"recipients": [agent_id]}
        )
    
    def _handle_message(self, message: AFPMessage):
        """
        Handle incoming AFP messages.
        
        Args:
            message: The received message
        """
        # Record receipt metrics
        self.metrics.record_message_received(len(message.serialize()), self.id)
        
        # Verify message authenticity
        if not self.authenticator.verify_message(message):
            print(f"Warning: Received unauthenticated message from {message.sender}")
            return
        
        # Store response if it's expected
        if message.metadata.get("query_id") in self.responses:
            self.responses[message.metadata["query_id"]] = {
                "result": message.content,
                "response_time": time.time() - message.metadata.get("start_time", 0)
            }
            
            if "response_time" in message.metadata:
                self.response_times.append(message.metadata["response_time"])
    
    def process_query(self, query: str, max_tokens: int = 100, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Process a query using AFP and Azure OpenAI.
        
        Args:
            query: The query to process
            max_tokens: Maximum number of tokens to generate
            timeout: Timeout for the request in seconds
            
        Returns:
            Processing result
        """
        # Generate a unique query ID
        query_id = str(uuid.uuid4())
        
        # Store the query ID for response tracking
        self.responses[query_id] = None
        
        # Create and send the message
        start_time = time.time()
        message = AFPMessage(
            sender=self.id,
            recipients=[self.ai_agent_id],
            content_type=ContentType.JSON,
            content={"query": query, "max_tokens": max_tokens},
            metadata={"query_id": query_id, "start_time": start_time, "requires_response": True}
        )
        
        # Sign the message
        signed_message = self.authenticator.sign_message(message, self.id, self.secret_key)
        
        # Record send metrics
        self.metrics.record_message_sent(len(signed_message.serialize()), self.id)
        
        # Send the message
        self.bus.send_message(signed_message)
        
        # Wait for response
        wait_start = time.time()
        while self.responses[query_id] is None:
            if time.time() - wait_start > timeout:
                del self.responses[query_id]
                return {"error": "Timeout waiting for response", "response_time": timeout}
            time.sleep(0.1)
        
        # Get the response
        response = self.responses[query_id]
        del self.responses[query_id]
        
        return response


class AIProcessorAgent:
    """Agent that processes AI requests using AFP."""
    
    def __init__(self, agent_id: str, bus: AFPCommunicationBus, authenticator: HMACAuthenticator):
        """
        Initialize an AI processor agent.
        
        Args:
            agent_id: Unique identifier for the agent
            bus: AFP communication bus
            authenticator: Authentication provider
        """
        self.id = agent_id
        self.bus = bus
        self.authenticator = authenticator
        self.metrics = AFPMetricsCollector()
        self.secret_key = authenticator.register_agent(agent_id)
        self.running = False
        self.processing_thread = None
        
        # Register with communication bus
        self.bus.register_agent(agent_id)
        
        # Subscribe to messages addressed to this agent
        self.bus.subscribe(
            subscriber=agent_id,
            callback=self._handle_message,
            pattern={"recipients": [agent_id]}
        )
    
    def start(self):
        """Start the agent."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print(f"AI Processor Agent {self.id} started")
    
    def stop(self):
        """Stop the agent."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print(f"AI Processor Agent {self.id} stopped")
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.running:
            time.sleep(0.1)
    
    def _handle_message(self, message: AFPMessage):
        """
        Handle incoming AFP messages.
        
        Args:
            message: The received message
        """
        # Record receipt metrics
        self.metrics.record_message_received(len(message.serialize()), self.id)
        
        # Verify message authenticity
        if not self.authenticator.verify_message(message):
            print(f"Warning: Received unauthenticated message from {message.sender}")
            return
        
        # Process the message
        if message.content_type == ContentType.JSON and isinstance(message.content, dict):
            if "query" in message.content:
                self._process_ai_query(message)
    
    def _process_ai_query(self, message: AFPMessage):
        """
        Process an AI query.
        
        Args:
            message: The query message
        """
        query = message.content.get("query", "")
        max_tokens = message.content.get("max_tokens", 100)
        query_id = message.metadata.get("query_id")
        start_time = message.metadata.get("start_time", 0)
        
        # Call Azure OpenAI API
        result = call_azure_openai(query, max_tokens)
        
        # Calculate response time
        response_time = result.get("response_time", 0)
        
        # Create response message
        response = message.create_response(
            content=result,
            content_type=ContentType.JSON,
            metadata={
                "query_id": query_id,
                "start_time": start_time,
                "response_time": response_time
            }
        )
        
        # Sign the response
        signed_response = self.authenticator.sign_message(response, self.id, self.secret_key)
        
        # Record send metrics
        self.metrics.record_message_sent(len(signed_response.serialize()), self.id)
        
        # Send the response
        self.bus.send_message(signed_response)


def test_direct_api_calls(num_queries=10, num_agents=5):
    """
    Test direct API calls (without AFP).
    
    Args:
        num_queries: Number of queries per agent
        num_agents: Number of agents
        
    Returns:
        Test results
    """
    print(f"\nRunning direct API calls test with {num_agents} agents and {num_queries} queries each...")
    
    # Create agents
    agents = [AIAgent(f"direct_agent_{i}") for i in range(num_agents)]
    
    # Test data
    test_queries = [
        "What are the key principles of artificial intelligence?",
        "Explain the concept of machine learning in simple terms.",
        "What is the difference between supervised and unsupervised learning?",
        "How does natural language processing work?",
        "What are neural networks and how do they function?",
        "Explain the concept of deep learning.",
        "What is reinforcement learning?",
        "How is computer vision used in AI applications?",
        "What ethical considerations are important in AI development?",
        "What is the future of artificial intelligence?"
    ]
    
    # Ensure we have enough test queries
    while len(test_queries) < num_queries:
        test_queries.extend(test_queries[:num_queries - len(test_queries)])
    
    # Truncate if we have too many
    test_queries = test_queries[:num_queries]
    
    start_time = time.time()
    
    # Run queries concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
        futures = []
        
        for agent in agents:
            for query in test_queries:
                futures.append(executor.submit(agent.process_query, query))
        
        # Wait for all futures to complete
        concurrent.futures.wait(futures)
    
    end_time = time.time()
    
    # Calculate metrics
    total_queries = num_agents * num_queries
    total_time = end_time - start_time
    queries_per_second = total_queries / total_time
    
    # Collect response times
    all_response_times = []
    for agent in agents:
        all_response_times.extend(agent.response_times)
    
    # Calculate response time statistics
    if all_response_times:
        avg_response_time = statistics.mean(all_response_times)
        p95_response_time = statistics.quantiles(all_response_times, n=20)[18] if len(all_response_times) >= 20 else avg_response_time
        p99_response_time = statistics.quantiles(all_response_times, n=100)[98] if len(all_response_times) >= 100 else avg_response_time
    else:
        avg_response_time = p95_response_time = p99_response_time = 0
    
    results = {
        "total_time_sec": total_time,
        "total_queries": total_queries,
        "queries_per_second": queries_per_second,
        "avg_response_time_sec": avg_response_time,
        "p95_response_time_sec": p95_response_time,
        "p99_response_time_sec": p99_response_time
    }
    
    print(f"Direct API calls completed in {total_time:.2f} seconds")
    print(f"Throughput: {queries_per_second:.2f} queries/second")
    print(f"Average response time: {avg_response_time:.2f} seconds")
    
    return results


def test_afp_api_calls(num_queries=10, num_agents=5):
    """
    Test API calls using AFP.
    
    Args:
        num_queries: Number of queries per agent
        num_agents: Number of agents
        
    Returns:
        Test results
    """
    print(f"\nRunning AFP API calls test with {num_agents} agents and {num_queries} queries each...")
    
    # Create communication bus
    bus = AFPCommunicationBus()
    
    # Create authenticator
    authenticator = HMACAuthenticator()
    
    # Create AI processor agent
    processor = AIProcessorAgent("ai_processor", bus, authenticator)
    processor.start()
    
    # Create agents
    agents = [AFPAIAgent(f"afp_agent_{i}", bus, authenticator) for i in range(num_agents)]
    
    # Test data
    test_queries = [
        "What are the key principles of artificial intelligence?",
        "Explain the concept of machine learning in simple terms.",
        "What is the difference between supervised and unsupervised learning?",
        "How does natural language processing work?",
        "What are neural networks and how do they function?",
        "Explain the concept of deep learning.",
        "What is reinforcement learning?",
        "How is computer vision used in AI applications?",
        "What ethical considerations are important in AI development?",
        "What is the future of artificial intelligence?"
    ]
    
    # Ensure we have enough test queries
    while len(test_queries) < num_queries:
        test_queries.extend(test_queries[:num_queries - len(test_queries)])
    
    # Truncate if we have too many
    test_queries = test_queries[:num_queries]
    
    # Allow time for agents to initialize
    time.sleep(1)
    
    start_time = time.time()
    
    # Run queries concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
        futures = []
        
        for agent in agents:
            for query in test_queries:
                futures.append(executor.submit(agent.process_query, query))
        
        # Wait for all futures to complete
        concurrent.futures.wait(futures)
    
    end_time = time.time()
    
    # Calculate metrics
    total_queries = num_agents * num_queries
    total_time = end_time - start_time
    queries_per_second = total_queries / total_time
    
    # Collect response times
    all_response_times = []
    for agent in agents:
        all_response_times.extend(agent.response_times)
    
    # Calculate response time statistics
    if all_response_times:
        avg_response_time = statistics.mean(all_response_times)
        p95_response_time = statistics.quantiles(all_response_times, n=20)[18] if len(all_response_times) >= 20 else avg_response_time
        p99_response_time = statistics.quantiles(all_response_times, n=100)[98] if len(all_response_times) >= 100 else avg_response_time
    else:
        avg_response_time = p95_response_time = p99_response_time = 0
    
    # Clean up
    processor.stop()
    bus.shutdown()
    
    results = {
        "total_time_sec": total_time,
        "total_queries": total_queries,
        "queries_per_second": queries_per_second,
        "avg_response_time_sec": avg_response_time,
        "p95_response_time_sec": p95_response_time,
        "p99_response_time_sec": p99_response_time
    }
    
    print(f"AFP API calls completed in {total_time:.2f} seconds")
    print(f"Throughput: {queries_per_second:.2f} queries/second")
    print(f"Average response time: {avg_response_time:.2f} seconds")
    
    return results


def test_async_workflow(use_afp=False, num_tasks=10):
    """
    Test a real-world asynchronous workflow.
    
    In this test, we simulate a workflow where multiple agents work together
    to process data, with some agents generating content and others analyzing it.
    
    Args:
        use_afp: Whether to use AFP
        num_tasks: Number of tasks to process
        
    Returns:
        Test results
    """
    print(f"\nRunning {'AFP' if use_afp else 'direct'} asynchronous workflow test with {num_tasks} tasks...")
    
    workflow_results = []
    start_time = time.time()
    
    if use_afp:
        # Create communication bus and authenticator
        bus = AFPCommunicationBus()
        authenticator = HMACAuthenticator()
        
        # Create processor agent
        processor = AIProcessorAgent("ai_processor", bus, authenticator)
        processor.start()
        
        # Create workflow agents
        generator_agent = AFPAIAgent("generator", bus, authenticator)
        analyzer_agent = AFPAIAgent("analyzer", bus, authenticator)
        
        # Allow time for agents to initialize
        time.sleep(1)
        
        # Run the workflow
        for i in range(num_tasks):
            # Step 1: Generate content
            generation_prompt = f"Generate a short paragraph about topic {i+1}: artificial intelligence."
            generation_result = generator_agent.process_query(generation_prompt)
            
            # Extract generated content
            if isinstance(generation_result, dict) and "result" in generation_result:
                generated_content = generation_result["result"].get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                generated_content = "Failed to generate content"
            
            # Step 2: Analyze the generated content
            analysis_prompt = f"Analyze the following content and provide a brief summary:\n\n{generated_content}"
            analysis_result = analyzer_agent.process_query(analysis_prompt)
            
            # Record workflow result
            workflow_results.append({
                "task_id": i,
                "generation_time": generation_result.get("response_time", 0),
                "analysis_time": analysis_result.get("response_time", 0),
                "total_time": generation_result.get("response_time", 0) + analysis_result.get("response_time", 0)
            })
        
        # Clean up
        processor.stop()
        bus.shutdown()
    
    else:
        # Create direct agents
        generator_agent = AIAgent("direct_generator")
        analyzer_agent = AIAgent("direct_analyzer")
        
        # Run the workflow
        for i in range(num_tasks):
            # Step 1: Generate content
            generation_prompt = f"Generate a short paragraph about topic {i+1}: artificial intelligence."
            generation_result = generator_agent.process_query(generation_prompt)
            
            # Extract generated content
            generated_content = generation_result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not generated_content:
                generated_content = "Failed to generate content"
            
            # Step 2: Analyze the generated content
            analysis_prompt = f"Analyze the following content and provide a brief summary:\n\n{generated_content}"
            analysis_result = analyzer_agent.process_query(analysis_prompt)
            
            # Record workflow result
            workflow_results.append({
                "task_id": i,
                "generation_time": generation_result.get("response_time", 0),
                "analysis_time": analysis_result.get("response_time", 0),
                "total_time": generation_result.get("response_time", 0) + analysis_result.get("response_time", 0)
            })
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    task_times = [result["total_time"] for result in workflow_results]
    
    if task_times:
        avg_task_time = statistics.mean(task_times)
        p95_task_time = statistics.quantiles(task_times, n=20)[18] if len(task_times) >= 20 else avg_task_time
    else:
        avg_task_time = p95_task_time = 0
    
    tasks_per_second = num_tasks / total_time
    
    results = {
        "total_time_sec": total_time,
        "num_tasks": num_tasks,
        "tasks_per_second": tasks_per_second,
        "avg_task_time_sec": avg_task_time,
        "p95_task_time_sec": p95_task_time
    }
    
    print(f"Asynchronous workflow completed in {total_time:.2f} seconds")
    print(f"Throughput: {tasks_per_second:.2f} tasks/second")
    print(f"Average task time: {avg_task_time:.2f} seconds")
    
    return results


def run_tests():
    """Run all tests and compare results."""
    if not check_azure_credentials():
        return
    
    print("=" * 80)
    print("Real-world AFP Integration Test with Azure OpenAI")
    print("=" * 80)
    
    # Store results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "direct_api": {},
        "afp_api": {},
        "direct_workflow": {},
        "afp_workflow": {},
        "comparison": {}
    }
    
    # Run direct API call tests
    try:
        results["direct_api"] = test_direct_api_calls(num_queries=5, num_agents=3)
    except Exception as e:
        print(f"Error running direct API tests: {e}")
        results["direct_api"] = {"error": str(e)}
    
    # Run AFP API call tests
    try:
        results["afp_api"] = test_afp_api_calls(num_queries=5, num_agents=3)
    except Exception as e:
        print(f"Error running AFP API tests: {e}")
        results["afp_api"] = {"error": str(e)}
    
    # Run direct workflow tests
    try:
        results["direct_workflow"] = test_async_workflow(use_afp=False, num_tasks=5)
    except Exception as e:
        print(f"Error running direct workflow tests: {e}")
        results["direct_workflow"] = {"error": str(e)}
    
    # Run AFP workflow tests
    try:
        results["afp_workflow"] = test_async_workflow(use_afp=True, num_tasks=5)
    except Exception as e:
        print(f"Error running AFP workflow tests: {e}")
        results["afp_workflow"] = {"error": str(e)}
    
    # Calculate comparison metrics
    if "queries_per_second" in results["direct_api"] and "queries_per_second" in results["afp_api"]:
        direct_qps = results["direct_api"]["queries_per_second"]
        afp_qps = results["afp_api"]["queries_per_second"]
        qps_change = ((afp_qps - direct_qps) / direct_qps) * 100 if direct_qps > 0 else 0
        
        results["comparison"]["api_throughput_change_percent"] = qps_change
    
    if "avg_response_time_sec" in results["direct_api"] and "avg_response_time_sec" in results["afp_api"]:
        direct_rt = results["direct_api"]["avg_response_time_sec"]
        afp_rt = results["afp_api"]["avg_response_time_sec"]
        rt_change = ((afp_rt - direct_rt) / direct_rt) * 100 if direct_rt > 0 else 0
        
        results["comparison"]["api_response_time_change_percent"] = rt_change
    
    if "tasks_per_second" in results["direct_workflow"] and "tasks_per_second" in results["afp_workflow"]:
        direct_tps = results["direct_workflow"]["tasks_per_second"]
        afp_tps = results["afp_workflow"]["tasks_per_second"]
        tps_change = ((afp_tps - direct_tps) / direct_tps) * 100 if direct_tps > 0 else 0
        
        results["comparison"]["workflow_throughput_change_percent"] = tps_change
    
    if "avg_task_time_sec" in results["direct_workflow"] and "avg_task_time_sec" in results["afp_workflow"]:
        direct_tt = results["direct_workflow"]["avg_task_time_sec"]
        afp_tt = results["afp_workflow"]["avg_task_time_sec"]
        tt_change = ((afp_tt - direct_tt) / direct_tt) * 100 if direct_tt > 0 else 0
        
        results["comparison"]["workflow_task_time_change_percent"] = tt_change
    
    # Save results to file
    with open("afp_ai_integration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    
    # Print API call comparison
    print("\nAPI Call Performance:")
    if "api_throughput_change_percent" in results["comparison"]:
        change = results["comparison"]["api_throughput_change_percent"]
        print(f"  Throughput: {'+'if change > 0 else ''}{change:.2f}% with AFP")
    
    if "api_response_time_change_percent" in results["comparison"]:
        change = results["comparison"]["api_response_time_change_percent"]
        print(f"  Response Time: {'+'if change > 0 else ''}{change:.2f}% with AFP")
    
    # Print workflow comparison
    print("\nWorkflow Performance:")
    if "workflow_throughput_change_percent" in results["comparison"]:
        change = results["comparison"]["workflow_throughput_change_percent"]
        print(f"  Throughput: {'+'if change > 0 else ''}{change:.2f}% with AFP")
    
    if "workflow_task_time_change_percent" in results["comparison"]:
        change = results["comparison"]["workflow_task_time_change_percent"]
        print(f"  Task Time: {'+'if change > 0 else ''}{change:.2f}% with AFP")
    
    print("\nDetailed results saved to 'afp_ai_integration_results.json'")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_tests() 