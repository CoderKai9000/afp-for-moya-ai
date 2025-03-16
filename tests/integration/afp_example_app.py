#!/usr/bin/env python
"""
Example application demonstrating Agent Flow Protocol (AFP) in action.

This application simulates a simple multi-agent system where agents collaborate
to process data, showcasing the key features of AFP.
"""

import time
import random
import uuid
import json
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

# Import AFP components
from moya.communication.afp.message import AFPMessage, ContentType
from moya.communication.afp.bus import AFPCommunicationBus
from moya.communication.afp.monitoring.metrics import AFPMetricsCollector
from moya.communication.afp.monitoring.tracing import AFPMessageTracer
from moya.communication.afp.reliability.circuit_breaker import CircuitBreakerRegistry
from moya.communication.afp.security.auth import HMACAuthenticator


# Define agent roles
class AgentRole(Enum):
    """Roles that agents can play in the system."""
    COORDINATOR = "coordinator"    # Coordinates tasks and workflows
    PROCESSOR = "processor"        # Processes data
    STORAGE = "storage"            # Stores and retrieves data
    ANALYZER = "analyzer"          # Analyzes data and provides insights
    API = "api"                    # Provides external API access


class Task:
    """Represents a task to be processed by agents."""
    
    def __init__(self, task_type: str, data: Dict[str, Any], priority: int = 1):
        """
        Initialize a task.
        
        Args:
            task_type: Type of task
            data: Task data
            priority: Task priority (higher = more important)
        """
        self.id = str(uuid.uuid4())
        self.type = task_type
        self.data = data
        self.priority = priority
        self.status = "created"
        self.created_at = time.time()
        self.completed_at = None
        self.assigned_to = None
        self.results = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "assigned_to": self.assigned_to,
            "results": self.results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        task = cls(data["type"], data["data"], data["priority"])
        task.id = data["id"]
        task.status = data["status"]
        task.created_at = data["created_at"]
        task.completed_at = data["completed_at"]
        task.assigned_to = data["assigned_to"]
        task.results = data["results"]
        return task


class Agent:
    """Base agent class using AFP for communication."""
    
    def __init__(self, agent_id: str, role: AgentRole, bus: AFPCommunicationBus, authenticator: HMACAuthenticator):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Role of the agent in the system
            bus: AFP communication bus
            authenticator: Authentication provider
        """
        self.id = agent_id
        self.role = role
        self.bus = bus
        self.authenticator = authenticator
        self.tracer = AFPMessageTracer()
        self.metrics = AFPMetricsCollector()
        self.circuit_breakers = CircuitBreakerRegistry()
        self.running = False
        self.tasks: Dict[str, Task] = {}  # Task ID -> Task
        
        # Get secret key for this agent
        self.secret_key = authenticator.register_agent(agent_id)
        
        # Register with communication bus
        self.bus.register_agent(agent_id)
        
        # Subscribe to messages addressed to this agent
        self.bus.subscribe(
            subscriber=agent_id,
            callback=self._handle_message,
            pattern={"recipients": [agent_id]}
        )
        
        # Subscribe to broadcast messages
        self.bus.subscribe(
            subscriber=agent_id,
            callback=self._handle_message,
            pattern={"recipients": ["*"]}
        )
        
        print(f"Agent {agent_id} ({role.value}) initialized")
    
    def start(self):
        """Start the agent's processing loop."""
        self.running = True
        self._processing_thread = threading.Thread(target=self._processing_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        print(f"Agent {self.id} started")
    
    def stop(self):
        """Stop the agent's processing loop."""
        self.running = False
        if hasattr(self, '_processing_thread'):
            self._processing_thread.join(timeout=2.0)
        print(f"Agent {self.id} stopped")
    
    def _processing_loop(self):
        """Main processing loop for the agent."""
        while self.running:
            # Default implementation just sleeps
            time.sleep(1.0)
    
    def _handle_message(self, message: AFPMessage):
        """
        Handle incoming AFP messages.
        
        Args:
            message: The received message
        """
        # Trace message receipt
        span_id = self.tracer.trace_message_receive(
            message.message_id,
            message.sender,
            self.id,
            message.content_type.name,
            len(message.serialize())
        )
        
        # Record metrics
        self.metrics.record_message_received(len(message.serialize()), self.id)
        
        # Verify message authenticity
        if not self.authenticator.verify_message(message):
            print(f"WARNING: Agent {self.id} received unauthenticated message from {message.sender}")
            self.tracer.add_event(span_id, "authentication_failed")
            self.tracer.end_span(span_id)
            return
        
        try:
            # Process message based on content type
            if message.content_type == ContentType.JSON:
                self._process_json_message(message)
            else:
                print(f"Agent {self.id} received message with unsupported content type: {message.content_type}")
            
            # End tracing span
            self.tracer.add_event(span_id, "message_processed")
            self.tracer.end_span(span_id)
            
        except Exception as e:
            # Record error and end span
            print(f"Error processing message in agent {self.id}: {e}")
            self.tracer.add_event(span_id, "error", {"error": str(e)})
            self.tracer.end_span(span_id)
            self.metrics.record_error("message_processing", self.id)
    
    def _process_json_message(self, message: AFPMessage):
        """
        Process a JSON message.
        
        Args:
            message: The message to process
        """
        content = message.content
        
        # Check if this is a task-related message
        if isinstance(content, dict) and "message_type" in content:
            if content["message_type"] == "task_assignment":
                self._handle_task_assignment(content["task"], message.sender)
            elif content["message_type"] == "task_result":
                self._handle_task_result(content["task_id"], content["results"], message.sender)
            elif content["message_type"] == "status_request":
                self._handle_status_request(message)
    
    def _handle_task_assignment(self, task_dict: Dict[str, Any], sender: str):
        """
        Handle a task assignment.
        
        Args:
            task_dict: Dictionary representation of the task
            sender: ID of the agent that sent the assignment
        """
        task = Task.from_dict(task_dict)
        print(f"Agent {self.id} received task assignment: {task.id} from {sender}")
        
        # Store the task
        self.tasks[task.id] = task
        task.status = "assigned"
        task.assigned_to = self.id
    
    def _handle_task_result(self, task_id: str, results: Any, sender: str):
        """
        Handle task results.
        
        Args:
            task_id: ID of the completed task
            results: Results of the task
            sender: ID of the agent that completed the task
        """
        print(f"Agent {self.id} received results for task {task_id} from {sender}")
        
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = "completed"
            task.completed_at = time.time()
            task.results = results
    
    def _handle_status_request(self, message: AFPMessage):
        """
        Handle a status request.
        
        Args:
            message: The status request message
        """
        # Create a status report
        status = {
            "agent_id": self.id,
            "role": self.role.value,
            "task_count": len(self.tasks),
            "active_tasks": [t.id for t in self.tasks.values() if t.status in ("assigned", "in_progress")]
        }
        
        # Send response
        response = message.create_response(
            content={"message_type": "status_response", "status": status},
            content_type=ContentType.JSON
        )
        
        # Sign the response
        signed_response = self.authenticator.sign_message(response, self.id, self.secret_key)
        
        # Send the response
        self.bus.send_message(signed_response)
    
    def send_message(self, recipient: str, content: Dict[str, Any], sync: bool = False, timeout: float = 5.0) -> Optional[AFPMessage]:
        """
        Send a message to another agent.
        
        Args:
            recipient: ID of the recipient agent
            content: Message content
            sync: Whether to wait for a response
            timeout: Timeout for sync requests in seconds
            
        Returns:
            Response message if sync=True, otherwise None
        """
        # Create message
        message = AFPMessage(
            sender=self.id,
            recipients=[recipient],
            content_type=ContentType.JSON,
            content=content,
            metadata={"requires_response": sync}
        )
        
        # Sign the message
        signed_message = self.authenticator.sign_message(message, self.id, self.secret_key)
        
        # Trace message sending
        span_id = self.tracer.trace_message_send(
            message.message_id,
            self.id,
            [recipient],
            ContentType.JSON.name,
            len(signed_message.serialize())
        )
        
        # Record metrics
        self.metrics.record_message_sent(len(signed_message.serialize()), self.id)
        
        try:
            # Use circuit breaker for sending messages
            circuit_breaker = self.circuit_breakers.get_or_create(f"send_to_{recipient}")
            
            def send_operation():
                if sync:
                    return self.bus.send_message_sync(signed_message, timeout=timeout)
                else:
                    return self.bus.send_message(signed_message)
            
            def fallback_operation():
                print(f"Circuit open for {recipient}, using fallback")
                return None
            
            result = circuit_breaker.execute(send_operation, fallback_operation)
            
            # End tracing span
            self.tracer.add_event(span_id, "message_sent")
            self.tracer.end_span(span_id)
            
            return result
            
        except Exception as e:
            # Record error and end span
            print(f"Error sending message from agent {self.id} to {recipient}: {e}")
            self.tracer.add_event(span_id, "error", {"error": str(e)})
            self.tracer.end_span(span_id)
            self.metrics.record_error("message_sending", self.id)
            return None
    
    def broadcast_message(self, content: Dict[str, Any]):
        """
        Broadcast a message to all agents.
        
        Args:
            content: Message content
        """
        # Create broadcast message
        message = AFPMessage(
            sender=self.id,
            recipients=["*"],
            content_type=ContentType.JSON,
            content=content,
            metadata={"broadcast": True}
        )
        
        # Sign the message
        signed_message = self.authenticator.sign_message(message, self.id, self.secret_key)
        
        # Send the message
        self.bus.send_message(signed_message)
        
        # Record metrics
        self.metrics.record_message_sent(len(signed_message.serialize()), self.id)


class CoordinatorAgent(Agent):
    """Coordinator agent that assigns tasks to other agents."""
    
    def __init__(self, agent_id: str, bus: AFPCommunicationBus, authenticator: HMACAuthenticator):
        """Initialize the coordinator agent."""
        super().__init__(agent_id, AgentRole.COORDINATOR, bus, authenticator)
        self.available_agents: Dict[str, AgentRole] = {}  # Agent ID -> Role
        self.pending_tasks: List[Task] = []
    
    def register_agent(self, agent_id: str, role: AgentRole):
        """
        Register an agent with the coordinator.
        
        Args:
            agent_id: ID of the agent to register
            role: Role of the agent
        """
        self.available_agents[agent_id] = role
        print(f"Coordinator {self.id} registered agent {agent_id} with role {role.value}")
    
    def create_task(self, task_type: str, data: Dict[str, Any], priority: int = 1) -> Task:
        """
        Create a new task.
        
        Args:
            task_type: Type of task
            data: Task data
            priority: Task priority
            
        Returns:
            The created task
        """
        task = Task(task_type, data, priority)
        self.pending_tasks.append(task)
        self.tasks[task.id] = task
        print(f"Coordinator {self.id} created task {task.id} of type {task_type}")
        return task
    
    def assign_task(self, task: Task, agent_id: str) -> bool:
        """
        Assign a task to a specific agent.
        
        Args:
            task: The task to assign
            agent_id: ID of the agent to assign to
            
        Returns:
            True if assignment was successful, False otherwise
        """
        if agent_id not in self.available_agents:
            print(f"Coordinator {self.id} cannot assign task to unknown agent {agent_id}")
            return False
        
        # Update task status
        task.status = "assigning"
        
        # Send task assignment message
        response = self.send_message(
            recipient=agent_id,
            content={
                "message_type": "task_assignment",
                "task": task.to_dict()
            },
            sync=True
        )
        
        if response:
            # Task was accepted
            task.status = "assigned"
            task.assigned_to = agent_id
            
            # Remove from pending tasks if it's there
            if task in self.pending_tasks:
                self.pending_tasks.remove(task)
                
            print(f"Coordinator {self.id} assigned task {task.id} to agent {agent_id}")
            return True
        else:
            # Assignment failed
            task.status = "pending"
            print(f"Coordinator {self.id} failed to assign task {task.id} to agent {agent_id}")
            return False
    
    def _processing_loop(self):
        """
        Main processing loop for the coordinator.
        
        This loop assigns pending tasks to available agents based on their roles.
        """
        while self.running:
            # Sort pending tasks by priority (highest first)
            self.pending_tasks.sort(key=lambda t: t.priority, reverse=True)
            
            # Try to assign each pending task
            for task in list(self.pending_tasks):
                # Find suitable agents for this task type
                suitable_agents = []
                
                if task.type == "data_processing":
                    # Find processor agents
                    suitable_agents = [agent_id for agent_id, role in self.available_agents.items() 
                                      if role == AgentRole.PROCESSOR]
                elif task.type == "data_storage":
                    # Find storage agents
                    suitable_agents = [agent_id for agent_id, role in self.available_agents.items() 
                                      if role == AgentRole.STORAGE]
                elif task.type == "data_analysis":
                    # Find analyzer agents
                    suitable_agents = [agent_id for agent_id, role in self.available_agents.items() 
                                      if role == AgentRole.ANALYZER]
                
                if suitable_agents:
                    # Select a random agent (in a real system, this would be more sophisticated)
                    agent_id = random.choice(suitable_agents)
                    self.assign_task(task, agent_id)
            
            # Sleep before next assignment cycle
            time.sleep(1.0)
    
    def _process_json_message(self, message: AFPMessage):
        """
        Process JSON messages.
        
        Args:
            message: The message to process
        """
        super()._process_json_message(message)
        
        content = message.content
        
        # Check for coordinator-specific messages
        if isinstance(content, dict) and "message_type" in content:
            if content["message_type"] == "agent_registration":
                self.register_agent(message.sender, AgentRole(content["role"]))
            elif content["message_type"] == "task_result":
                # Update task status
                task_id = content["task_id"]
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    task.status = "completed"
                    task.completed_at = time.time()
                    task.results = content["results"]
                    print(f"Coordinator {self.id} received results for task {task_id}")


class ProcessorAgent(Agent):
    """Processor agent that processes data tasks."""
    
    def __init__(self, agent_id: str, bus: AFPCommunicationBus, authenticator: HMACAuthenticator):
        """Initialize the processor agent."""
        super().__init__(agent_id, AgentRole.PROCESSOR, bus, authenticator)
    
    def _processing_loop(self):
        """
        Main processing loop for the processor.
        
        This loop processes assigned tasks.
        """
        while self.running:
            # Find tasks that need processing
            for task_id, task in list(self.tasks.items()):
                if task.status == "assigned":
                    # Start processing
                    print(f"Processor {self.id} processing task {task_id}")
                    task.status = "in_progress"
                    
                    # Simulate processing time based on data size
                    data_size = len(json.dumps(task.data))
                    processing_time = 0.01 * (data_size / 100)  # 10ms per 100 bytes
                    time.sleep(processing_time)
                    
                    # Generate results
                    if task.type == "data_processing":
                        results = self._process_data(task.data)
                    else:
                        results = {"error": "Unsupported task type"}
                    
                    # Mark as completed
                    task.status = "completed"
                    task.completed_at = time.time()
                    task.results = results
                    
                    # Send results back to coordinator
                    self.send_message(
                        recipient=task.data.get("coordinator", "coordinator"),
                        content={
                            "message_type": "task_result",
                            "task_id": task_id,
                            "results": results
                        }
                    )
                    
                    print(f"Processor {self.id} completed task {task_id}")
            
            # Sleep before next processing cycle
            time.sleep(0.1)
    
    def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data.
        
        Args:
            data: Data to process
            
        Returns:
            Processing results
        """
        # Simple data processing example
        if "numbers" in data:
            numbers = data["numbers"]
            return {
                "sum": sum(numbers),
                "average": sum(numbers) / len(numbers) if numbers else 0,
                "min": min(numbers) if numbers else None,
                "max": max(numbers) if numbers else None
            }
        elif "text" in data:
            text = data["text"]
            words = text.split()
            return {
                "word_count": len(words),
                "char_count": len(text),
                "uppercase_count": sum(1 for c in text if c.isupper()),
                "lowercase_count": sum(1 for c in text if c.islower())
            }
        else:
            return {"error": "Unsupported data format"}


class StorageAgent(Agent):
    """Storage agent that stores and retrieves data."""
    
    def __init__(self, agent_id: str, bus: AFPCommunicationBus, authenticator: HMACAuthenticator):
        """Initialize the storage agent."""
        super().__init__(agent_id, AgentRole.STORAGE, bus, authenticator)
        self.stored_data: Dict[str, Any] = {}
    
    def _processing_loop(self):
        """
        Main processing loop for the storage agent.
        
        This loop processes storage and retrieval tasks.
        """
        while self.running:
            # Find tasks that need processing
            for task_id, task in list(self.tasks.items()):
                if task.status == "assigned":
                    # Start processing
                    print(f"Storage {self.id} processing task {task_id}")
                    task.status = "in_progress"
                    
                    # Generate results based on operation
                    if task.type == "data_storage":
                        operation = task.data.get("operation")
                        if operation == "store":
                            results = self._store_data(task.data)
                        elif operation == "retrieve":
                            results = self._retrieve_data(task.data)
                        else:
                            results = {"error": "Unsupported storage operation"}
                    else:
                        results = {"error": "Unsupported task type"}
                    
                    # Mark as completed
                    task.status = "completed"
                    task.completed_at = time.time()
                    task.results = results
                    
                    # Send results back to coordinator
                    self.send_message(
                        recipient=task.data.get("coordinator", "coordinator"),
                        content={
                            "message_type": "task_result",
                            "task_id": task_id,
                            "results": results
                        }
                    )
                    
                    print(f"Storage {self.id} completed task {task_id}")
            
            # Sleep before next processing cycle
            time.sleep(0.1)
    
    def _store_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store data.
        
        Args:
            data: Data containing storage information
            
        Returns:
            Storage results
        """
        if "key" in data and "value" in data:
            key = data["key"]
            value = data["value"]
            self.stored_data[key] = value
            return {"status": "stored", "key": key}
        else:
            return {"error": "Missing key or value for storage"}
    
    def _retrieve_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve data.
        
        Args:
            data: Data containing retrieval information
            
        Returns:
            Retrieval results
        """
        if "key" in data:
            key = data["key"]
            if key in self.stored_data:
                return {"status": "retrieved", "key": key, "value": self.stored_data[key]}
            else:
                return {"error": f"Key not found: {key}"}
        else:
            return {"error": "Missing key for retrieval"}


class AnalyzerAgent(Agent):
    """Analyzer agent that analyzes data."""
    
    def __init__(self, agent_id: str, bus: AFPCommunicationBus, authenticator: HMACAuthenticator):
        """Initialize the analyzer agent."""
        super().__init__(agent_id, AgentRole.ANALYZER, bus, authenticator)
    
    def _processing_loop(self):
        """
        Main processing loop for the analyzer.
        
        This loop processes analysis tasks.
        """
        while self.running:
            # Find tasks that need processing
            for task_id, task in list(self.tasks.items()):
                if task.status == "assigned":
                    # Start processing
                    print(f"Analyzer {self.id} processing task {task_id}")
                    task.status = "in_progress"
                    
                    # Simulate processing time based on analysis complexity
                    complexity = task.data.get("complexity", 1)
                    processing_time = 0.1 * complexity  # 100ms per complexity unit
                    time.sleep(processing_time)
                    
                    # Generate results
                    if task.type == "data_analysis":
                        results = self._analyze_data(task.data)
                    else:
                        results = {"error": "Unsupported task type"}
                    
                    # Mark as completed
                    task.status = "completed"
                    task.completed_at = time.time()
                    task.results = results
                    
                    # Send results back to coordinator
                    self.send_message(
                        recipient=task.data.get("coordinator", "coordinator"),
                        content={
                            "message_type": "task_result",
                            "task_id": task_id,
                            "results": results
                        }
                    )
                    
                    print(f"Analyzer {self.id} completed task {task_id}")
            
            # Sleep before next processing cycle
            time.sleep(0.1)
    
    def _analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Analysis results
        """
        # Simple analysis example
        if "dataset" in data:
            dataset = data["dataset"]
            if isinstance(dataset, list):
                # Numeric dataset analysis
                if all(isinstance(x, (int, float)) for x in dataset):
                    return {
                        "mean": sum(dataset) / len(dataset) if dataset else 0,
                        "median": sorted(dataset)[len(dataset)//2] if dataset else None,
                        "variance": sum((x - (sum(dataset) / len(dataset)))**2 for x in dataset) / len(dataset) if dataset else 0,
                        "range": max(dataset) - min(dataset) if dataset else 0
                    }
                # Text dataset analysis
                elif all(isinstance(x, str) for x in dataset):
                    total_length = sum(len(x) for x in dataset)
                    word_counts = [len(x.split()) for x in dataset]
                    return {
                        "total_items": len(dataset),
                        "avg_length": total_length / len(dataset) if dataset else 0,
                        "avg_words": sum(word_counts) / len(dataset) if dataset else 0,
                        "longest_item": max(dataset, key=len) if dataset else None
                    }
            return {"error": "Unsupported dataset type"}
        else:
            return {"error": "No dataset provided for analysis"}


def run_example_application():
    """Run the example AFP application with multiple agents."""
    # Create communication bus
    bus = AFPCommunicationBus()
    
    # Create authenticator
    authenticator = HMACAuthenticator()
    
    # Create agents
    coordinator = CoordinatorAgent("coordinator", bus, authenticator)
    processor1 = ProcessorAgent("processor1", bus, authenticator)
    processor2 = ProcessorAgent("processor2", bus, authenticator)
    storage = StorageAgent("storage", bus, authenticator)
    analyzer = AnalyzerAgent("analyzer", bus, authenticator)
    
    try:
        # Start all agents
        coordinator.start()
        processor1.start()
        processor2.start()
        storage.start()
        analyzer.start()
        
        # Register agents with coordinator (in a real system, agents would register themselves)
        print("\nRegistering agents with coordinator...")
        time.sleep(1)  # Wait for agents to start
        
        # Register agents with coordinator
        processor1.send_message(
            recipient=coordinator.id,
            content={"message_type": "agent_registration", "role": AgentRole.PROCESSOR.value},
            sync=True
        )
        processor2.send_message(
            recipient=coordinator.id,
            content={"message_type": "agent_registration", "role": AgentRole.PROCESSOR.value},
            sync=True
        )
        storage.send_message(
            recipient=coordinator.id,
            content={"message_type": "agent_registration", "role": AgentRole.STORAGE.value},
            sync=True
        )
        analyzer.send_message(
            recipient=coordinator.id,
            content={"message_type": "agent_registration", "role": AgentRole.ANALYZER.value},
            sync=True
        )
        
        print("Agents registered with coordinator")
        time.sleep(1)  # Give time for registrations to process
        
        # Create and run tasks
        print("\nCreating and running tasks...")
        
        # Create data processing task
        process_task = coordinator.create_task(
            task_type="data_processing",
            data={
                "coordinator": coordinator.id,
                "numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            },
            priority=2
        )
        
        # Create data storage task
        storage_task = coordinator.create_task(
            task_type="data_storage",
            data={
                "coordinator": coordinator.id,
                "operation": "store",
                "key": "test_data",
                "value": {"name": "Test Data", "values": [10, 20, 30]}
            }
        )
        
        # Create data retrieval task
        retrieval_task = coordinator.create_task(
            task_type="data_storage",
            data={
                "coordinator": coordinator.id,
                "operation": "retrieve",
                "key": "test_data"
            }
        )
        
        # Create data analysis task
        analysis_task = coordinator.create_task(
            task_type="data_analysis",
            data={
                "coordinator": coordinator.id,
                "dataset": [12, 15, 18, 22, 30, 35, 42],
                "complexity": 2
            },
            priority=3
        )
        
        # Wait for tasks to complete
        print("\nWaiting for tasks to complete...")
        time.sleep(30)  # Increase wait time to ensure tasks complete
        
        # Print task results
        print("\nTask Results:")
        print(f"Process Task: {process_task.results}")
        print(f"Storage Task: {storage_task.results}")
        print(f"Retrieval Task: {retrieval_task.results}")
        print(f"Analysis Task: {analysis_task.results}")
        
        # Print metrics
        print("\nPerformance Metrics:")
        print(f"Coordinator messages sent: {coordinator.metrics.metrics.get_counter('afp.messages.sent')}")
        print(f"Processor1 messages received: {processor1.metrics.metrics.get_counter('afp.messages.received')}")
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    
    finally:
        # Stop all agents
        print("\nStopping agents...")
        coordinator.stop()
        processor1.stop()
        processor2.stop()
        storage.stop()
        analyzer.stop()
        
        # Shutdown communication bus
        bus.shutdown()
        
        print("\nApplication shutdown complete")


if __name__ == "__main__":
    run_example_application() 