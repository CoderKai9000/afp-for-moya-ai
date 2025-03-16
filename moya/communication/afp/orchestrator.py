class AFPOrchestrator:
    """
    Orchestration layer for AFP agents.
    
    Provides message routing between agents, handles agent registration,
    and manages communication patterns.
    """
    
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
        
        # Subscribe to messages intended for the orchestrator
        self.bus.subscribe(
            self.id,
            lambda msg: self.id in msg.recipients or not msg.recipients
        )
    
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
            List of message IDs for delivered messages
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
        delivered_msgs = []
        
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
            recipient_msg.metadata["routed_by"] = self.id
            if workflow_id:
                recipient_msg.metadata["workflow_id"] = workflow_id
            
            # Use send_message_sync for immediate delivery
            self.bus.send_message_sync(recipient_msg)
            delivered_msgs.append(recipient_msg.message_id)
        
        return delivered_msgs 