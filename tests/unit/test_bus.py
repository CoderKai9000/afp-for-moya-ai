"""
Tests for the AFP communication bus.
"""

import unittest
import time
import threading
import sys
import os
from queue import Queue
from typing import List

# Add the current directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath('.'))

from moya.communication.afp.message import AFPMessage, ContentType
from moya.communication.afp.bus import AFPCommunicationBus
from moya.communication.afp.exceptions import (
    AFPMessageError, AFPRoutingError, AFPSubscriptionError, AFPTimeoutError
)


class TestAFPCommunicationBus(unittest.TestCase):
    """Tests for the AFPCommunicationBus class."""
    
    def setUp(self):
        """Set up a fresh communication bus for each test."""
        self.bus = AFPCommunicationBus()
        
        # Register some test agents
        self.bus.register_agent("agent1")
        self.bus.register_agent("agent2")
        self.bus.register_agent("agent3")
    
    def tearDown(self):
        """Clean up after each test."""
        self.bus.shutdown()
    
    def test_agent_registration(self):
        """Test agent registration and unregistration."""
        # Test registering a new agent
        self.assertTrue(self.bus.register_agent("agent4"))
        self.assertEqual(self.bus.get_agent_count(), 4)
        
        # Test registering an existing agent (should fail)
        self.assertFalse(self.bus.register_agent("agent1"))
        self.assertEqual(self.bus.get_agent_count(), 4)
        
        # Test unregistering an agent
        self.assertTrue(self.bus.unregister_agent("agent1"))
        self.assertEqual(self.bus.get_agent_count(), 3)
        
        # Test unregistering a non-existent agent (should fail)
        self.assertFalse(self.bus.unregister_agent("agent99"))
        self.assertEqual(self.bus.get_agent_count(), 3)
    
    def test_subscription(self):
        """Test subscription creation and removal."""
        # Create a test callback
        def callback(message):
            pass
        
        # Subscribe to messages
        sub_id = self.bus.subscribe(
            subscriber="agent1",
            callback=callback,
            pattern={"sender": "agent2"}
        )
        
        self.assertIsNotNone(sub_id)
        self.assertEqual(self.bus.get_subscription_count(), 2)  # +1 for internal response handler
        
        # Unsubscribe
        self.assertTrue(self.bus.unsubscribe(sub_id))
        self.assertEqual(self.bus.get_subscription_count(), 1)  # Only internal response handler left
        
        # Test unsubscribing a non-existent subscription (should fail)
        self.assertFalse(self.bus.unsubscribe("non_existent_id"))
    
    def test_subscription_validation(self):
        """Test subscription validation."""
        def callback(message):
            pass
        
        # Test subscribing with an unregistered agent (should fail)
        with self.assertRaises(AFPSubscriptionError):
            self.bus.subscribe(
                subscriber="unregistered_agent",
                callback=callback,
                pattern={"sender": "agent1"}
            )
    
    def test_async_message_delivery(self):
        """Test asynchronous message delivery."""
        # Create a queue to track received messages
        received_messages = Queue()
        
        # Create a callback that adds messages to the queue
        def callback(message):
            received_messages.put(message)
        
        # Subscribe agent2 to messages from agent1
        self.bus.subscribe(
            subscriber="agent2",
            callback=callback,
            pattern={"sender": "agent1"}
        )
        
        # Create and send a message
        message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="Hello, agent2!"
        )
        
        self.bus.send_message(message)
        
        # Wait for message delivery (may take a moment due to threading)
        time.sleep(0.1)
        
        # Check that the message was received
        self.assertFalse(received_messages.empty())
        received = received_messages.get(block=False)
        self.assertEqual(received.sender, "agent1")
        self.assertEqual(received.content, "Hello, agent2!")
    
    def test_sync_message_delivery(self):
        """Test synchronous message delivery and response."""
        # Create a callback that sends a response
        def callback(message):
            # Create and send a response
            response = message.create_response(
                content="Hello, agent1! I got your message."
            )
            self.bus.send_message(response)
        
        # Subscribe agent2 to messages from agent1
        self.bus.subscribe(
            subscriber="agent2",
            callback=callback,
            pattern={"sender": "agent1"}
        )
        
        # Create and send a message synchronously
        message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="Hello, agent2! Please respond."
        )
        
        response = self.bus.send_message_sync(message, timeout=1.0)
        
        # Check the response
        self.assertEqual(response.sender, "agent2")
        self.assertEqual(response.recipients, ["agent1"])
        self.assertEqual(response.content, "Hello, agent1! I got your message.")
        self.assertEqual(response.parent_message_id, message.message_id)
    
    def test_sync_timeout(self):
        """Test timeout for synchronous messages."""
        # Create a callback that doesn't respond
        def callback(message):
            pass  # No response
        
        # Subscribe agent2 to messages from agent1
        self.bus.subscribe(
            subscriber="agent2",
            callback=callback,
            pattern={"sender": "agent1"}
        )
        
        # Create and send a message synchronously with a short timeout
        message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="Hello, agent2! Please respond."
        )
        
        # This should timeout
        with self.assertRaises(AFPTimeoutError):
            self.bus.send_message_sync(message, timeout=0.1)
    
    def test_broadcast_message(self):
        """Test broadcast message delivery."""
        # Create queues to track received messages
        agent2_messages = []
        agent3_messages = []
        
        # Create callbacks
        def agent2_callback(message):
            agent2_messages.append(message)
        
        def agent3_callback(message):
            agent3_messages.append(message)
        
        # Subscribe agents to all messages
        self.bus.subscribe(
            subscriber="agent2",
            callback=agent2_callback,
            pattern={}  # Empty pattern matches everything
        )
        
        self.bus.subscribe(
            subscriber="agent3",
            callback=agent3_callback,
            pattern={}
        )
        
        # Create and send a broadcast message
        broadcast = AFPMessage(
            sender="agent1",
            recipients=["*"],  # Broadcast
            content_type=ContentType.TEXT,
            content="Attention all agents!"
        )
        
        self.bus.send_message(broadcast)
        
        # Wait for message delivery
        time.sleep(0.1)
        
        # Check that both agents received the message
        self.assertEqual(len(agent2_messages), 1)
        self.assertEqual(len(agent3_messages), 1)
        self.assertEqual(agent2_messages[0].content, "Attention all agents!")
        self.assertEqual(agent3_messages[0].content, "Attention all agents!")
    
    def test_message_validation(self):
        """Test message validation."""
        # Test sending a message with an empty sender
        with self.assertRaises(AFPMessageError):
            self.bus.send_message(AFPMessage(
                sender="",
                recipients=["agent2"],
                content_type=ContentType.TEXT,
                content="Invalid message"
            ))
        
        # Test sending a message with no recipients
        with self.assertRaises(AFPMessageError):
            self.bus.send_message(AFPMessage(
                sender="agent1",
                recipients=[],
                content_type=ContentType.TEXT,
                content="Invalid message"
            ))
        
        # Test sending a message from an unregistered agent
        with self.assertRaises(AFPRoutingError):
            self.bus.send_message(AFPMessage(
                sender="unregistered_agent",
                recipients=["agent2"],
                content_type=ContentType.TEXT,
                content="Invalid message"
            ))
        
        # Test sending a message to an unregistered agent
        with self.assertRaises(AFPRoutingError):
            self.bus.send_message(AFPMessage(
                sender="agent1",
                recipients=["unregistered_agent"],
                content_type=ContentType.TEXT,
                content="Invalid message"
            ))
    
    def test_unregister_with_subscriptions(self):
        """Test unregistering an agent with active subscriptions."""
        # Create a callback
        def callback(message):
            pass
        
        # Subscribe agent1 to messages
        self.bus.subscribe(
            subscriber="agent1",
            callback=callback,
            pattern={"sender": "agent2"}
        )
        
        # Check subscription count
        self.assertEqual(self.bus.get_subscription_count(), 2)  # +1 for internal response handler
        
        # Unregister agent1
        self.assertTrue(self.bus.unregister_agent("agent1"))
        
        # Check that subscriptions were removed
        self.assertEqual(self.bus.get_subscription_count(), 1)  # Only internal response handler left


if __name__ == "__main__":
    unittest.main(verbosity=2) 