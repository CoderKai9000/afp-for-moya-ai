"""
Tests for the AFP message module.
"""

import unittest
import time
import json
from moya.communication.afp.message import AFPMessage, ContentType


class TestAFPMessage(unittest.TestCase):
    """Tests for the AFPMessage class."""
    
    def test_basic_creation(self):
        """Test basic message creation."""
        message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="Hello, agent2!"
        )
        
        self.assertEqual(message.sender, "agent1")
        self.assertEqual(message.recipients, ["agent2"])
        self.assertEqual(message.content_type, ContentType.TEXT)
        self.assertEqual(message.content, "Hello, agent2!")
        self.assertIsNotNone(message.message_id)
        self.assertIsNone(message.parent_message_id)
        self.assertIsNotNone(message.timestamp)
        self.assertIsNone(message.ttl)
        self.assertEqual(message.trace_path, ["agent1"])
        
    def test_broadcast_detection(self):
        """Test broadcast message detection."""
        broadcast_msg = AFPMessage(
            sender="agent1",
            recipients=["*"],
            content_type=ContentType.TEXT,
            content="Attention all agents!"
        )
        
        direct_msg = AFPMessage(
            sender="agent1",
            recipients=["agent2", "agent3"],
            content_type=ContentType.TEXT,
            content="Attention specific agents!"
        )
        
        self.assertTrue(broadcast_msg.is_broadcast())
        self.assertFalse(direct_msg.is_broadcast())
        
    def test_expiration(self):
        """Test message expiration."""
        # Message with short TTL
        expiring_msg = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="This will expire soon",
            ttl=0.1  # 100ms TTL
        )
        
        # Message with no TTL
        permanent_msg = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="This won't expire"
        )
        
        self.assertFalse(expiring_msg.has_expired())  # Not expired immediately
        time.sleep(0.2)  # Wait for expiration
        self.assertTrue(expiring_msg.has_expired())  # Should be expired now
        self.assertFalse(permanent_msg.has_expired())  # Never expires
        
    def test_response_creation(self):
        """Test creating a response to a message."""
        original = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="Hello, agent2!"
        )
        
        response = original.create_response(
            content="Hello, agent1!",
            metadata={"reply_type": "greeting"}
        )
        
        self.assertEqual(response.sender, "agent2")
        self.assertEqual(response.recipients, ["agent1"])
        self.assertEqual(response.content_type, ContentType.TEXT)
        self.assertEqual(response.content, "Hello, agent1!")
        self.assertEqual(response.metadata, {"reply_type": "greeting"})
        self.assertEqual(response.parent_message_id, original.message_id)
        self.assertEqual(response.trace_path, ["agent1", "agent2"])
        
    def test_serialization(self):
        """Test message serialization and deserialization."""
        original = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.JSON,
            content={"key": "value"},
            metadata={"importance": "high"}
        )
        
        # Test to_dict and from_dict
        message_dict = original.to_dict()
        self.assertEqual(message_dict["sender"], "agent1")
        self.assertEqual(message_dict["content_type"], "JSON")
        
        restored_from_dict = AFPMessage.from_dict(message_dict)
        self.assertEqual(restored_from_dict.sender, original.sender)
        self.assertEqual(restored_from_dict.content_type, original.content_type)
        
        # Test serialize and deserialize
        serialized = original.serialize()
        self.assertIsInstance(serialized, str)
        
        deserialized = AFPMessage.deserialize(serialized)
        self.assertEqual(deserialized.sender, original.sender)
        self.assertEqual(deserialized.content_type, original.content_type)
        self.assertEqual(deserialized.content, original.content)


if __name__ == "__main__":
    unittest.main(verbosity=2) 