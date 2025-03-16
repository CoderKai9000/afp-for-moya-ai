"""
Tests for the AFP subscription module.
"""

import unittest
import re
import sys
import os

# Add the current directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath('.'))

from moya.communication.afp.message import AFPMessage, ContentType
from moya.communication.afp.subscription import AFPSubscription
from moya.communication.afp.exceptions import AFPSubscriptionError


class TestAFPSubscription(unittest.TestCase):
    """Tests for the AFPSubscription class."""
    
    def test_basic_subscription(self):
        """Test basic subscription creation and matching."""
        # Create a callback function that tracks calls
        self.callback_called = False
        self.delivered_message = None
        
        def callback(message):
            self.callback_called = True
            self.delivered_message = message
        
        # Create a subscription for messages from agent1
        subscription = AFPSubscription(
            subscriber="agent2",
            callback=callback,
            pattern={"sender": "agent1"}
        )
        
        # Create a matching message
        message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="Hello from agent1"
        )
        
        # Create a non-matching message
        non_matching = AFPMessage(
            sender="agent3",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="Hello from agent3"
        )
        
        # Test matching
        self.assertTrue(subscription.matches(message))
        self.assertFalse(subscription.matches(non_matching))
        
        # Test delivery
        self.assertTrue(subscription.deliver(message))
        self.assertTrue(self.callback_called)
        self.assertEqual(self.delivered_message, message)
        
        # Reset and test non-matching delivery
        self.callback_called = False
        self.delivered_message = None
        self.assertFalse(subscription.deliver(non_matching))
        self.assertFalse(self.callback_called)
        self.assertIsNone(self.delivered_message)
    
    def test_content_type_matching(self):
        """Test matching based on content type."""
        def callback(message):
            pass
        
        # Subscribe to JSON messages
        subscription = AFPSubscription(
            subscriber="agent2",
            callback=callback,
            pattern={"content_type": "JSON"}
        )
        
        # Create messages with different content types
        json_message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.JSON,
            content={"key": "value"}
        )
        
        text_message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="Plain text"
        )
        
        # Test matching
        self.assertTrue(subscription.matches(json_message))
        self.assertFalse(subscription.matches(text_message))
    
    def test_metadata_matching(self):
        """Test matching based on metadata."""
        def callback(message):
            pass
        
        # Subscribe to messages with specific metadata
        subscription = AFPSubscription(
            subscriber="agent2",
            callback=callback,
            pattern={"metadata": {"priority": "high"}}
        )
        
        # Create messages with different metadata
        high_priority = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="Important message",
            metadata={"priority": "high", "category": "alert"}
        )
        
        low_priority = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="Regular message",
            metadata={"priority": "low", "category": "info"}
        )
        
        # Test matching
        self.assertTrue(subscription.matches(high_priority))
        self.assertFalse(subscription.matches(low_priority))
    
    def test_regex_matching(self):
        """Test matching based on regular expressions."""
        def callback(message):
            pass
        
        # Subscribe to messages with content matching a pattern (case-insensitive)
        pattern = re.compile(r"urgent|emergency", re.IGNORECASE)
        print(f"\nRegex pattern: {pattern.pattern} (case-insensitive)")
        
        subscription = AFPSubscription(
            subscriber="agent2",
            callback=callback,
            regex_patterns={"content": pattern}
        )
        
        # Create messages with different content
        urgent_message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="This is an urgent message"
        )
        print(f"Urgent message content: '{urgent_message.content}'")
        print(f"Pattern search result: {pattern.search(urgent_message.content)}")
        
        emergency_message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="Emergency notification"
        )
        print(f"Emergency message content: '{emergency_message.content}'")
        print(f"Pattern search result: {pattern.search(emergency_message.content)}")
        
        regular_message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type=ContentType.TEXT,
            content="Regular update"
        )
        print(f"Regular message content: '{regular_message.content}'")
        print(f"Pattern search result: {pattern.search(regular_message.content)}")
        
        # Test matching
        urgent_match = subscription.matches(urgent_message)
        print(f"Urgent message matches: {urgent_match}")
        self.assertTrue(urgent_match)
        
        emergency_match = subscription.matches(emergency_message)
        print(f"Emergency message matches: {emergency_match}")
        self.assertTrue(emergency_match)
        
        regular_match = subscription.matches(regular_message)
        print(f"Regular message matches: {regular_match}")
        self.assertFalse(regular_match)
    
    def test_validation(self):
        """Test subscription validation."""
        def callback(message):
            pass
        
        # Test invalid pattern field
        with self.assertRaises(AFPSubscriptionError):
            AFPSubscription(
                subscriber="agent2",
                callback=callback,
                pattern={"invalid_field": "value"}
            )
        
        # Test invalid regex pattern field
        with self.assertRaises(AFPSubscriptionError):
            AFPSubscription(
                subscriber="agent2",
                callback=callback,
                regex_patterns={"invalid_field": re.compile(r".*")}
            )
        
        # Test missing subscriber
        with self.assertRaises(AFPSubscriptionError):
            AFPSubscription(
                subscriber="",
                callback=callback,
                pattern={"sender": "agent1"}
            )
        
        # Test invalid callback
        with self.assertRaises(AFPSubscriptionError):
            AFPSubscription(
                subscriber="agent2",
                callback="not_callable",
                pattern={"sender": "agent1"}
            )


if __name__ == "__main__":
    unittest.main(verbosity=2) 