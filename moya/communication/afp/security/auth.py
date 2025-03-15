"""
Authentication module for Agent Flow Protocol.

Provides authentication mechanisms for verifying agent identities
and securing communications between agents.
"""

import hmac
import hashlib
import time
import secrets
import base64
from typing import Dict, Optional, Tuple, List, Any, Callable

from ..exceptions import AFPSecurityError
from ..message import AFPMessage


class AFPAuthenticator:
    """
    Base authenticator class for AFP.
    
    Provides the interface for authentication mechanisms.
    """
    
    def authenticate_agent(self, agent_id: str, credentials: Any) -> bool:
        """
        Authenticate an agent based on provided credentials.
        
        Args:
            agent_id: The ID of the agent to authenticate
            credentials: Authentication credentials (implementation-specific)
            
        Returns:
            True if authentication is successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement authenticate_agent")
    
    def sign_message(self, message: AFPMessage, agent_id: str, key: Any) -> AFPMessage:
        """
        Sign a message to prove authenticity.
        
        Args:
            message: The message to sign
            agent_id: The ID of the agent signing the message
            key: The signing key
            
        Returns:
            The signed message (with signature in metadata)
        """
        raise NotImplementedError("Subclasses must implement sign_message")
    
    def verify_message(self, message: AFPMessage) -> bool:
        """
        Verify the authenticity of a signed message.
        
        Args:
            message: The message to verify
            
        Returns:
            True if the message signature is valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement verify_message")


class HMACAuthenticator(AFPAuthenticator):
    """
    HMAC-based authenticator for AFP.
    
    Uses HMAC with SHA-256 for message signing and verification.
    """
    
    def __init__(self):
        """Initialize the HMAC authenticator."""
        # Agent credentials (agent_id -> secret key)
        self._credentials: Dict[str, str] = {}
        
        # Nonce tracking to prevent replay attacks
        self._used_nonces: Dict[str, List[str]] = {}
        
        # Maximum age of nonces to track (in seconds)
        self._nonce_ttl = 3600  # 1 hour
    
    def register_agent(self, agent_id: str, secret_key: Optional[str] = None) -> str:
        """
        Register an agent with the authenticator.
        
        Args:
            agent_id: The ID of the agent to register
            secret_key: Optional secret key (generated if not provided)
            
        Returns:
            The secret key for the agent
        """
        if agent_id in self._credentials:
            raise AFPSecurityError(f"Agent {agent_id} is already registered")
        
        # Generate a secret key if not provided
        if not secret_key:
            secret_key = secrets.token_hex(32)
        
        self._credentials[agent_id] = secret_key
        self._used_nonces[agent_id] = []
        
        return secret_key
    
    def authenticate_agent(self, agent_id: str, secret_key: str) -> bool:
        """
        Authenticate an agent based on secret key.
        
        Args:
            agent_id: The ID of the agent to authenticate
            secret_key: The secret key for authentication
            
        Returns:
            True if authentication is successful, False otherwise
        """
        if agent_id not in self._credentials:
            return False
        
        return self._credentials[agent_id] == secret_key
    
    def sign_message(self, message: AFPMessage, agent_id: str, secret_key: str) -> AFPMessage:
        """
        Sign a message using HMAC-SHA256.
        
        Args:
            message: The message to sign
            agent_id: The ID of the agent signing the message
            secret_key: The secret key for signing
            
        Returns:
            The signed message (with signature in metadata)
            
        Raises:
            AFPSecurityError: If the agent is not the sender of the message
        """
        if message.sender != agent_id:
            raise AFPSecurityError("Only the sender can sign a message")
        
        # Create a copy of the message to avoid modifying the original
        message_dict = message.to_dict()
        
        # Add a nonce and timestamp to prevent replay attacks
        nonce = secrets.token_hex(16)
        timestamp = str(int(time.time()))
        
        # Add authentication metadata
        if "auth" not in message_dict["metadata"]:
            message_dict["metadata"]["auth"] = {}
        
        message_dict["metadata"]["auth"].update({
            "nonce": nonce,
            "timestamp": timestamp,
            "agent_id": agent_id
        })
        
        # Remove existing signature if present
        if "signature" in message_dict["metadata"]["auth"]:
            del message_dict["metadata"]["auth"]["signature"]
        
        # Create a canonical representation for signing
        canonical = self._create_canonical_representation(message_dict)
        
        # Create HMAC signature
        signature = hmac.new(
            secret_key.encode(),
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Add signature to metadata
        message_dict["metadata"]["auth"]["signature"] = signature
        
        # Create a new message with the signature
        return AFPMessage.from_dict(message_dict)
    
    def verify_message(self, message: AFPMessage) -> bool:
        """
        Verify the authenticity of a signed message.
        
        Args:
            message: The message to verify
            
        Returns:
            True if the message signature is valid, False otherwise
        """
        # Check if message has authentication metadata
        if "auth" not in message.metadata:
            return False
        
        auth = message.metadata["auth"]
        
        # Check for required authentication fields
        required_fields = ["agent_id", "nonce", "timestamp", "signature"]
        if not all(field in auth for field in required_fields):
            return False
        
        agent_id = auth["agent_id"]
        nonce = auth["nonce"]
        timestamp = auth["timestamp"]
        signature = auth["signature"]
        
        # Check if agent is registered
        if agent_id not in self._credentials:
            return False
        
        # Check if the sender matches the signing agent
        if message.sender != agent_id:
            return False
        
        # Check for replay attacks
        if not self._validate_nonce(agent_id, nonce, timestamp):
            return False
        
        # Get the secret key
        secret_key = self._credentials[agent_id]
        
        # Create a copy of the message without the signature
        message_dict = message.to_dict()
        del message_dict["metadata"]["auth"]["signature"]
        
        # Create a canonical representation for verification
        canonical = self._create_canonical_representation(message_dict)
        
        # Compute expected signature
        expected_signature = hmac.new(
            secret_key.encode(),
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(signature, expected_signature)
    
    def _create_canonical_representation(self, message_dict: Dict) -> str:
        """
        Create a canonical string representation of a message for signing.
        
        Args:
            message_dict: Dictionary representation of the message
            
        Returns:
            A canonical string representation
        """
        # Create a simplified representation with only the fields needed for verification
        canonical_dict = {
            "sender": message_dict["sender"],
            "recipients": message_dict["recipients"],
            "content_type": message_dict["content_type"],
            "content": message_dict["content"],
            "message_id": message_dict["message_id"],
            "timestamp": message_dict["timestamp"],
            "auth": message_dict["metadata"]["auth"]
        }
        
        # Convert to a stable string representation
        return str(sorted(canonical_dict.items()))
    
    def _validate_nonce(self, agent_id: str, nonce: str, timestamp: str) -> bool:
        """
        Validate a nonce to prevent replay attacks.
        
        Args:
            agent_id: The ID of the agent
            nonce: The nonce to validate
            timestamp: The timestamp of the message
            
        Returns:
            True if the nonce is valid, False otherwise
        """
        # Check if nonce has been used before
        if nonce in self._used_nonces[agent_id]:
            return False
        
        # Check if timestamp is recent
        try:
            msg_time = int(timestamp)
            current_time = int(time.time())
            
            # Check if message is too old or from the future
            if msg_time < current_time - self._nonce_ttl or msg_time > current_time + 60:
                return False
            
        except ValueError:
            return False
        
        # Add nonce to used nonces
        self._used_nonces[agent_id].append(nonce)
        
        # Clean up old nonces
        self._clean_old_nonces(agent_id)
        
        return True
    
    def _clean_old_nonces(self, agent_id: str):
        """
        Clean up old nonces to prevent memory leaks.
        
        Args:
            agent_id: The ID of the agent
        """
        # Limit the number of stored nonces per agent
        if len(self._used_nonces[agent_id]) > 1000:
            self._used_nonces[agent_id] = self._used_nonces[agent_id][-1000:] 