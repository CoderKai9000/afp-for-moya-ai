"""
Encryption module for Agent Flow Protocol.

Provides encryption mechanisms for securing message content
between agents, ensuring confidentiality of communications.
"""

import os
import base64
import json
from typing import Dict, Any, Optional, Tuple, Union
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..exceptions import AFPSecurityError
from ..message import AFPMessage, ContentType


class AFPEncryptor:
    """
    Base encryptor class for AFP.
    
    Provides the interface for encryption mechanisms.
    """
    
    def encrypt_message(self, message: AFPMessage, key: Any) -> AFPMessage:
        """
        Encrypt a message to ensure confidentiality.
        
        Args:
            message: The message to encrypt
            key: The encryption key
            
        Returns:
            The encrypted message
        """
        raise NotImplementedError("Subclasses must implement encrypt_message")
    
    def decrypt_message(self, message: AFPMessage, key: Any) -> AFPMessage:
        """
        Decrypt an encrypted message.
        
        Args:
            message: The encrypted message
            key: The decryption key
            
        Returns:
            The decrypted message
        """
        raise NotImplementedError("Subclasses must implement decrypt_message")


class AESGCMEncryptor(AFPEncryptor):
    """
    AES-GCM based encryptor for AFP.
    
    Uses AES-GCM for authenticated encryption of message content.
    """
    
    def __init__(self, key_size: int = 32):
        """
        Initialize the AES-GCM encryptor.
        
        Args:
            key_size: Size of the encryption key in bytes (default: 32 for AES-256)
        """
        self.key_size = key_size
    
    def generate_key(self, passphrase: Optional[str] = None, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Generate an encryption key.
        
        Args:
            passphrase: Optional passphrase to derive the key from
            salt: Optional salt for key derivation
            
        Returns:
            Tuple of (key, salt)
        """
        # Generate a random salt if not provided
        if salt is None:
            salt = os.urandom(16)
        
        # If passphrase is provided, derive key using PBKDF2
        if passphrase:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.key_size,
                salt=salt,
                iterations=100000,
            )
            key = kdf.derive(passphrase.encode())
        else:
            # Otherwise, generate a random key
            key = os.urandom(self.key_size)
        
        return key, salt
    
    def encrypt_message(self, message: AFPMessage, key: bytes) -> AFPMessage:
        """
        Encrypt a message using AES-GCM.
        
        Args:
            message: The message to encrypt
            key: The encryption key
            
        Returns:
            The encrypted message
            
        Raises:
            AFPSecurityError: If encryption fails
        """
        try:
            # Create a copy of the message to avoid modifying the original
            message_dict = message.to_dict()
            
            # Serialize the content
            if message.content_type == ContentType.JSON:
                content_bytes = json.dumps(message.content).encode()
            else:
                content_bytes = str(message.content).encode()
            
            # Generate a random nonce
            nonce = os.urandom(12)
            
            # Create an AES-GCM cipher
            aesgcm = AESGCM(key)
            
            # Encrypt the content
            encrypted_content = aesgcm.encrypt(nonce, content_bytes, None)
            
            # Create encryption metadata
            if "encryption" not in message_dict["metadata"]:
                message_dict["metadata"]["encryption"] = {}
            
            # Store encryption details in metadata
            message_dict["metadata"]["encryption"].update({
                "algorithm": "AES-GCM",
                "nonce": base64.b64encode(nonce).decode(),
                "is_encrypted": True
            })
            
            # Replace content with encrypted content
            message_dict["content"] = base64.b64encode(encrypted_content).decode()
            
            # Set content type to indicate encryption
            message_dict["content_type"] = ContentType.BINARY.name
            
            # Create a new message with the encrypted content
            return AFPMessage.from_dict(message_dict)
            
        except Exception as e:
            raise AFPSecurityError(f"Failed to encrypt message: {str(e)}")
    
    def decrypt_message(self, message: AFPMessage, key: bytes) -> AFPMessage:
        """
        Decrypt an encrypted message.
        
        Args:
            message: The encrypted message
            key: The decryption key
            
        Returns:
            The decrypted message
            
        Raises:
            AFPSecurityError: If decryption fails or the message is not encrypted
        """
        # Check if message is encrypted
        if "encryption" not in message.metadata or not message.metadata["encryption"].get("is_encrypted"):
            raise AFPSecurityError("Message is not encrypted")
        
        try:
            # Get encryption metadata
            encryption = message.metadata["encryption"]
            
            # Check encryption algorithm
            if encryption.get("algorithm") != "AES-GCM":
                raise AFPSecurityError(f"Unsupported encryption algorithm: {encryption.get('algorithm')}")
            
            # Get the nonce
            nonce = base64.b64decode(encryption["nonce"])
            
            # Get the encrypted content
            encrypted_content = base64.b64decode(message.content)
            
            # Create an AES-GCM cipher
            aesgcm = AESGCM(key)
            
            # Decrypt the content
            decrypted_content_bytes = aesgcm.decrypt(nonce, encrypted_content, None)
            
            # Create a copy of the message to avoid modifying the original
            message_dict = message.to_dict()
            
            # Get the original content type from metadata
            original_content_type = encryption.get("original_content_type", ContentType.TEXT.name)
            
            # Parse the decrypted content based on the original content type
            if original_content_type == ContentType.JSON.name:
                decrypted_content = json.loads(decrypted_content_bytes.decode())
            else:
                decrypted_content = decrypted_content_bytes.decode()
            
            # Update the message with decrypted content
            message_dict["content"] = decrypted_content
            message_dict["content_type"] = original_content_type
            
            # Remove encryption metadata
            del message_dict["metadata"]["encryption"]
            
            # Create a new message with the decrypted content
            return AFPMessage.from_dict(message_dict)
            
        except Exception as e:
            raise AFPSecurityError(f"Failed to decrypt message: {str(e)}")
    
    def encrypt_for_recipient(self, 
                              message: AFPMessage, 
                              recipient_keys: Dict[str, bytes]) -> AFPMessage:
        """
        Encrypt a message for specific recipients using their keys.
        
        Args:
            message: The message to encrypt
            recipient_keys: Dictionary mapping recipient IDs to their encryption keys
            
        Returns:
            The encrypted message
            
        Raises:
            AFPSecurityError: If encryption fails or recipients don't match
        """
        # Verify that all recipients have keys
        missing_recipients = set(message.recipients) - set(recipient_keys.keys())
        if missing_recipients and not message.is_broadcast():
            raise AFPSecurityError(f"Missing encryption keys for recipients: {missing_recipients}")
        
        # Generate a random message key for this message
        message_key = os.urandom(self.key_size)
        
        # Encrypt the message with the message key
        encrypted_message = self.encrypt_message(message, message_key)
        
        # Encrypt the message key for each recipient
        encrypted_keys = {}
        for recipient, recipient_key in recipient_keys.items():
            if recipient in message.recipients or message.is_broadcast():
                # Generate a random nonce for each recipient
                nonce = os.urandom(12)
                
                # Create an AES-GCM cipher with the recipient's key
                aesgcm = AESGCM(recipient_key)
                
                # Encrypt the message key for this recipient
                encrypted_key = aesgcm.encrypt(nonce, message_key, None)
                
                # Store the encrypted key and nonce
                encrypted_keys[recipient] = {
                    "key": base64.b64encode(encrypted_key).decode(),
                    "nonce": base64.b64encode(nonce).decode()
                }
        
        # Add recipient keys to metadata
        encrypted_message_dict = encrypted_message.to_dict()
        encrypted_message_dict["metadata"]["encryption"]["recipient_keys"] = encrypted_keys
        
        # Create a new message with the recipient keys
        return AFPMessage.from_dict(encrypted_message_dict)
    
    def decrypt_for_recipient(self, 
                              message: AFPMessage, 
                              recipient_id: str, 
                              recipient_key: bytes) -> AFPMessage:
        """
        Decrypt a message that was encrypted for a specific recipient.
        
        Args:
            message: The encrypted message
            recipient_id: The ID of the recipient
            recipient_key: The recipient's decryption key
            
        Returns:
            The decrypted message
            
        Raises:
            AFPSecurityError: If decryption fails or the message is not encrypted for this recipient
        """
        # Check if message is encrypted
        if "encryption" not in message.metadata or not message.metadata["encryption"].get("is_encrypted"):
            raise AFPSecurityError("Message is not encrypted")
        
        # Check if message was encrypted for this recipient
        if "recipient_keys" not in message.metadata["encryption"]:
            # If no recipient keys, assume it was encrypted with a shared key
            return self.decrypt_message(message, recipient_key)
        
        recipient_keys = message.metadata["encryption"]["recipient_keys"]
        if recipient_id not in recipient_keys:
            raise AFPSecurityError(f"Message is not encrypted for recipient {recipient_id}")
        
        try:
            # Get the encrypted message key and nonce for this recipient
            encrypted_key = base64.b64decode(recipient_keys[recipient_id]["key"])
            nonce = base64.b64decode(recipient_keys[recipient_id]["nonce"])
            
            # Create an AES-GCM cipher with the recipient's key
            aesgcm = AESGCM(recipient_key)
            
            # Decrypt the message key
            message_key = aesgcm.decrypt(nonce, encrypted_key, None)
            
            # Decrypt the message with the message key
            return self.decrypt_message(message, message_key)
            
        except Exception as e:
            raise AFPSecurityError(f"Failed to decrypt message for recipient {recipient_id}: {str(e)}") 