"""
Authorization module for Agent Flow Protocol.

Provides authorization mechanisms for controlling access to resources
and operations within the AFP system.
"""

from typing import Dict, Set, List, Optional, Any, Callable
from enum import Enum, auto

from ..exceptions import AFPSecurityError
from ..message import AFPMessage


class Permission(Enum):
    """Enumeration of permissions that can be granted to agents."""
    # Message permissions
    SEND_MESSAGE = auto()
    RECEIVE_MESSAGE = auto()
    BROADCAST_MESSAGE = auto()
    
    # Subscription permissions
    CREATE_SUBSCRIPTION = auto()
    DELETE_SUBSCRIPTION = auto()
    
    # Agent permissions
    REGISTER_AGENT = auto()
    UNREGISTER_AGENT = auto()
    
    # Admin permissions
    ADMIN = auto()


class Role(Enum):
    """Predefined roles with associated permissions."""
    # Basic role with minimal permissions
    BASIC = {Permission.SEND_MESSAGE, Permission.RECEIVE_MESSAGE}
    
    # Standard role with common permissions
    STANDARD = {
        Permission.SEND_MESSAGE, 
        Permission.RECEIVE_MESSAGE,
        Permission.CREATE_SUBSCRIPTION,
        Permission.DELETE_SUBSCRIPTION
    }
    
    # Admin role with all permissions
    ADMIN = {perm for perm in Permission}


class AFPAuthorizer:
    """
    Authorization manager for AFP.
    
    Controls access to AFP operations based on agent permissions.
    """
    
    def __init__(self):
        """Initialize the authorizer."""
        # Agent permissions (agent_id -> set of permissions)
        self._agent_permissions: Dict[str, Set[Permission]] = {}
        
        # Agent roles (agent_id -> role)
        self._agent_roles: Dict[str, Role] = {}
        
        # Custom permission checks (permission -> check function)
        self._custom_checks: Dict[Permission, Callable[[str, Any], bool]] = {}
    
    def add_agent(self, agent_id: str, role: Role = Role.BASIC):
        """
        Add an agent with a specific role.
        
        Args:
            agent_id: The ID of the agent
            role: The role to assign to the agent
            
        Raises:
            AFPSecurityError: If the agent already exists
        """
        if agent_id in self._agent_permissions:
            raise AFPSecurityError(f"Agent {agent_id} already exists")
        
        self._agent_roles[agent_id] = role
        self._agent_permissions[agent_id] = set(role.value)
    
    def remove_agent(self, agent_id: str):
        """
        Remove an agent.
        
        Args:
            agent_id: The ID of the agent to remove
            
        Raises:
            AFPSecurityError: If the agent does not exist
        """
        if agent_id not in self._agent_permissions:
            raise AFPSecurityError(f"Agent {agent_id} does not exist")
        
        del self._agent_permissions[agent_id]
        del self._agent_roles[agent_id]
    
    def grant_permission(self, agent_id: str, permission: Permission):
        """
        Grant a permission to an agent.
        
        Args:
            agent_id: The ID of the agent
            permission: The permission to grant
            
        Raises:
            AFPSecurityError: If the agent does not exist
        """
        if agent_id not in self._agent_permissions:
            raise AFPSecurityError(f"Agent {agent_id} does not exist")
        
        self._agent_permissions[agent_id].add(permission)
    
    def revoke_permission(self, agent_id: str, permission: Permission):
        """
        Revoke a permission from an agent.
        
        Args:
            agent_id: The ID of the agent
            permission: The permission to revoke
            
        Raises:
            AFPSecurityError: If the agent does not exist
        """
        if agent_id not in self._agent_permissions:
            raise AFPSecurityError(f"Agent {agent_id} does not exist")
        
        if permission in self._agent_permissions[agent_id]:
            self._agent_permissions[agent_id].remove(permission)
    
    def set_role(self, agent_id: str, role: Role):
        """
        Set the role of an agent.
        
        Args:
            agent_id: The ID of the agent
            role: The role to assign
            
        Raises:
            AFPSecurityError: If the agent does not exist
        """
        if agent_id not in self._agent_permissions:
            raise AFPSecurityError(f"Agent {agent_id} does not exist")
        
        self._agent_roles[agent_id] = role
        self._agent_permissions[agent_id] = set(role.value)
    
    def has_permission(self, agent_id: str, permission: Permission, context: Any = None) -> bool:
        """
        Check if an agent has a specific permission.
        
        Args:
            agent_id: The ID of the agent
            permission: The permission to check
            context: Optional context for custom permission checks
            
        Returns:
            True if the agent has the permission, False otherwise
        """
        # Check if agent exists
        if agent_id not in self._agent_permissions:
            return False
        
        # Check for ADMIN permission (grants all permissions)
        if Permission.ADMIN in self._agent_permissions[agent_id]:
            return True
        
        # Check for the specific permission
        has_perm = permission in self._agent_permissions[agent_id]
        
        # If there's a custom check for this permission, apply it
        if has_perm and permission in self._custom_checks and context is not None:
            return self._custom_checks[permission](agent_id, context)
        
        return has_perm
    
    def register_custom_check(self, 
                              permission: Permission, 
                              check_func: Callable[[str, Any], bool]):
        """
        Register a custom permission check function.
        
        Args:
            permission: The permission to check
            check_func: Function that takes (agent_id, context) and returns a boolean
        """
        self._custom_checks[permission] = check_func
    
    def authorize_message_send(self, message: AFPMessage) -> bool:
        """
        Authorize sending a message.
        
        Args:
            message: The message to authorize
            
        Returns:
            True if the sender is authorized to send the message, False otherwise
        """
        sender = message.sender
        
        # Check if sender has permission to send messages
        if not self.has_permission(sender, Permission.SEND_MESSAGE):
            return False
        
        # For broadcast messages, check if sender has broadcast permission
        if message.is_broadcast() and not self.has_permission(sender, Permission.BROADCAST_MESSAGE):
            return False
        
        return True
    
    def authorize_message_receive(self, message: AFPMessage, recipient: str) -> bool:
        """
        Authorize receiving a message.
        
        Args:
            message: The message to authorize
            recipient: The ID of the recipient
            
        Returns:
            True if the recipient is authorized to receive the message, False otherwise
        """
        # Check if recipient has permission to receive messages
        if not self.has_permission(recipient, Permission.RECEIVE_MESSAGE):
            return False
        
        # Check if recipient is in the message recipients or it's a broadcast
        if not message.is_broadcast() and recipient not in message.recipients:
            return False
        
        return True
    
    def get_agent_permissions(self, agent_id: str) -> Set[Permission]:
        """
        Get the permissions of an agent.
        
        Args:
            agent_id: The ID of the agent
            
        Returns:
            Set of permissions granted to the agent
            
        Raises:
            AFPSecurityError: If the agent does not exist
        """
        if agent_id not in self._agent_permissions:
            raise AFPSecurityError(f"Agent {agent_id} does not exist")
        
        return self._agent_permissions[agent_id].copy()
    
    def get_agent_role(self, agent_id: str) -> Role:
        """
        Get the role of an agent.
        
        Args:
            agent_id: The ID of the agent
            
        Returns:
            The role of the agent
            
        Raises:
            AFPSecurityError: If the agent does not exist
        """
        if agent_id not in self._agent_roles:
            raise AFPSecurityError(f"Agent {agent_id} does not exist")
        
        return self._agent_roles[agent_id] 