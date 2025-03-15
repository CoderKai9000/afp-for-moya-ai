"""
Exception classes for Agent Flow Protocol.

Defines custom exceptions for various error conditions that might arise
during AFP operations, such as message validation failures, routing issues,
and timeout conditions.
"""


class AFPError(Exception):
    """Base exception for all AFP errors."""
    pass


class AFPMessageError(AFPError):
    """Error related to AFP message creation or validation."""
    pass


class AFPRoutingError(AFPError):
    """Error related to routing AFP messages."""
    pass


class AFPSubscriptionError(AFPError):
    """Error related to AFP subscriptions."""
    pass


class AFPTimeoutError(AFPError):
    """Error when a synchronous request times out."""
    pass


class AFPSecurityError(AFPError):
    """Error related to AFP security operations."""
    pass


class AFPReliabilityError(AFPError):
    """Error related to reliable delivery mechanisms."""
    pass


class AFPDataStoreError(AFPError):
    """Error related to AFP data storage operations."""
    pass


class AFPMonitoringError(AFPError):
    """Error related to AFP monitoring operations."""
    pass 