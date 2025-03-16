"""
Circuit breaker module for Agent Flow Protocol.

Implements the circuit breaker pattern to prevent cascading failures
and provide fault tolerance in distributed systems.
"""

import time
import threading
from enum import Enum, auto
from typing import Dict, Callable, Any, Optional, List, Tuple


class CircuitState(Enum):
    """Enumeration of circuit breaker states."""
    CLOSED = auto()    # Normal operation, requests are allowed
    OPEN = auto()      # Failure threshold exceeded, requests are blocked
    HALF_OPEN = auto() # Testing if the system has recovered


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.
    
    Monitors failures and temporarily blocks operations when a failure
    threshold is exceeded, preventing system overload during failures.
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 reset_timeout: float = 30.0,
                 half_open_max_calls: int = 1):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            reset_timeout: Time to wait before testing if the system has recovered (seconds)
            half_open_max_calls: Maximum number of test calls in half-open state
        """
        # Configuration
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        
        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_calls = 0
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def execute(self, 
                operation: Callable[..., Any], 
                fallback: Optional[Callable[..., Any]] = None, 
                *args, **kwargs) -> Any:
        """
        Execute an operation with circuit breaker protection.
        
        Args:
            operation: The operation to execute
            fallback: Optional fallback function to call if the circuit is open
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation
            
        Returns:
            The result of the operation or fallback
            
        Raises:
            Exception: If the circuit is open and no fallback is provided
        """
        with self._lock:
            # Check if circuit is open
            if self._state == CircuitState.OPEN:
                # Check if it's time to try half-open
                if time.time() - self._last_failure_time > self.reset_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                else:
                    # Circuit is open, use fallback or raise exception
                    if fallback:
                        return fallback(*args, **kwargs)
                    else:
                        raise Exception("Circuit is open")
            
            # Check if we've exceeded half-open call limit
            if self._state == CircuitState.HALF_OPEN and self._half_open_calls >= self.half_open_max_calls:
                # Too many half-open calls, use fallback or raise exception
                if fallback:
                    return fallback(*args, **kwargs)
                else:
                    raise Exception("Circuit is half-open and call limit exceeded")
            
            # Increment half-open call count
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
        
        # Execute the operation
        try:
            result = operation(*args, **kwargs)
            
            # Operation succeeded, reset circuit if needed
            with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    # Success in half-open state, close the circuit
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                elif self._state == CircuitState.CLOSED:
                    # Success in closed state, reset failure count
                    self._failure_count = 0
            
            return result
            
        except Exception as e:
            # Operation failed, update circuit state
            with self._lock:
                self._last_failure_time = time.time()
                
                if self._state == CircuitState.HALF_OPEN:
                    # Failure in half-open state, open the circuit
                    self._state = CircuitState.OPEN
                elif self._state == CircuitState.CLOSED:
                    # Failure in closed state, increment failure count
                    self._failure_count += 1
                    
                    # Check if failure threshold is exceeded
                    if self._failure_count >= self.failure_threshold:
                        self._state = CircuitState.OPEN
            
            # Use fallback or re-raise exception
            if fallback:
                return fallback(*args, **kwargs)
            else:
                raise
    
    def get_state(self) -> CircuitState:
        """Get the current state of the circuit breaker."""
        with self._lock:
            return self._state
    
    def get_failure_count(self) -> int:
        """Get the current failure count."""
        with self._lock:
            return self._failure_count
    
    def reset(self):
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
    
    def force_open(self):
        """Force the circuit breaker to open state."""
        with self._lock:
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Provides a centralized way to manage and monitor circuit breakers
    for different operations or services.
    """
    
    def __init__(self):
        """Initialize the circuit breaker registry."""
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_or_create(self, 
                      name: str, 
                      failure_threshold: int = 5,
                      reset_timeout: float = 30.0,
                      half_open_max_calls: int = 1) -> CircuitBreaker:
        """
        Get or create a circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening the circuit
            reset_timeout: Time to wait before testing if the system has recovered (seconds)
            half_open_max_calls: Maximum number of test calls in half-open state
            
        Returns:
            The circuit breaker
        """
        with self._lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    reset_timeout=reset_timeout,
                    half_open_max_calls=half_open_max_calls
                )
            
            return self._circuit_breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get a circuit breaker by name.
        
        Args:
            name: Name of the circuit breaker
            
        Returns:
            The circuit breaker, or None if not found
        """
        with self._lock:
            return self._circuit_breakers.get(name)
    
    def remove(self, name: str) -> bool:
        """
        Remove a circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            
        Returns:
            True if the circuit breaker was removed, False if not found
        """
        with self._lock:
            if name in self._circuit_breakers:
                del self._circuit_breakers[name]
                return True
            return False
    
    def get_all(self) -> Dict[str, CircuitBreaker]:
        """
        Get all circuit breakers.
        
        Returns:
            Dictionary of circuit breakers (name -> circuit breaker)
        """
        with self._lock:
            return dict(self._circuit_breakers)
    
    def get_states(self) -> Dict[str, CircuitState]:
        """
        Get the states of all circuit breakers.
        
        Returns:
            Dictionary of circuit breaker states (name -> state)
        """
        with self._lock:
            return {name: cb.get_state() for name, cb in self._circuit_breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers to closed state."""
        with self._lock:
            for cb in self._circuit_breakers.values():
                cb.reset() 