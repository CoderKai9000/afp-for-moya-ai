"""
Metrics module for Agent Flow Protocol.

Provides mechanisms for collecting and reporting metrics about AFP operations,
enabling monitoring and performance analysis.
"""

import time
import threading
import json
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from collections import defaultdict, deque


class MetricType(Enum):
    """Enumeration of metric types."""
    COUNTER = auto()    # Monotonically increasing counter
    GAUGE = auto()      # Value that can go up and down
    HISTOGRAM = auto()  # Distribution of values
    TIMER = auto()      # Duration of operations


class AFPMetrics:
    """
    Metrics collector for AFP.
    
    Collects and reports metrics about AFP operations, such as message counts,
    delivery times, and error rates.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize the metrics collector.
        
        Args:
            max_history: Maximum number of historical values to keep for histograms
        """
        # Metrics storage
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
        # Metric metadata
        self._metric_types: Dict[str, MetricType] = {}
        self._metric_descriptions: Dict[str, str] = {}
        self._metric_tags: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.
        
        Args:
            name: Name of the metric
            value: Value to increment by
            tags: Optional tags for the metric
        """
        with self._lock:
            self._metric_types[name] = MetricType.COUNTER
            self._counters[name] += value
            
            if tags:
                self._metric_tags[name].update(tags)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric.
        
        Args:
            name: Name of the metric
            value: Value to set
            tags: Optional tags for the metric
        """
        with self._lock:
            self._metric_types[name] = MetricType.GAUGE
            self._gauges[name] = value
            
            if tags:
                self._metric_tags[name].update(tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Record a value in a histogram metric.
        
        Args:
            name: Name of the metric
            value: Value to record
            tags: Optional tags for the metric
        """
        with self._lock:
            self._metric_types[name] = MetricType.HISTOGRAM
            self._histograms[name].append(value)
            
            if tags:
                self._metric_tags[name].update(tags)
    
    def start_timer(self, name: str) -> int:
        """
        Start a timer for an operation.
        
        Args:
            name: Name of the metric
            
        Returns:
            Timer ID for stopping the timer
        """
        with self._lock:
            self._metric_types[name] = MetricType.TIMER
            timer_id = len(self._timers[name])
            self._timers[name].append(time.time())
            return timer_id
    
    def stop_timer(self, name: str, timer_id: int, tags: Optional[Dict[str, str]] = None) -> float:
        """
        Stop a timer and record the duration.
        
        Args:
            name: Name of the metric
            timer_id: Timer ID from start_timer
            tags: Optional tags for the metric
            
        Returns:
            Duration in seconds
            
        Raises:
            ValueError: If the timer ID is invalid
        """
        with self._lock:
            if name not in self._timers or timer_id >= len(self._timers[name]):
                raise ValueError(f"Invalid timer ID: {timer_id} for metric {name}")
            
            start_time = self._timers[name][timer_id]
            duration = time.time() - start_time
            
            # Record the duration in the histogram
            self.record_histogram(f"{name}.duration", duration, tags)
            
            if tags:
                self._metric_tags[name].update(tags)
            
            return duration
    
    def set_description(self, name: str, description: str):
        """
        Set a description for a metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
        """
        with self._lock:
            self._metric_descriptions[name] = description
    
    def get_counter(self, name: str) -> int:
        """
        Get the value of a counter metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Current value of the counter
        """
        with self._lock:
            return self._counters.get(name, 0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """
        Get the value of a gauge metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Current value of the gauge, or None if not set
        """
        with self._lock:
            return self._gauges.get(name)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a histogram metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Dictionary of statistics (min, max, mean, p50, p90, p95, p99)
        """
        with self._lock:
            if name not in self._histograms or not self._histograms[name]:
                return {
                    'min': 0,
                    'max': 0,
                    'mean': 0,
                    'p50': 0,
                    'p90': 0,
                    'p95': 0,
                    'p99': 0,
                    'count': 0
                }
            
            values = sorted(self._histograms[name])
            count = len(values)
            
            return {
                'min': values[0],
                'max': values[-1],
                'mean': sum(values) / count,
                'p50': values[int(count * 0.5)],
                'p90': values[int(count * 0.9)],
                'p95': values[int(count * 0.95)],
                'p99': values[int(count * 0.99)],
                'count': count
            }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all metrics.
        
        Returns:
            Dictionary of metrics (name -> {type, value, description, tags})
        """
        with self._lock:
            metrics = {}
            
            # Add counters
            for name, value in self._counters.items():
                metrics[name] = {
                    'type': 'counter',
                    'value': value,
                    'description': self._metric_descriptions.get(name, ''),
                    'tags': dict(self._metric_tags.get(name, {}))
                }
            
            # Add gauges
            for name, value in self._gauges.items():
                metrics[name] = {
                    'type': 'gauge',
                    'value': value,
                    'description': self._metric_descriptions.get(name, ''),
                    'tags': dict(self._metric_tags.get(name, {}))
                }
            
            # Add histograms
            for name in self._histograms:
                metrics[name] = {
                    'type': 'histogram',
                    'value': self.get_histogram_stats(name),
                    'description': self._metric_descriptions.get(name, ''),
                    'tags': dict(self._metric_tags.get(name, {}))
                }
            
            return metrics
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
    
    def to_json(self) -> str:
        """
        Convert metrics to JSON string.
        
        Returns:
            JSON string representation of metrics
        """
        return json.dumps(self.get_all_metrics(), indent=2)


class AFPMetricsCollector:
    """
    Metrics collector for AFP operations.
    
    Provides pre-defined metrics for common AFP operations and events.
    """
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = AFPMetrics()
        
        # Define standard metrics
        self._define_standard_metrics()
    
    def _define_standard_metrics(self):
        """Define standard metrics for AFP operations."""
        # Message metrics
        self.metrics.set_description("afp.messages.sent", "Number of messages sent")
        self.metrics.set_description("afp.messages.received", "Number of messages received")
        self.metrics.set_description("afp.messages.delivered", "Number of messages successfully delivered")
        self.metrics.set_description("afp.messages.failed", "Number of messages that failed to deliver")
        self.metrics.set_description("afp.messages.expired", "Number of messages that expired before delivery")
        self.metrics.set_description("afp.messages.size", "Size of messages in bytes")
        
        # Agent metrics
        self.metrics.set_description("afp.agents.registered", "Number of registered agents")
        self.metrics.set_description("afp.agents.active", "Number of active agents")
        
        # Subscription metrics
        self.metrics.set_description("afp.subscriptions.active", "Number of active subscriptions")
        self.metrics.set_description("afp.subscriptions.created", "Number of subscriptions created")
        self.metrics.set_description("afp.subscriptions.deleted", "Number of subscriptions deleted")
        
        # Performance metrics
        self.metrics.set_description("afp.performance.message_delivery_time", "Time to deliver messages (seconds)")
        self.metrics.set_description("afp.performance.message_processing_time", "Time to process messages (seconds)")
        
        # Error metrics
        self.metrics.set_description("afp.errors.routing", "Number of routing errors")
        self.metrics.set_description("afp.errors.validation", "Number of validation errors")
        self.metrics.set_description("afp.errors.timeout", "Number of timeout errors")
        self.metrics.set_description("afp.errors.security", "Number of security errors")
    
    def record_message_sent(self, message_size: int, agent_id: str):
        """
        Record a message being sent.
        
        Args:
            message_size: Size of the message in bytes
            agent_id: ID of the sending agent
        """
        self.metrics.increment_counter("afp.messages.sent", tags={"agent_id": agent_id})
        self.metrics.record_histogram("afp.messages.size", message_size, tags={"direction": "sent"})
    
    def record_message_received(self, message_size: int, agent_id: str):
        """
        Record a message being received.
        
        Args:
            message_size: Size of the message in bytes
            agent_id: ID of the receiving agent
        """
        self.metrics.increment_counter("afp.messages.received", tags={"agent_id": agent_id})
        self.metrics.record_histogram("afp.messages.size", message_size, tags={"direction": "received"})
    
    def record_message_delivered(self, agent_id: str, delivery_time: float):
        """
        Record a message being successfully delivered.
        
        Args:
            agent_id: ID of the receiving agent
            delivery_time: Time taken to deliver the message (seconds)
        """
        self.metrics.increment_counter("afp.messages.delivered", tags={"agent_id": agent_id})
        self.metrics.record_histogram("afp.performance.message_delivery_time", delivery_time)
    
    def record_message_failed(self, agent_id: str, reason: str):
        """
        Record a message delivery failure.
        
        Args:
            agent_id: ID of the intended receiving agent
            reason: Reason for the failure
        """
        self.metrics.increment_counter("afp.messages.failed", tags={"agent_id": agent_id, "reason": reason})
    
    def record_message_expired(self, agent_id: str):
        """
        Record a message expiring before delivery.
        
        Args:
            agent_id: ID of the intended receiving agent
        """
        self.metrics.increment_counter("afp.messages.expired", tags={"agent_id": agent_id})
    
    def record_agent_registered(self, agent_id: str):
        """
        Record an agent being registered.
        
        Args:
            agent_id: ID of the agent
        """
        self.metrics.increment_counter("afp.agents.registered")
        self.metrics.set_gauge("afp.agents.active", self.metrics.get_gauge("afp.agents.active") or 0 + 1)
    
    def record_agent_unregistered(self, agent_id: str):
        """
        Record an agent being unregistered.
        
        Args:
            agent_id: ID of the agent
        """
        active = self.metrics.get_gauge("afp.agents.active") or 0
        self.metrics.set_gauge("afp.agents.active", max(0, active - 1))
    
    def record_subscription_created(self, agent_id: str):
        """
        Record a subscription being created.
        
        Args:
            agent_id: ID of the subscribing agent
        """
        self.metrics.increment_counter("afp.subscriptions.created", tags={"agent_id": agent_id})
        self.metrics.set_gauge("afp.subscriptions.active", self.metrics.get_gauge("afp.subscriptions.active") or 0 + 1)
    
    def record_subscription_deleted(self, agent_id: str):
        """
        Record a subscription being deleted.
        
        Args:
            agent_id: ID of the subscribing agent
        """
        self.metrics.increment_counter("afp.subscriptions.deleted", tags={"agent_id": agent_id})
        active = self.metrics.get_gauge("afp.subscriptions.active") or 0
        self.metrics.set_gauge("afp.subscriptions.active", max(0, active - 1))
    
    def record_error(self, error_type: str, agent_id: Optional[str] = None):
        """
        Record an error.
        
        Args:
            error_type: Type of error (routing, validation, timeout, security)
            agent_id: Optional ID of the agent involved
        """
        metric_name = f"afp.errors.{error_type}"
        tags = {}
        if agent_id:
            tags["agent_id"] = agent_id
        
        self.metrics.increment_counter(metric_name, tags=tags)
    
    def start_message_processing_timer(self, agent_id: str) -> int:
        """
        Start a timer for message processing.
        
        Args:
            agent_id: ID of the processing agent
            
        Returns:
            Timer ID for stopping the timer
        """
        return self.metrics.start_timer("afp.performance.message_processing_time")
    
    def stop_message_processing_timer(self, timer_id: int, agent_id: str) -> float:
        """
        Stop a timer for message processing and record the duration.
        
        Args:
            timer_id: Timer ID from start_message_processing_timer
            agent_id: ID of the processing agent
            
        Returns:
            Duration in seconds
        """
        return self.metrics.stop_timer(
            "afp.performance.message_processing_time", 
            timer_id, 
            tags={"agent_id": agent_id}
        )
    
    def get_metrics_report(self) -> str:
        """
        Get a JSON report of all metrics.
        
        Returns:
            JSON string representation of metrics
        """
        return self.metrics.to_json() 