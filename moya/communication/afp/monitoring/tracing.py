"""
Tracing module for Agent Flow Protocol.

Provides mechanisms for tracing message flows and operations across
the distributed system, enabling debugging and performance analysis.
"""

import time
import uuid
import json
import threading
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Set, Callable, Tuple, Union
from collections import defaultdict, deque


class SpanType(Enum):
    """Enumeration of span types."""
    MESSAGE = auto()     # Message transmission
    PROCESSING = auto()  # Message processing
    OPERATION = auto()   # General operation


class TraceSpan:
    """
    Span representing a single operation in a trace.
    
    A span can represent a message transmission, message processing,
    or any other operation that is part of a trace.
    """
    
    def __init__(self, 
                 trace_id: str,
                 span_id: str,
                 parent_span_id: Optional[str],
                 name: str,
                 span_type: SpanType,
                 start_time: float,
                 end_time: Optional[float] = None,
                 attributes: Optional[Dict[str, Any]] = None,
                 events: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize a trace span.
        
        Args:
            trace_id: ID of the trace this span belongs to
            span_id: ID of this span
            parent_span_id: ID of the parent span, or None if this is a root span
            name: Name of the operation
            span_type: Type of span
            start_time: Start time of the span (seconds since epoch)
            end_time: End time of the span, or None if not ended
            attributes: Additional attributes for the span
            events: List of events that occurred during the span
        """
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.name = name
        self.span_type = span_type
        self.start_time = start_time
        self.end_time = end_time
        self.attributes = attributes or {}
        self.events = events or []
    
    def end(self, end_time: Optional[float] = None):
        """
        End the span.
        
        Args:
            end_time: End time of the span (seconds since epoch), or None for current time
        """
        self.end_time = end_time or time.time()
    
    def add_event(self, name: str, timestamp: Optional[float] = None, attributes: Optional[Dict[str, Any]] = None):
        """
        Add an event to the span.
        
        Args:
            name: Name of the event
            timestamp: Time of the event (seconds since epoch), or None for current time
            attributes: Additional attributes for the event
        """
        self.events.append({
            'name': name,
            'timestamp': timestamp or time.time(),
            'attributes': attributes or {}
        })
    
    def add_attribute(self, key: str, value: Any):
        """
        Add an attribute to the span.
        
        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value
    
    def duration(self) -> Optional[float]:
        """
        Get the duration of the span.
        
        Returns:
            Duration in seconds, or None if the span has not ended
        """
        if self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the span to a dictionary.
        
        Returns:
            Dictionary representation of the span
        """
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'name': self.name,
            'span_type': self.span_type.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration(),
            'attributes': self.attributes,
            'events': self.events
        }


class AFPTracer:
    """
    Tracer for AFP operations.
    
    Provides mechanisms for tracing message flows and operations across
    the distributed system.
    """
    
    def __init__(self, max_traces: int = 1000):
        """
        Initialize the tracer.
        
        Args:
            max_traces: Maximum number of traces to keep
        """
        # Traces storage (trace_id -> list of spans)
        self._traces: Dict[str, List[TraceSpan]] = defaultdict(list)
        
        # Active spans (span_id -> span)
        self._active_spans: Dict[str, TraceSpan] = {}
        
        # Maximum number of traces to keep
        self.max_traces = max_traces
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def create_trace(self) -> str:
        """
        Create a new trace.
        
        Returns:
            Trace ID
        """
        trace_id = str(uuid.uuid4())
        
        with self._lock:
            self._traces[trace_id] = []
            
            # Trim traces if needed
            if len(self._traces) > self.max_traces:
                # Remove oldest traces
                oldest_traces = sorted(self._traces.keys(), 
                                      key=lambda tid: min(span.start_time for span in self._traces[tid]) 
                                      if self._traces[tid] else float('inf'))
                for tid in oldest_traces[:len(self._traces) - self.max_traces]:
                    del self._traces[tid]
        
        return trace_id
    
    def start_span(self, 
                   name: str, 
                   span_type: SpanType,
                   trace_id: Optional[str] = None,
                   parent_span_id: Optional[str] = None,
                   attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new span.
        
        Args:
            name: Name of the operation
            span_type: Type of span
            trace_id: ID of the trace, or None to create a new trace
            parent_span_id: ID of the parent span, or None if this is a root span
            attributes: Additional attributes for the span
            
        Returns:
            Span ID
        """
        # Create a new trace if needed
        if trace_id is None:
            trace_id = self.create_trace()
        
        # Create a new span
        span_id = str(uuid.uuid4())
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            span_type=span_type,
            start_time=time.time(),
            attributes=attributes
        )
        
        with self._lock:
            # Add span to trace
            self._traces[trace_id].append(span)
            
            # Add span to active spans
            self._active_spans[span_id] = span
        
        return span_id
    
    def end_span(self, span_id: str, end_time: Optional[float] = None):
        """
        End a span.
        
        Args:
            span_id: ID of the span to end
            end_time: End time of the span (seconds since epoch), or None for current time
        """
        with self._lock:
            if span_id in self._active_spans:
                span = self._active_spans[span_id]
                span.end(end_time)
                del self._active_spans[span_id]
    
    def add_event(self, 
                  span_id: str, 
                  name: str, 
                  timestamp: Optional[float] = None, 
                  attributes: Optional[Dict[str, Any]] = None):
        """
        Add an event to a span.
        
        Args:
            span_id: ID of the span
            name: Name of the event
            timestamp: Time of the event (seconds since epoch), or None for current time
            attributes: Additional attributes for the event
        """
        with self._lock:
            if span_id in self._active_spans:
                self._active_spans[span_id].add_event(name, timestamp, attributes)
    
    def add_attribute(self, span_id: str, key: str, value: Any):
        """
        Add an attribute to a span.
        
        Args:
            span_id: ID of the span
            key: Attribute key
            value: Attribute value
        """
        with self._lock:
            if span_id in self._active_spans:
                self._active_spans[span_id].add_attribute(key, value)
    
    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        Get all spans in a trace.
        
        Args:
            trace_id: ID of the trace
            
        Returns:
            List of spans in the trace
        """
        with self._lock:
            if trace_id not in self._traces:
                return []
            
            return [span.to_dict() for span in self._traces[trace_id]]
    
    def get_all_traces(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all traces.
        
        Returns:
            Dictionary of traces (trace_id -> list of spans)
        """
        with self._lock:
            return {tid: [span.to_dict() for span in spans] for tid, spans in self._traces.items()}
    
    def clear_trace(self, trace_id: str):
        """
        Clear a trace.
        
        Args:
            trace_id: ID of the trace to clear
        """
        with self._lock:
            if trace_id in self._traces:
                # Remove active spans for this trace
                for span_id, span in list(self._active_spans.items()):
                    if span.trace_id == trace_id:
                        del self._active_spans[span_id]
                
                # Remove trace
                del self._traces[trace_id]
    
    def clear_all_traces(self):
        """Clear all traces."""
        with self._lock:
            self._traces.clear()
            self._active_spans.clear()
    
    def to_json(self) -> str:
        """
        Convert all traces to JSON string.
        
        Returns:
            JSON string representation of all traces
        """
        return json.dumps(self.get_all_traces(), indent=2)


class AFPMessageTracer:
    """
    Specialized tracer for AFP message flows.
    
    Provides high-level methods for tracing message flows between agents.
    """
    
    def __init__(self):
        """Initialize the message tracer."""
        self.tracer = AFPTracer()
    
    def trace_message_send(self, 
                           message_id: str, 
                           sender: str, 
                           recipients: List[str],
                           content_type: str,
                           content_size: int,
                           trace_id: Optional[str] = None,
                           parent_span_id: Optional[str] = None) -> str:
        """
        Trace a message being sent.
        
        Args:
            message_id: ID of the message
            sender: ID of the sending agent
            recipients: IDs of the receiving agents
            content_type: Type of message content
            content_size: Size of message content in bytes
            trace_id: Optional trace ID to use
            parent_span_id: Optional parent span ID
            
        Returns:
            Span ID for the send operation
        """
        attributes = {
            'message_id': message_id,
            'sender': sender,
            'recipients': recipients,
            'content_type': content_type,
            'content_size': content_size
        }
        
        return self.tracer.start_span(
            name=f"send_message:{message_id}",
            span_type=SpanType.MESSAGE,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            attributes=attributes
        )
    
    def trace_message_receive(self, 
                              message_id: str, 
                              sender: str, 
                              recipient: str,
                              content_type: str,
                              content_size: int,
                              trace_id: Optional[str] = None,
                              parent_span_id: Optional[str] = None) -> str:
        """
        Trace a message being received.
        
        Args:
            message_id: ID of the message
            sender: ID of the sending agent
            recipient: ID of the receiving agent
            content_type: Type of message content
            content_size: Size of message content in bytes
            trace_id: Optional trace ID to use
            parent_span_id: Optional parent span ID
            
        Returns:
            Span ID for the receive operation
        """
        attributes = {
            'message_id': message_id,
            'sender': sender,
            'recipient': recipient,
            'content_type': content_type,
            'content_size': content_size
        }
        
        return self.tracer.start_span(
            name=f"receive_message:{message_id}",
            span_type=SpanType.MESSAGE,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            attributes=attributes
        )
    
    def trace_message_process(self, 
                              message_id: str, 
                              agent_id: str,
                              trace_id: Optional[str] = None,
                              parent_span_id: Optional[str] = None) -> str:
        """
        Trace a message being processed.
        
        Args:
            message_id: ID of the message
            agent_id: ID of the processing agent
            trace_id: Optional trace ID to use
            parent_span_id: Optional parent span ID
            
        Returns:
            Span ID for the processing operation
        """
        attributes = {
            'message_id': message_id,
            'agent_id': agent_id
        }
        
        return self.tracer.start_span(
            name=f"process_message:{message_id}",
            span_type=SpanType.PROCESSING,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            attributes=attributes
        )
    
    def trace_operation(self, 
                        name: str, 
                        agent_id: str,
                        trace_id: Optional[str] = None,
                        parent_span_id: Optional[str] = None,
                        attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Trace a general operation.
        
        Args:
            name: Name of the operation
            agent_id: ID of the agent performing the operation
            trace_id: Optional trace ID to use
            parent_span_id: Optional parent span ID
            attributes: Additional attributes for the operation
            
        Returns:
            Span ID for the operation
        """
        attrs = attributes or {}
        attrs['agent_id'] = agent_id
        
        return self.tracer.start_span(
            name=name,
            span_type=SpanType.OPERATION,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            attributes=attrs
        )
    
    def end_span(self, span_id: str):
        """
        End a span.
        
        Args:
            span_id: ID of the span to end
        """
        self.tracer.end_span(span_id)
    
    def add_event(self, 
                  span_id: str, 
                  name: str, 
                  attributes: Optional[Dict[str, Any]] = None):
        """
        Add an event to a span.
        
        Args:
            span_id: ID of the span
            name: Name of the event
            attributes: Additional attributes for the event
        """
        self.tracer.add_event(span_id, name, attributes=attributes)
    
    def get_message_trace(self, message_id: str) -> List[Dict[str, Any]]:
        """
        Get all spans related to a message.
        
        Args:
            message_id: ID of the message
            
        Returns:
            List of spans related to the message
        """
        all_traces = self.tracer.get_all_traces()
        message_spans = []
        
        for trace_spans in all_traces.values():
            for span in trace_spans:
                if 'attributes' in span and 'message_id' in span['attributes'] and span['attributes']['message_id'] == message_id:
                    message_spans.append(span)
        
        return message_spans
    
    def get_agent_traces(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get all spans related to an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of spans related to the agent
        """
        all_traces = self.tracer.get_all_traces()
        agent_spans = []
        
        for trace_spans in all_traces.values():
            for span in trace_spans:
                if 'attributes' in span and (
                    ('agent_id' in span['attributes'] and span['attributes']['agent_id'] == agent_id) or
                    ('sender' in span['attributes'] and span['attributes']['sender'] == agent_id) or
                    ('recipient' in span['attributes'] and span['attributes']['recipient'] == agent_id) or
                    ('recipients' in span['attributes'] and agent_id in span['attributes']['recipients'])
                ):
                    agent_spans.append(span)
        
        return agent_spans 