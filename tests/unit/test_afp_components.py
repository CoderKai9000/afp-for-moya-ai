#!/usr/bin/env python
"""
Test script to verify the AFP components.

This script tests the basic functionality of the AFP components,
including the circuit breaker and metrics collection.
"""

import unittest
import sys
import os
from enum import Enum
import time
import random

# Add the parent directory to the path so we can import the moya modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from moya.communication.afp.reliability.circuit_breaker import CircuitBreaker, CircuitState
    from moya.communication.afp.monitoring.metrics import AFPMetrics, MetricType
    from moya.communication.afp.monitoring.tracing import AFPTracer, SpanType
except ImportError as e:
    print(f"Error importing AFP modules: {e}")
    print("Make sure you have created all the necessary files.")
    sys.exit(1)


class TestCircuitBreaker(unittest.TestCase):
    """Test the CircuitBreaker class."""

    def setUp(self):
        """Set up the test case."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout=1.0
        )

    def test_initial_state(self):
        """Test the initial state of the circuit breaker."""
        self.assertEqual(self.circuit_breaker.get_state(), CircuitState.CLOSED)
        self.assertEqual(self.circuit_breaker.get_failure_count(), 0)

    def test_record_success(self):
        """Test recording a successful operation."""
        # We'll simulate a successful operation using the execute method
        def operation():
            return "success"
        
        self.circuit_breaker.execute(operation)
        self.assertEqual(self.circuit_breaker.get_failure_count(), 0)
        self.assertEqual(self.circuit_breaker.get_state(), CircuitState.CLOSED)

    def test_record_failure(self):
        """Test recording a failed operation."""
        # We'll simulate a failed operation using the execute method
        def operation():
            raise Exception("Simulated failure")
        
        try:
            self.circuit_breaker.execute(operation)
        except Exception:
            pass  # Expected exception
        
        self.assertEqual(self.circuit_breaker.get_failure_count(), 1)
        self.assertEqual(self.circuit_breaker.get_state(), CircuitState.CLOSED)

    def test_open_circuit(self):
        """Test opening the circuit after reaching the failure threshold."""
        # Simulate multiple failures to open the circuit
        def operation():
            raise Exception("Simulated failure")
        
        for _ in range(3):
            try:
                self.circuit_breaker.execute(operation)
            except Exception:
                pass  # Expected exception
        
        self.assertEqual(self.circuit_breaker.get_state(), CircuitState.OPEN)

    def test_half_open_after_timeout(self):
        """Test transitioning to half-open state after the recovery timeout."""
        # First, open the circuit
        def operation():
            raise Exception("Simulated failure")
        
        for _ in range(3):
            try:
                self.circuit_breaker.execute(operation)
            except Exception:
                pass  # Expected exception
        
        self.assertEqual(self.circuit_breaker.get_state(), CircuitState.OPEN)
        
        # Wait for the recovery timeout
        time.sleep(1.1)
        
        # The next call should transition to half-open
        # We need to check the state directly since execute will change it
        # Note: In the actual implementation, the state only changes to HALF_OPEN
        # when we try to execute an operation after the timeout
        def success_operation():
            return "success"
        
        try:
            self.circuit_breaker.execute(success_operation)
        except Exception:
            pass  # In case it fails
        
        # Now check if the state is HALF_OPEN or CLOSED (if the operation succeeded)
        state = self.circuit_breaker.get_state()
        self.assertTrue(state == CircuitState.HALF_OPEN or state == CircuitState.CLOSED,
                        f"Expected HALF_OPEN or CLOSED, got {state}")

    def test_execute_with_fallback(self):
        """Test executing an operation with a fallback."""
        def operation():
            return "success"
        
        def fallback():
            return "fallback"
        
        # Test successful operation
        result = self.circuit_breaker.execute(operation, fallback)
        self.assertEqual(result, "success")
        
        # Force the circuit to open
        self.circuit_breaker._state = CircuitState.OPEN
        self.circuit_breaker._last_failure_time = time.time()
        
        # Test fallback when circuit is open
        result = self.circuit_breaker.execute(operation, fallback)
        self.assertEqual(result, "fallback")


class TestAFPMetrics(unittest.TestCase):
    """Test the AFPMetrics class."""

    def setUp(self):
        """Set up the test case."""
        self.metrics = AFPMetrics()

    def test_counter(self):
        """Test incrementing a counter."""
        self.metrics.increment_counter("test_counter", 1)
        self.assertEqual(self.metrics.get_counter("test_counter"), 1)
        
        self.metrics.increment_counter("test_counter", 2)
        self.assertEqual(self.metrics.get_counter("test_counter"), 3)

    def test_gauge(self):
        """Test setting a gauge."""
        self.metrics.set_gauge("test_gauge", 42)
        self.assertEqual(self.metrics.get_gauge("test_gauge"), 42)
        
        self.metrics.set_gauge("test_gauge", 84)
        self.assertEqual(self.metrics.get_gauge("test_gauge"), 84)

    def test_histogram(self):
        """Test recording histogram values."""
        values = [1, 2, 3, 4, 5]
        for value in values:
            self.metrics.record_histogram("test_histogram", value)
        
        stats = self.metrics.get_histogram_stats("test_histogram")
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["min"], 1)
        self.assertEqual(stats["max"], 5)
        self.assertEqual(stats["mean"], 3)

    def test_timer(self):
        """Test timing operations."""
        timer_id = self.metrics.start_timer("test_timer")
        time.sleep(0.1)
        duration = self.metrics.stop_timer("test_timer", timer_id)
        
        self.assertGreaterEqual(duration, 0.1)
        
        # Get all metrics to check if timer was recorded
        all_metrics = self.metrics.get_all_metrics()
        self.assertIn("test_timer.duration", all_metrics)
        self.assertEqual(all_metrics["test_timer.duration"]["type"], "histogram")


class TestAFPTracer(unittest.TestCase):
    """Test the AFPTracer class."""

    def setUp(self):
        """Set up the test case."""
        self.tracer = AFPTracer()

    def test_create_trace(self):
        """Test creating a trace."""
        trace_id = self.tracer.create_trace()
        self.assertIsNotNone(trace_id)
        self.assertIn(trace_id, self.tracer._traces)

    def test_start_span(self):
        """Test starting a span."""
        trace_id = self.tracer.create_trace()
        span_id = self.tracer.start_span("test_span", SpanType.OPERATION, trace_id)
        
        self.assertIsNotNone(span_id)
        self.assertIn(span_id, self.tracer._active_spans)
        self.assertEqual(self.tracer._active_spans[span_id].name, "test_span")
        self.assertEqual(self.tracer._active_spans[span_id].span_type, SpanType.OPERATION)

    def test_end_span(self):
        """Test ending a span."""
        trace_id = self.tracer.create_trace()
        span_id = self.tracer.start_span("test_span", SpanType.OPERATION, trace_id)
        
        time.sleep(0.1)
        self.tracer.end_span(span_id)
        
        # The span should no longer be active
        self.assertNotIn(span_id, self.tracer._active_spans)
        
        # But it should still be in the trace
        spans = self.tracer._traces[trace_id]
        span = next((s for s in spans if s.span_id == span_id), None)
        self.assertIsNotNone(span)
        self.assertIsNotNone(span.end_time)
        self.assertGreaterEqual(span.duration(), 0.1)

    def test_add_event(self):
        """Test adding an event to a span."""
        trace_id = self.tracer.create_trace()
        span_id = self.tracer.start_span("test_span", SpanType.OPERATION, trace_id)
        
        self.tracer.add_event(span_id, "test_event")
        
        span = self.tracer._active_spans[span_id]
        self.assertEqual(len(span.events), 1)
        self.assertEqual(span.events[0]["name"], "test_event")

    def test_add_attribute(self):
        """Test adding an attribute to a span."""
        trace_id = self.tracer.create_trace()
        span_id = self.tracer.start_span("test_span", SpanType.OPERATION, trace_id)
        
        self.tracer.add_attribute(span_id, "test_key", "test_value")
        
        span = self.tracer._active_spans[span_id]
        self.assertEqual(span.attributes["test_key"], "test_value")

    def test_get_trace(self):
        """Test getting a trace."""
        trace_id = self.tracer.create_trace()
        span_id1 = self.tracer.start_span("span1", SpanType.OPERATION, trace_id)
        span_id2 = self.tracer.start_span("span2", SpanType.OPERATION, trace_id)
        
        self.tracer.end_span(span_id1)
        self.tracer.end_span(span_id2)
        
        trace = self.tracer.get_trace(trace_id)
        self.assertEqual(len(trace), 2)


def run_tests():
    """Run the tests."""
    unittest.main()


if __name__ == "__main__":
    run_tests() 