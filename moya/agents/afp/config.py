"""
Configuration classes for Agent Flow Protocol.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class AFPConfig:
    """Base configuration for AFP."""
    circuit_breaker_enabled: bool = True
    tracing_enabled: bool = True
    metrics_enabled: bool = True
