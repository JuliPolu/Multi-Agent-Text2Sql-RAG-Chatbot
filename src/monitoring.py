"""
Observability and Monitoring Module
Handles integration with Langfuse and Phoenix for tracing and metrics.
"""

import os
import logging
from typing import Optional, Dict, Any
from functools import wraps

# Observability imports
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

try:
    from phoenix.trace import TracerProvider
    from phoenix.trace.langchain import LangChainInstrumentor
    import opentelemetry.trace as trace
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringSystem:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MonitoringSystem, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.langfuse = None
        self.tracer = None
        self._setup_observability()
        self._initialized = True

    def _setup_observability(self):
        """Initialize observability tools if configured"""
        
        # Initialize Langfuse
        if LANGFUSE_AVAILABLE and os.getenv("LANGFUSE_PUBLIC_KEY"):
            try:
                self.langfuse = Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                )
                logger.info("✅ Langfuse observability initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Langfuse: {e}")

        # Initialize Phoenix
        if PHOENIX_AVAILABLE and os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
            try:
                tracer_provider = TracerProvider()
                trace.set_tracer_provider(tracer_provider)
                self.tracer = trace.get_tracer(__name__)
                logger.info("✅ Phoenix tracing initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Phoenix: {e}")

    def trace(self, name: str = None, **kwargs):
        """Decorator for tracing functions"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **func_kwargs):
                # Langfuse observation
                if LANGFUSE_AVAILABLE and self.langfuse:
                    # We rely on the @observe decorator from langfuse if used directly,
                    # but here we can add custom logic if needed.
                    # For now, we'll just pass through as this is a simple wrapper.
                    pass
                
                # Phoenix tracing
                span = None
                if self.tracer:
                    span_name = name or func.__name__
                    span = self.tracer.start_span(span_name)
                    for k, v in kwargs.items():
                        span.set_attribute(k, str(v))
                
                try:
                    result = func(*args, **func_kwargs)
                    return result
                except Exception as e:
                    if span:
                        span.record_exception(e)
                    raise
                finally:
                    if span:
                        span.end()
            return wrapper
        return decorator

# Global instance
monitor = MonitoringSystem()
