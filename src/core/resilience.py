"""Resilience patterns - Circuit Breaker and Rate Limiter for external service calls."""

import time
import threading
from enum import Enum
from functools import wraps
from typing import Callable, Any
import structlog

log = structlog.get_logger()


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Thread-safe circuit breaker for external service calls.

    States:
      CLOSED → normal operation, track failures
      OPEN → reject calls immediately, wait for recovery_timeout
      HALF_OPEN → allow one trial call to test recovery
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exceptions: tuple = (Exception,),
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    log.info("circuit_half_open", name=self.name)
            return self._state

    def record_success(self):
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                log.info("circuit_closed", name=self.name)

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                log.warning("circuit_opened", name=self.name, failures=self._failure_count)

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        current_state = self.state
        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit '{self.name}' is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except self.expected_exceptions as e:
            self.record_failure()
            raise

    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            log.info("circuit_reset", name=self.name)


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and rejects the call."""
    pass


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Allows burst up to max_tokens, refills at rate tokens/second.
    Thread-safe.
    """

    def __init__(self, name: str, max_tokens: int = 30, refill_rate: float = 10.0):
        self.name = name
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second

        self._tokens = float(max_tokens)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def acquire(self, tokens: int = 1, timeout: float = 5.0) -> bool:
        """Try to acquire tokens. Returns True if successful, False if rate limited."""
        deadline = time.time() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
            if time.time() >= deadline:
                log.warning("rate_limited", name=self.name, requested=tokens)
                return False
            time.sleep(0.05)  # Brief sleep before retry

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with rate limiting."""
        if not self.acquire():
            raise RateLimitExceededError(f"Rate limit exceeded for '{self.name}'")
        return func(*args, **kwargs)


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    pass


# ─── Pre-configured instances for external services ─────────────────────────

# Groq LLM circuit breaker (trips after 5 failures, recovers after 30s)
llm_circuit = CircuitBreaker(
    name="groq_llm",
    failure_threshold=5,
    recovery_timeout=30.0,
)

# Groq rate limiter (30 requests burst, 10/sec refill)
llm_rate_limiter = RateLimiter(
    name="groq_llm",
    max_tokens=30,
    refill_rate=10.0,
)

# Redis circuit breaker (trips after 3 failures, recovers after 10s)
redis_circuit = CircuitBreaker(
    name="redis",
    failure_threshold=3,
    recovery_timeout=10.0,
)

# Database circuit breaker
db_circuit = CircuitBreaker(
    name="database",
    failure_threshold=3,
    recovery_timeout=15.0,
)

# Embedding model circuit breaker
embedding_circuit = CircuitBreaker(
    name="embedding",
    failure_threshold=3,
    recovery_timeout=20.0,
)


def with_circuit_breaker(circuit: CircuitBreaker):
    """Decorator to wrap a function with circuit breaker protection."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return circuit.call(func, *args, **kwargs)
        return wrapper
    return decorator


def with_rate_limiter(limiter: RateLimiter):
    """Decorator to wrap a function with rate limiting."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return limiter.call(func, *args, **kwargs)
        return wrapper
    return decorator


def resilient_call(
    func: Callable,
    *args,
    circuit: CircuitBreaker | None = None,
    rate_limiter: RateLimiter | None = None,
    fallback: Any = None,
    **kwargs,
) -> Any:
    """Execute a function with combined circuit breaker + rate limiter protection.

    Args:
        func: The function to call
        circuit: Optional circuit breaker
        rate_limiter: Optional rate limiter
        fallback: Value to return if call fails (None raises exception)
    """
    try:
        if rate_limiter and not rate_limiter.acquire():
            if fallback is not None:
                log.warning("resilient_call_rate_limited", func=func.__name__)
                return fallback
            raise RateLimitExceededError(f"Rate limit exceeded for {func.__name__}")

        if circuit:
            return circuit.call(func, *args, **kwargs)
        return func(*args, **kwargs)
    except (CircuitBreakerOpenError, RateLimitExceededError):
        if fallback is not None:
            log.warning("resilient_call_fallback", func=func.__name__)
            return fallback
        raise
    except Exception:
        if fallback is not None:
            return fallback
        raise
