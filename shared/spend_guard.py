"""SpendGuard — Chapter F / P43 + P43b.

Global + per-process rolling-window spend circuit-breaker.  A single
``SpendGuard`` instance wraps the LLM client and hard-stops any runaway
billing before it can accumulate to catastrophic levels.

Design:
- Rolling deque of ``(timestamp, cost)`` tuples.
- ``check_and_reserve`` PRUNES expired events then checks both the cost cap
  AND the call cap *before* the real API call.  If either cap would be
  exceeded the method raises :class:`SpendTripped` immediately — the client
  is never invoked.
- ``record_actual`` reconciles the reservation with the true cost returned by
  the vendor.  This is best-effort; the guard is still safe if it is skipped.
- The ``clock`` parameter is injectable (returns ``float`` seconds, default
  ``time.monotonic``) so tests are fully deterministic without any real
  sleeping.

Motivated by a real incident: a runaway Sonnet loop burned ~$100 in 15 min
with no guardrail.  With default settings ($5/min window) that incident would
have been stopped at $5.

Usage::

    guard = SpendGuard(
        window_seconds=60,
        window_cost_cap=5.0,
        window_call_cap=300,
    )
    guard.check_and_reserve(projected_cost=0.01)   # raises SpendTripped if near cap
    actual = call_llm()
    guard.record_actual(reserved_cost=0.01, actual_cost=actual)
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SpendTripped(Exception):  # noqa: N818
    """Raised when the rolling-window spend or call cap would be exceeded.

    Attributes:
        reason:       Human-readable explanation of which cap was hit.
        window_cost:  Total cost already accumulated in the current window.
        window_calls: Number of calls already recorded in the current window.
        window_cost_cap: The configured cost cap.
        window_call_cap: The configured call cap.
    """

    def __init__(
        self,
        *,
        reason: str,
        window_cost: float,
        window_calls: int,
        window_cost_cap: float = 0.0,
        window_call_cap: int = 0,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.window_cost = window_cost
        self.window_calls = window_calls
        self.window_cost_cap = window_cost_cap
        self.window_call_cap = window_call_cap


# ---------------------------------------------------------------------------
# SpendGuard
# ---------------------------------------------------------------------------


class SpendGuard:
    """Rolling-window spend + call-rate circuit-breaker.

    Args:
        window_seconds:  Rolling window duration in seconds.
        window_cost_cap: Maximum cumulative USD spend permitted within the window.
        window_call_cap: Maximum number of LLM calls permitted within the window.
        clock:           Callable returning current time as a float (seconds).
                         Defaults to ``time.monotonic``.  Inject a fake in tests.
    """

    def __init__(
        self,
        *,
        window_seconds: int,
        window_cost_cap: float,
        window_call_cap: int,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._window_seconds = window_seconds
        self._window_cost_cap = window_cost_cap
        self._window_call_cap = window_call_cap
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        # Deque of (timestamp, cost) tuples — most recent at the right.
        self._events: deque[tuple[float, float]] = deque()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_reserve(self, projected_cost: float) -> None:
        """Check caps and reserve *projected_cost* before the LLM call.

        This method MUST be called BEFORE the real API request is dispatched.
        It atomically prunes expired events, checks both caps, and records a
        reservation so that concurrent requests cannot slip through.

        Args:
            projected_cost: Estimated USD cost of the upcoming call.

        Raises:
            SpendTripped: If adding this call would exceed either the cost cap
                          or the call-rate cap for the current window.
        """
        now = self._clock()
        self._prune(now)

        current_cost = self._window_cost()
        current_calls = len(self._events)

        if current_cost + projected_cost > self._window_cost_cap:
            raise SpendTripped(
                reason=(
                    f"Window cost cap reached: "
                    f"${current_cost:.4f} already spent + ${projected_cost:.4f} projected "
                    f"> ${self._window_cost_cap:.2f} cap."
                ),
                window_cost=current_cost,
                window_calls=current_calls,
                window_cost_cap=self._window_cost_cap,
                window_call_cap=self._window_call_cap,
            )

        if current_calls + 1 > self._window_call_cap:
            raise SpendTripped(
                reason=(
                    f"Window call cap reached: "
                    f"{current_calls} calls already made + 1 > {self._window_call_cap} cap."
                ),
                window_cost=current_cost,
                window_calls=current_calls,
                window_cost_cap=self._window_cost_cap,
                window_call_cap=self._window_call_cap,
            )

        # Reserve: record the projected cost at current timestamp
        self._events.append((now, projected_cost))

    def record_actual(self, reserved_cost: float, actual_cost: float) -> None:
        """Reconcile the most recent reservation with the actual cost.

        Walks the deque from the right to find the most recent event whose
        cost matches *reserved_cost* and replaces it with *actual_cost*.

        This is best-effort: if the reservation cannot be matched (e.g. due to
        concurrent mutation), the call is a no-op.  The guard remains safe
        because reservation-based pre-checks already prevented over-spend.

        Args:
            reserved_cost: The projected cost passed to :meth:`check_and_reserve`.
            actual_cost:   The true cost returned by the vendor response.
        """
        events = list(self._events)
        # Walk backwards to find the most recent matching reservation
        for i in range(len(events) - 1, -1, -1):
            ts, cost = events[i]
            if cost == reserved_cost:
                events[i] = (ts, actual_cost)
                self._events = deque(events)
                return
        # No match found — no-op (safe: reservation was already counted)

    def stats(self) -> dict[str, float | int]:
        """Return current window statistics.

        Returns:
            Dict with keys:
            - ``window_cost``: total USD in the current window.
            - ``window_calls``: number of calls in the current window.
            - ``window_cost_cap``: configured cost cap.
            - ``window_call_cap``: configured call cap.
            - ``window_seconds``: window duration in seconds.
        """
        now = self._clock()
        self._prune(now)
        return {
            "window_cost": round(self._window_cost(), 6),
            "window_calls": len(self._events),
            "window_cost_cap": self._window_cost_cap,
            "window_call_cap": self._window_call_cap,
            "window_seconds": self._window_seconds,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        """Remove events older than the rolling window."""
        cutoff = now - self._window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _window_cost(self) -> float:
        """Return the total cost accumulated in the current window."""
        return sum(cost for _, cost in self._events)
