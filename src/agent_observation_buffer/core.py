"""Sliding TTL buffer of agent observations with tag-based queries.

:class:`ObservationBuffer` accumulates :class:`Observation` objects and
supports TTL-based expiry, tag filtering, and source filtering.  Call
:meth:`~ObservationBuffer.expire_stale` to remove expired items explicitly,
or rely on :meth:`~ObservationBuffer.fresh` which filters them on the fly.

Example::

    buf = ObservationBuffer(default_ttl=60.0)
    buf.add("User clicked submit", source="ui",  tags=["user", "form"])
    buf.add("API returned 200",    source="api", tags=["api"])

    for obs in buf.fresh():
        print(obs.text)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


class ObservationNotFoundError(KeyError):
    """Raised when an observation ID is not found."""

    def __init__(self, obs_id: int) -> None:
        self.obs_id = obs_id
        super().__init__(f"Observation {obs_id!r} not found.")


@dataclass
class Observation:
    """A single buffered observation.

    Attributes:
        id:         Auto-assigned integer (1-based).
        text:       Content of the observation.
        source:     Where it came from (e.g. ``"ui"``, ``"api"``).
        tags:       Immutable set of string labels.
        created_at: Unix timestamp of creation.
        expires_at: Unix timestamp after which the observation is stale,
                    or ``None`` for never-expiring observations.
    """

    id: int
    text: str
    source: str = ""
    tags: frozenset[str] = field(default_factory=frozenset)
    created_at: float = 0.0
    expires_at: float | None = None

    def is_fresh(self, now: float) -> bool:
        """Return ``True`` if the observation has not yet expired."""
        if self.expires_at is None:
            return True
        return now < self.expires_at

    def is_expired(self, now: float) -> bool:
        """Return ``True`` if the observation has expired."""
        return not self.is_fresh(now)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "tags": sorted(self.tags),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        """Reconstruct an :class:`Observation` from a plain dict."""
        return cls(
            id=int(data["id"]),
            text=data["text"],
            source=data.get("source", ""),
            tags=frozenset(data.get("tags", [])),
            created_at=float(data.get("created_at", 0.0)),
            expires_at=(
                float(data["expires_at"])
                if data.get("expires_at") is not None
                else None
            ),
        )

    def __repr__(self) -> str:
        preview = self.text[:40] + "..." if len(self.text) > 40 else self.text
        return f"Observation(id={self.id}, source={self.source!r}, text={preview!r})"


class ObservationBuffer:
    """A TTL-aware buffer of agent observations.

    Args:
        default_ttl: Default seconds until an observation expires.
            ``None`` means observations never expire by default.
        max_size:    If set, the buffer keeps only the most-recent
            *max_size* observations (FIFO eviction when exceeded).
        clock:       Callable returning current Unix time.

    Example::

        buf = ObservationBuffer(default_ttl=30.0)
        buf.add("sensor reading 42", source="sensor", tags=["temp"])
        fresh = buf.fresh()  # only non-expired items
    """

    def __init__(
        self,
        *,
        default_ttl: float | None = None,
        max_size: int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._items: list[Observation] = []
        self._next_id: int = 1
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._clock: Callable[[], float] = clock if clock is not None else time.time

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        *,
        source: str = "",
        tags: list[str] | tuple[str, ...] | None = None,
        ttl: float | None = None,
    ) -> Observation:
        """Add an observation to the buffer.

        Args:
            text:   Content of the observation.
            source: Origin label (e.g. ``"sensor"``, ``"user"``, ``"api"``).
            tags:   Iterable of string labels for filtering.
            ttl:    Seconds until this observation expires.  Falls back to
                    *default_ttl*; ``None`` means never expires.

        Returns:
            The new :class:`Observation`.
        """
        now = self._clock()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (now + effective_ttl) if effective_ttl is not None else None
        obs = Observation(
            id=self._next_id,
            text=text,
            source=source,
            tags=frozenset(tags or []),
            created_at=now,
            expires_at=expires_at,
        )
        self._items.append(obs)
        self._next_id += 1
        # Enforce max_size by evicting oldest first
        if self._max_size is not None and len(self._items) > self._max_size:
            self._items.pop(0)
        return obs

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, obs_id: int) -> Observation:
        """Return the observation with *obs_id* (including expired ones).

        Raises:
            ObservationNotFoundError: If not found.
        """
        for obs in self._items:
            if obs.id == obs_id:
                return obs
        raise ObservationNotFoundError(obs_id)

    def all(self) -> list[Observation]:
        """All observations in insertion order (including expired)."""
        return list(self._items)

    def fresh(self) -> list[Observation]:
        """Non-expired observations, newest first."""
        now = self._clock()
        return [o for o in reversed(self._items) if o.is_fresh(now)]

    def stale(self) -> list[Observation]:
        """Expired observations in insertion order."""
        now = self._clock()
        return [o for o in self._items if o.is_expired(now)]

    def by_tag(self, tag: str) -> list[Observation]:
        """Non-expired observations that have *tag*, newest first."""
        now = self._clock()
        return [o for o in reversed(self._items) if o.is_fresh(now) and tag in o.tags]

    def by_source(self, source: str) -> list[Observation]:
        """Non-expired observations from *source*, newest first."""
        now = self._clock()
        return [
            o for o in reversed(self._items) if o.is_fresh(now) and o.source == source
        ]

    def by_tags(
        self, tags: list[str] | tuple[str, ...], *, require_all: bool = False
    ) -> list[Observation]:
        """Non-expired observations matching *tags*, newest first.

        Args:
            tags:        Labels to match.
            require_all: If ``True``, all tags must match; otherwise any
                         tag is sufficient.
        """
        tag_set = frozenset(tags)
        now = self._clock()
        if require_all:
            check = lambda o: tag_set.issubset(o.tags)  # noqa: E731
        else:
            check = lambda o: bool(tag_set & o.tags)  # noqa: E731
        return [o for o in reversed(self._items) if o.is_fresh(now) and check(o)]

    def count(self, *, fresh_only: bool = True) -> int:
        """Count observations, optionally limiting to non-expired items."""
        if not fresh_only:
            return len(self._items)
        now = self._clock()
        return sum(1 for o in self._items if o.is_fresh(now))

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, obs_id: int) -> bool:
        return any(o.id == obs_id for o in self._items)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def expire_stale(self) -> int:
        """Remove expired observations.

        Returns:
            Number of observations removed.
        """
        now = self._clock()
        before = len(self._items)
        self._items = [o for o in self._items if o.is_fresh(now)]
        return before - len(self._items)

    def clear(self) -> None:
        """Remove all observations and reset the ID counter."""
        self._items.clear()
        self._next_id = 1

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the buffer to a plain dict."""
        return {
            "default_ttl": self._default_ttl,
            "max_size": self._max_size,
            "next_id": self._next_id,
            "items": [o.to_dict() for o in self._items],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        clock: Callable[[], float] | None = None,
    ) -> ObservationBuffer:
        """Reconstruct an :class:`ObservationBuffer` from a plain dict."""
        buf = cls(
            default_ttl=data.get("default_ttl"),
            max_size=data.get("max_size"),
            clock=clock,
        )
        for d in data.get("items", []):
            obs = Observation.from_dict(d)
            buf._items.append(obs)
        if "next_id" in data:
            buf._next_id = int(data["next_id"])
        else:
            # Derive from the highest existing id so freshly added items can
            # never collide with restored ones (ids may be non-contiguous,
            # e.g. after max_size eviction).
            buf._next_id = max((o.id for o in buf._items), default=0) + 1
        return buf

    def __repr__(self) -> str:
        fresh = self.count(fresh_only=True)
        return f"ObservationBuffer(total={len(self._items)}, fresh={fresh})"
