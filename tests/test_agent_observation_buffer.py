"""Tests for agent-observation-buffer."""

from __future__ import annotations

import pytest

from agent_observation_buffer import Observation, ObservationBuffer
from agent_observation_buffer.core import ObservationNotFoundError

# ---------------------------------------------------------------------------
# Observation — construction and serialisation
# ---------------------------------------------------------------------------


def test_observation_minimal():
    obs = Observation(id=1, text="Hello.")
    assert obs.id == 1
    assert obs.text == "Hello."
    assert obs.source == ""
    assert obs.tags == frozenset()
    assert obs.expires_at is None


def test_observation_with_fields():
    obs = Observation(
        id=2,
        text="Reading.",
        source="sensor",
        tags=frozenset(["temp", "env"]),
        created_at=1.0,
        expires_at=61.0,
    )
    assert obs.source == "sensor"
    assert "temp" in obs.tags
    assert obs.expires_at == 61.0


def test_observation_is_fresh_no_expiry():
    obs = Observation(id=1, text="x")
    assert obs.is_fresh(now=9999.0)


def test_observation_is_fresh_not_expired():
    obs = Observation(id=1, text="x", expires_at=100.0)
    assert obs.is_fresh(now=50.0)
    assert not obs.is_fresh(now=100.0)


def test_observation_is_expired():
    obs = Observation(id=1, text="x", expires_at=10.0)
    assert obs.is_expired(now=15.0)
    assert not obs.is_expired(now=5.0)


def test_observation_to_dict():
    obs = Observation(
        id=1,
        text="T",
        source="src",
        tags=frozenset(["a", "b"]),
        created_at=0.0,
        expires_at=30.0,
    )
    d = obs.to_dict()
    assert d["id"] == 1
    assert d["text"] == "T"
    assert d["source"] == "src"
    assert sorted(d["tags"]) == ["a", "b"]
    assert d["expires_at"] == 30.0


def test_observation_from_dict_round_trip():
    obs = Observation(
        id=3,
        text="Round trip.",
        source="test",
        tags=frozenset(["x"]),
        created_at=5.0,
        expires_at=None,
    )
    restored = Observation.from_dict(obs.to_dict())
    assert restored.id == obs.id
    assert restored.text == obs.text
    assert restored.source == obs.source
    assert restored.tags == obs.tags
    assert restored.expires_at == obs.expires_at


def test_observation_repr():
    obs = Observation(id=1, text="Short", source="s")
    r = repr(obs)
    assert "1" in r
    assert "Short" in r


def test_observation_repr_truncated():
    obs = Observation(id=1, text="x" * 50)
    assert "..." in repr(obs)


# ---------------------------------------------------------------------------
# ObservationBuffer — add
# ---------------------------------------------------------------------------


def test_add_returns_observation():
    buf = ObservationBuffer(clock=lambda: 0.0)
    obs = buf.add("Hello.")
    assert obs.id == 1
    assert obs.text == "Hello."


def test_add_assigns_incremental_ids():
    buf = ObservationBuffer(clock=lambda: 0.0)
    o1 = buf.add("A")
    o2 = buf.add("B")
    assert o1.id == 1
    assert o2.id == 2


def test_add_with_source_and_tags():
    buf = ObservationBuffer(clock=lambda: 0.0)
    obs = buf.add("T", source="api", tags=["a", "b"])
    assert obs.source == "api"
    assert obs.tags == frozenset(["a", "b"])


def test_add_default_ttl():
    t = [0.0]
    buf = ObservationBuffer(default_ttl=10.0, clock=lambda: t[0])
    obs = buf.add("With ttl.")
    assert obs.expires_at == pytest.approx(10.0)


def test_add_per_item_ttl_overrides_default():
    t = [0.0]
    buf = ObservationBuffer(default_ttl=10.0, clock=lambda: t[0])
    obs = buf.add("Short.", ttl=2.0)
    assert obs.expires_at == pytest.approx(2.0)


def test_add_no_expiry_when_no_ttl():
    buf = ObservationBuffer(clock=lambda: 0.0)
    obs = buf.add("No expiry.")
    assert obs.expires_at is None


def test_max_size_evicts_oldest():
    buf = ObservationBuffer(max_size=2, clock=lambda: 0.0)
    o1 = buf.add("A")
    buf.add("B")
    buf.add("C")
    assert len(buf) == 2
    assert o1.id not in buf


# ---------------------------------------------------------------------------
# ObservationBuffer — retrieval
# ---------------------------------------------------------------------------


def test_get():
    buf = ObservationBuffer(clock=lambda: 0.0)
    obs = buf.add("Hello.")
    assert buf.get(obs.id) is obs


def test_get_missing_raises():
    buf = ObservationBuffer()
    with pytest.raises(ObservationNotFoundError) as exc_info:
        buf.get(999)
    assert exc_info.value.obs_id == 999


def test_all():
    buf = ObservationBuffer(clock=lambda: 0.0)
    buf.add("A")
    buf.add("B")
    assert len(buf.all()) == 2


def test_fresh_excludes_expired():
    t = [0.0]
    buf = ObservationBuffer(clock=lambda: t[0])
    buf.add("Fresh.", ttl=100.0)
    buf.add("Stale.", ttl=1.0)
    t[0] = 50.0  # advance time past stale TTL
    fresh = buf.fresh()
    assert len(fresh) == 1
    assert fresh[0].text == "Fresh."


def test_fresh_newest_first():
    buf = ObservationBuffer(clock=lambda: 0.0)
    buf.add("First.")
    buf.add("Second.")
    fresh = buf.fresh()
    assert fresh[0].text == "Second."
    assert fresh[1].text == "First."


def test_fresh_no_ttl_all_included():
    buf = ObservationBuffer(clock=lambda: 999.0)
    buf.add("A")
    buf.add("B")
    assert len(buf.fresh()) == 2


def test_stale():
    t = [0.0]
    buf = ObservationBuffer(clock=lambda: t[0])
    buf.add("Stale.", ttl=1.0)
    buf.add("Fresh.", ttl=100.0)
    t[0] = 50.0
    stale = buf.stale()
    assert len(stale) == 1
    assert stale[0].text == "Stale."


def test_by_tag():
    buf = ObservationBuffer(clock=lambda: 0.0)
    buf.add("Tagged.", tags=["alpha"])
    buf.add("Other.", tags=["beta"])
    result = buf.by_tag("alpha")
    assert len(result) == 1
    assert result[0].text == "Tagged."


def test_by_tag_excludes_expired():
    t = [0.0]
    buf = ObservationBuffer(clock=lambda: t[0])
    buf.add("Expired tagged.", tags=["x"], ttl=1.0)
    buf.add("Fresh tagged.", tags=["x"])
    t[0] = 10.0
    result = buf.by_tag("x")
    assert len(result) == 1
    assert result[0].text == "Fresh tagged."


def test_by_source():
    buf = ObservationBuffer(clock=lambda: 0.0)
    buf.add("API.", source="api")
    buf.add("UI.", source="ui")
    result = buf.by_source("api")
    assert len(result) == 1
    assert result[0].text == "API."


def test_by_tags_any():
    buf = ObservationBuffer(clock=lambda: 0.0)
    buf.add("A+B.", tags=["a", "b"])
    buf.add("A only.", tags=["a"])
    buf.add("C only.", tags=["c"])
    result = buf.by_tags(["a", "c"])
    assert len(result) == 3


def test_by_tags_require_all():
    buf = ObservationBuffer(clock=lambda: 0.0)
    buf.add("A+B.", tags=["a", "b"])
    buf.add("A only.", tags=["a"])
    result = buf.by_tags(["a", "b"], require_all=True)
    assert len(result) == 1
    assert result[0].text == "A+B."


def test_count_fresh_only():
    t = [0.0]
    buf = ObservationBuffer(clock=lambda: t[0])
    buf.add("Fresh.")
    buf.add("Stale.", ttl=1.0)
    t[0] = 10.0
    assert buf.count(fresh_only=True) == 1
    assert buf.count(fresh_only=False) == 2


def test_len():
    buf = ObservationBuffer(clock=lambda: 0.0)
    buf.add("A")
    buf.add("B")
    assert len(buf) == 2


def test_contains():
    buf = ObservationBuffer(clock=lambda: 0.0)
    obs = buf.add("X")
    assert obs.id in buf
    assert 999 not in buf


# ---------------------------------------------------------------------------
# ObservationBuffer — housekeeping
# ---------------------------------------------------------------------------


def test_expire_stale():
    t = [0.0]
    buf = ObservationBuffer(clock=lambda: t[0])
    buf.add("Stale.", ttl=1.0)
    buf.add("Fresh.")
    t[0] = 10.0
    removed = buf.expire_stale()
    assert removed == 1
    assert len(buf) == 1


def test_expire_stale_none_expired():
    buf = ObservationBuffer(clock=lambda: 0.0)
    buf.add("A")
    assert buf.expire_stale() == 0


def test_clear():
    buf = ObservationBuffer(clock=lambda: 0.0)
    buf.add("A")
    buf.clear()
    assert len(buf) == 0
    obs = buf.add("After clear.")
    assert obs.id == 1


# ---------------------------------------------------------------------------
# ObservationBuffer — serialisation
# ---------------------------------------------------------------------------


def test_to_dict_round_trip():
    t = [0.0]
    buf = ObservationBuffer(default_ttl=60.0, clock=lambda: t[0])
    buf.add("A", source="s1", tags=["x"])
    buf.add("B", ttl=5.0)

    restored = ObservationBuffer.from_dict(buf.to_dict())
    assert len(restored) == 2
    assert restored.all()[0].text == "A"
    assert restored.all()[1].expires_at == pytest.approx(5.0)


def test_from_dict_uses_explicit_next_id():
    data = {
        "next_id": 10,
        "items": [
            {"id": 3, "text": "a", "tags": [], "created_at": 0.0, "expires_at": None},
        ],
    }
    buf = ObservationBuffer.from_dict(data, clock=lambda: 0.0)
    assert buf.add("new").id == 10


def test_from_dict_without_next_id_avoids_id_collision():
    # Items can have non-contiguous ids (e.g. after max_size eviction). When
    # ``next_id`` is absent it must be derived from the highest existing id so
    # newly added observations never collide with restored ones.
    data = {
        "max_size": 2,
        "items": [
            {"id": 2, "text": "b", "tags": [], "created_at": 0.0, "expires_at": None},
            {"id": 3, "text": "c", "tags": [], "created_at": 0.0, "expires_at": None},
        ],
    }
    buf = ObservationBuffer.from_dict(data, clock=lambda: 0.0)
    new = buf.add("new")
    assert new.id == 4
    ids = [o.id for o in buf.all()]
    assert len(ids) == len(set(ids))  # no duplicate ids


def test_from_dict_without_next_id_empty():
    buf = ObservationBuffer.from_dict({"items": []}, clock=lambda: 0.0)
    assert buf.add("first").id == 1


def test_repr():
    buf = ObservationBuffer(clock=lambda: 0.0)
    buf.add("Hello.")
    r = repr(buf)
    assert "ObservationBuffer" in r
    assert "1" in r
