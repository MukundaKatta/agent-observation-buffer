# agent-observation-buffer

Sliding TTL buffer of agent observations with tag-based queries.

Add observations from various sources, query by tag or source, and let stale observations expire automatically. Useful for agents that need to track recent perceptions (sensor readings, user events, API responses) without unbounded growth.

## Install

```bash
pip install agent-observation-buffer
```

## Quick start

```python
from agent_observation_buffer import ObservationBuffer

buf = ObservationBuffer(default_ttl=60.0)  # observations expire after 60 seconds
buf.add("User clicked submit", source="ui",     tags=["user", "form"])
buf.add("API returned 200",    source="api",    tags=["api"])
buf.add("Sensor temp=42",      source="sensor", tags=["temp", "env"])

# Only non-expired observations, newest first
for obs in buf.fresh():
    print(obs.text)

# Filter by tag
api_obs = buf.by_tag("api")

# Filter by source
ui_obs = buf.by_source("ui")

# Remove expired observations
removed_count = buf.expire_stale()
```

## API

### `ObservationBuffer`

```python
ObservationBuffer(*, default_ttl=None, max_size=None, clock=None)
```

| Method | Description |
|---|---|
| `add(text, *, source, tags, ttl)` | Add an observation |
| `fresh()` | Non-expired observations, newest first |
| `stale()` | Expired observations in insertion order |
| `by_tag(tag)` | Fresh observations with `tag` |
| `by_source(source)` | Fresh observations from `source` |
| `by_tags(tags, *, require_all)` | Fresh observations matching tags |
| `get(id)` | Observation by id (including expired) |
| `all()` | All observations (including expired) |
| `count(*, fresh_only=True)` | Count observations |
| `expire_stale()` | Remove expired items; returns count removed |
| `clear()` | Remove all, reset ID counter |
| `to_dict()` / `from_dict(data)` | Serialise/restore |

### `Observation`

```python
@dataclass
class Observation:
    id: int
    text: str
    source: str
    tags: frozenset[str]
    created_at: float
    expires_at: float | None  # None = never expires
```

## License

MIT
