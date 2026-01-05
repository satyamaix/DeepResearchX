# DRX Deep Research Platform - Redis Key Patterns

> Redis 7.x Configuration for Session State, Messaging, and Caching

## Overview

This document defines the Redis key patterns and data structures used in the DRX Deep Research platform. Redis serves as the primary store for:

- Real-time session state management
- Agent coordination and metrics
- Circuit breaker state
- Rate limiting
- Server-Sent Events (SSE) streaming
- Task queue management (Celery)
- Distributed caching

## Key Namespace Convention

```
{domain}:{entity_id}:{sub_resource}[:{qualifier}]
```

All keys use colon (`:`) as the delimiter. Entity IDs are UUIDs or well-known identifiers.

---

## 1. Session State Management

### Session State
Stores the current state of a research session for fast access and resumability.

```
Key:     session:{session_id}:state
Type:    HASH
TTL:     24 hours (86400 seconds)
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Current session status (pending, running, paused, completed, failed, cancelled) |
| `progress_pct` | int | Progress percentage (0-100) |
| `current_step_id` | string | UUID of currently executing step |
| `current_agent` | string | Agent type currently active |
| `user_id` | string | Owner user ID |
| `query_hash` | string | SHA256 hash of the research query |
| `started_at` | timestamp | ISO8601 timestamp when session started |
| `last_activity` | timestamp | ISO8601 timestamp of last activity |
| `tokens_used` | int | Total tokens consumed |
| `error_message` | string | Last error message if any |

**Example:**
```redis
HSET session:550e8400-e29b-41d4-a716-446655440000:state \
    status "running" \
    progress_pct 45 \
    current_step_id "660e8400-e29b-41d4-a716-446655440001" \
    current_agent "searcher" \
    user_id "user_123" \
    tokens_used 15000
```

### Session Lock
Distributed lock for session mutations to prevent race conditions.

```
Key:     session:{session_id}:lock
Type:    STRING (with NX)
TTL:     30 seconds
Value:   {worker_id}:{timestamp}
```

**Example:**
```redis
SET session:550e8400-e29b-41d4-a716-446655440000:lock \
    "worker-01:1704067200" NX EX 30
```

### Session Checkpoint
Checkpoint data for session resumability.

```
Key:     session:{session_id}:checkpoint
Type:    STRING (JSON blob)
TTL:     7 days (604800 seconds)
```

### Active Sessions Set
Set of currently active session IDs for a user.

```
Key:     user:{user_id}:active_sessions
Type:    SET
TTL:     None (managed by application)
```

**Example:**
```redis
SADD user:user_123:active_sessions "550e8400-e29b-41d4-a716-446655440000"
SREM user:user_123:active_sessions "550e8400-e29b-41d4-a716-446655440000"
```

---

## 2. DAG Execution State

### DAG Node State
State of individual DAG nodes for parallel execution tracking.

```
Key:     dag:{session_id}:node:{node_id}
Type:    HASH
TTL:     24 hours
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Node status (pending, ready, running, succeeded, failed) |
| `assigned_worker` | string | Worker ID processing this node |
| `started_at` | timestamp | Execution start time |
| `completed_at` | timestamp | Execution completion time |
| `retry_count` | int | Current retry attempt |
| `output_ref` | string | Reference to output data |

### DAG Ready Queue
Priority queue of nodes ready for execution.

```
Key:     dag:{session_id}:ready_queue
Type:    SORTED SET
Score:   Priority (1-10, higher = more priority) * 1000 + (1000000 - created_timestamp_ms % 1000000)
TTL:     24 hours
```

**Example:**
```redis
ZADD dag:550e8400:ready_queue 5001234567 "node_id_1"
ZPOPMAX dag:550e8400:ready_queue  # Get highest priority ready node
```

### DAG Dependency Counter
Tracks remaining dependencies for each node.

```
Key:     dag:{session_id}:deps:{node_id}
Type:    STRING (integer counter)
TTL:     24 hours
```

**Example:**
```redis
SET dag:550e8400:deps:node_3 2       # Node 3 has 2 dependencies
DECR dag:550e8400:deps:node_3        # Decrement when a dependency completes
```

---

## 3. Agent Metrics & State

### Agent Metrics
Real-time metrics for agent performance monitoring.

```
Key:     agent:{agent_id}:metrics
Type:    HASH
TTL:     1 hour (refreshed on activity)
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `total_invocations` | int | Total invocations since last reset |
| `successful_invocations` | int | Successful invocations |
| `failed_invocations` | int | Failed invocations |
| `total_tokens` | int | Total tokens consumed |
| `total_latency_ms` | int | Cumulative latency |
| `avg_latency_ms` | float | Average latency |
| `last_invocation` | timestamp | Last invocation timestamp |
| `error_rate` | float | Current error rate (0.0-1.0) |

**Example:**
```redis
HINCRBY agent:searcher_v1:metrics total_invocations 1
HINCRBY agent:searcher_v1:metrics total_tokens 1500
HINCRBYFLOAT agent:searcher_v1:metrics avg_latency_ms 0.5
```

### Agent Metrics Time Series
Time-series metrics using Redis Streams for historical analysis.

```
Key:     agent:{agent_id}:metrics:stream
Type:    STREAM
TTL:     7 days (XTRIM by time)
```

**Example:**
```redis
XADD agent:searcher_v1:metrics:stream MAXLEN ~ 10000 * \
    invocation_id "inv_123" \
    latency_ms 450 \
    tokens 1500 \
    status "success"
```

### Agent Health
Current health status of agents.

```
Key:     agent:{agent_id}:health
Type:    STRING (JSON)
TTL:     5 minutes
```

**Structure:**
```json
{
  "status": "healthy|degraded|unhealthy",
  "last_check": "2024-01-01T00:00:00Z",
  "consecutive_failures": 0,
  "latency_p99_ms": 500
}
```

---

## 4. Circuit Breaker State

### Circuit Breaker
Implements circuit breaker pattern for fault tolerance.

```
Key:     circuit:{agent_id}
Type:    HASH
TTL:     None (managed by application)
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `state` | string | Circuit state: closed, open, half_open |
| `failure_count` | int | Consecutive failures |
| `success_count` | int | Consecutive successes (in half_open) |
| `last_failure` | timestamp | Last failure timestamp |
| `opened_at` | timestamp | When circuit opened |
| `half_open_at` | timestamp | When circuit transitioned to half_open |

**State Machine:**
```
CLOSED -> (failure_threshold exceeded) -> OPEN
OPEN -> (recovery_timeout elapsed) -> HALF_OPEN
HALF_OPEN -> (success_threshold reached) -> CLOSED
HALF_OPEN -> (failure) -> OPEN
```

**Example:**
```redis
HSET circuit:searcher_v1 state "open" failure_count 5 opened_at "1704067200"
HGET circuit:searcher_v1 state
```

### Circuit Breaker Config
Configuration for circuit breaker behavior.

```
Key:     circuit:{agent_id}:config
Type:    HASH
TTL:     None
```

**Fields:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `failure_threshold` | int | 5 | Failures to open circuit |
| `success_threshold` | int | 3 | Successes to close from half_open |
| `recovery_timeout_sec` | int | 30 | Time before half_open |
| `half_open_max_calls` | int | 3 | Max calls in half_open state |

---

## 5. Rate Limiting

### Rate Limit Counters
Sliding window rate limiting for API endpoints.

```
Key:     ratelimit:{user_id}:{endpoint}
Type:    SORTED SET
Score:   Timestamp (milliseconds)
TTL:     Window duration + buffer
```

**Algorithm:** Sliding Window Log
```redis
# Add request timestamp
ZADD ratelimit:user_123:/api/research * {timestamp_ms}

# Remove old entries outside window
ZREMRANGEBYSCORE ratelimit:user_123:/api/research 0 {window_start_ms}

# Count requests in window
ZCARD ratelimit:user_123:/api/research
```

### Rate Limit Config
Per-endpoint rate limit configuration.

```
Key:     ratelimit:config:{endpoint}
Type:    HASH
TTL:     None
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `requests_per_minute` | int | Max requests per minute |
| `requests_per_hour` | int | Max requests per hour |
| `tokens_per_minute` | int | Max tokens per minute |
| `burst_size` | int | Maximum burst allowance |

### Token Bucket Rate Limiting
Alternative token bucket for high-throughput endpoints.

```
Key:     tokenbucket:{user_id}:{resource}
Type:    HASH
TTL:     1 hour
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `tokens` | float | Available tokens |
| `last_update` | timestamp | Last token update time |
| `capacity` | int | Maximum token capacity |
| `refill_rate` | float | Tokens added per second |

---

## 6. SSE Streaming Channels

### Stream Channel
Pub/Sub channel for Server-Sent Events to clients.

```
Key:     stream:{interaction_id}
Type:    STREAM
TTL:     1 hour (XTRIM by time)
```

**Event Structure:**
```json
{
  "event_type": "step_started|step_completed|progress|error|result",
  "timestamp": "2024-01-01T00:00:00Z",
  "data": {
    "step_id": "...",
    "agent": "...",
    "message": "...",
    "progress_pct": 45
  }
}
```

**Example:**
```redis
# Publish event
XADD stream:interaction_123 MAXLEN ~ 1000 * \
    event_type "step_completed" \
    timestamp "2024-01-01T00:00:00Z" \
    data '{"step_id":"step_1","agent":"searcher","message":"Found 15 sources"}'

# Subscribe and read
XREAD BLOCK 5000 STREAMS stream:interaction_123 $
```

### Stream Subscribers
Track active SSE connections per interaction.

```
Key:     stream:{interaction_id}:subscribers
Type:    SET
TTL:     1 hour
```

**Example:**
```redis
SADD stream:interaction_123:subscribers "conn_abc" "conn_def"
SCARD stream:interaction_123:subscribers  # Count active subscribers
```

### User Stream Index
Map user to their active streams.

```
Key:     user:{user_id}:streams
Type:    SET
TTL:     24 hours
```

---

## 7. Celery Task Queue Keys

### Task Queue
Celery default task queue.

```
Key:     celery
Type:    LIST
TTL:     None
```

### Priority Queues
Priority-based task routing.

```
Key:     celery:priority:{priority}
Type:    LIST
TTL:     None
Priority: high, default, low
```

**Example:**
```
celery:priority:high     # Critical tasks
celery:priority:default  # Normal tasks
celery:priority:low      # Background tasks
```

### Task Results
Celery task result storage.

```
Key:     celery-task-meta-{task_id}
Type:    STRING (JSON)
TTL:     24 hours (configurable)
```

**Structure:**
```json
{
  "status": "PENDING|STARTED|SUCCESS|FAILURE|RETRY",
  "result": {...},
  "traceback": null,
  "date_done": "2024-01-01T00:00:00Z",
  "task_id": "...",
  "children": []
}
```

### Task State
Real-time task state for long-running operations.

```
Key:     celery:task:{task_id}:state
Type:    HASH
TTL:     24 hours
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Task status |
| `progress` | int | Progress percentage |
| `current_step` | string | Current operation |
| `started_at` | timestamp | Start time |
| `worker_id` | string | Assigned worker |

### Worker Heartbeat
Worker health monitoring.

```
Key:     celery:worker:{worker_id}:heartbeat
Type:    STRING
TTL:     60 seconds
Value:   Timestamp
```

### Scheduled Tasks
Celery Beat scheduled task registry.

```
Key:     celery:beat:schedule
Type:    HASH
TTL:     None
```

---

## 8. Caching

### Query Result Cache
Cache for duplicate query detection and fast response.

```
Key:     cache:query:{query_hash}
Type:    STRING (JSON)
TTL:     1 hour
```

### Document Chunk Cache
Cache for frequently accessed document chunks.

```
Key:     cache:chunk:{chunk_id}
Type:    STRING (JSON)
TTL:     30 minutes
```

### Embedding Cache
Cache for computed embeddings.

```
Key:     cache:embedding:{content_hash}
Type:    STRING (binary - vector bytes)
TTL:     24 hours
```

### Source Cache
Cache for fetched web sources.

```
Key:     cache:source:{url_hash}
Type:    HASH
TTL:     6 hours
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `content` | string | Extracted content |
| `title` | string | Page title |
| `fetched_at` | timestamp | Fetch timestamp |
| `status_code` | int | HTTP status code |

---

## 9. Distributed Locks

### Global Lock Pattern
For distributed coordination.

```
Key:     lock:{resource}:{resource_id}
Type:    STRING
TTL:     30 seconds (with renewal)
Value:   {lock_holder_id}:{timestamp}:{random_token}
```

**Lua Script for Safe Lock Release:**
```lua
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
```

---

## 10. Pub/Sub Channels

### Event Channels
For real-time event distribution.

```
Channel: events:session:{session_id}
Purpose: Session-specific events

Channel: events:agent:{agent_id}
Purpose: Agent-specific events

Channel: events:system
Purpose: System-wide broadcasts
```

---

## Memory Management

### Eviction Policy
Configure Redis with `volatile-lru` or `allkeys-lru` eviction.

```
maxmemory 2gb
maxmemory-policy volatile-lru
```

### Key Expiration Best Practices

| Key Pattern | TTL | Rationale |
|-------------|-----|-----------|
| session:*:state | 24h | Active session data |
| session:*:checkpoint | 7d | Recovery data |
| circuit:* | None | Critical state |
| ratelimit:* | 1-60min | Window-based |
| stream:* | 1h | Real-time only |
| cache:* | 30min-24h | Performance optimization |
| celery-task-meta-* | 24h | Result retrieval |

---

## Monitoring Keys

### Metrics Export
Keys for Prometheus/metrics export.

```
Key:     metrics:redis:commands
Type:    STREAM
Purpose: Command latency tracking

Key:     metrics:redis:memory
Type:    HASH
Purpose: Memory usage snapshots
```

---

## Usage Examples

### Session Lifecycle
```python
import redis.asyncio as redis

async def start_session(r: redis.Redis, session_id: str, user_id: str):
    pipe = r.pipeline()

    # Initialize session state
    pipe.hset(f"session:{session_id}:state", mapping={
        "status": "running",
        "progress_pct": 0,
        "user_id": user_id,
        "started_at": datetime.utcnow().isoformat()
    })
    pipe.expire(f"session:{session_id}:state", 86400)

    # Add to user's active sessions
    pipe.sadd(f"user:{user_id}:active_sessions", session_id)

    await pipe.execute()
```

### Rate Limiting Check
```python
async def check_rate_limit(r: redis.Redis, user_id: str, endpoint: str) -> bool:
    key = f"ratelimit:{user_id}:{endpoint}"
    now = time.time() * 1000
    window_start = now - 60000  # 1 minute window

    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)
    pipe.zadd(key, {str(now): now})
    pipe.zcard(key)
    pipe.expire(key, 120)

    results = await pipe.execute()
    count = results[2]

    return count <= 60  # 60 requests per minute
```

### Circuit Breaker Check
```python
async def is_circuit_open(r: redis.Redis, agent_id: str) -> bool:
    state = await r.hget(f"circuit:{agent_id}", "state")
    return state == "open"
```

---

## 11. Context Store (WP-M3 Context Propagation)

The Context Store provides storage for ResearchContext objects used in context propagation between agents and iterations. Supports Redis for active contexts with TTL and PostgreSQL for persistent storage.

### Key Namespace

All Context Store keys use the `drx:context:` prefix:

```
drx:context:{resource}:{identifier}
```

### Context Data

Stores the serialized ResearchContext object.

```
Key:     drx:context:ctx:{context_id}
Type:    STRING (JSON)
TTL:     Configurable (default 3600 seconds)
```

**Structure:**
```json
{
  "context_id": "ctx-550e8400-e29b-41d4-a716-446655440000",
  "session_id": "session-123",
  "summary": "Compressed context summary for propagation",
  "key_entities": ["entity1", "entity2"],
  "relevance_vector": [0.1, 0.2, ...],
  "chunk_refs": ["chunk-1", "chunk-2"],
  "created_at": "2024-01-01T00:00:00Z",
  "ttl_seconds": 3600
}
```

**Example:**
```redis
SETEX drx:context:ctx:ctx-550e8400 3600 '{"context_id":"ctx-550e8400","session_id":"session-123",...}'
GET drx:context:ctx:ctx-550e8400
```

### Context Metadata

Stores optional metadata for a research context.

```
Key:     drx:context:meta:{context_id}
Type:    HASH
TTL:     Same as associated context
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `source_agent` | string | Agent that created this context |
| `iteration_number` | int | Iteration number in research cycle |
| `parent_context_id` | string | Parent context for lineage tracking |
| `compression_ratio` | float | Ratio of compression applied |
| `original_token_count` | int | Token count before compression |
| `compressed_token_count` | int | Token count after compression |
| `status` | string | Context status (active, expired, archived) |
| `updated_at` | timestamp | Last update timestamp |

**Example:**
```redis
HSET drx:context:meta:ctx-550e8400 \
    source_agent "planner" \
    iteration_number 2 \
    compression_ratio 0.35 \
    status "active"
```

### Session Context Index

Sorted set tracking all contexts for a session, ordered by creation time.

```
Key:     drx:context:session:{session_id}
Type:    SORTED SET
Score:   Unix timestamp of context creation
TTL:     Extended to match longest context TTL
```

**Example:**
```redis
ZADD drx:context:session:session-123 1704067200 "ctx-550e8400"
ZREVRANGE drx:context:session:session-123 0 9  # Get 10 most recent contexts
```

---

## 12. Policy Firewall (WP-M6 Metadata Firewall)

The Policy Firewall provides policy enforcement for tool invocations including budget tracking, rate limiting, and violation event streaming.

### Key Namespace

Policy Firewall keys use the `drx:policy:` and `drx:events:` prefixes:

```
drx:policy:{policy_type}:{agent_id}[:{session_id}]
drx:events:{event_type}
```

### Budget Tracking

Tracks spending per agent, optionally scoped to session.

```
Key:     drx:policy:budget:{agent_id}
Type:    STRING (float)
TTL:     None (application managed)
Value:   Total spend in USD
```

**Session-Scoped Budget:**
```
Key:     drx:policy:budget:{agent_id}:{session_id}
Type:    STRING (float)
TTL:     24 hours (86400 seconds)
Value:   Session spend in USD
```

**Example:**
```redis
# Record spend for agent
INCRBYFLOAT drx:policy:budget:searcher_v1 0.0015

# Record session-scoped spend
INCRBYFLOAT drx:policy:budget:searcher_v1:session-123 0.0015
EXPIRE drx:policy:budget:searcher_v1:session-123 86400

# Get current spend
GET drx:policy:budget:searcher_v1
```

### Rate Limit Counters

Tracks request and token counts per agent for rate limiting.

```
Key:     drx:policy:ratelimit:{agent_id}
Type:    HASH
TTL:     60 seconds (sliding window)
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `requests_count` | int | Number of requests in current window |
| `requests_limit` | int | Maximum requests per minute |
| `tokens_count` | int | Number of tokens in current window |
| `tokens_limit` | int | Maximum tokens per minute |
| `window_start` | timestamp | ISO8601 timestamp of window start |

**Example:**
```redis
HINCRBY drx:policy:ratelimit:searcher_v1 requests_count 1
HINCRBY drx:policy:ratelimit:searcher_v1 tokens_count 1500
HSET drx:policy:ratelimit:searcher_v1 window_start "2024-01-01T00:00:00Z"
EXPIRE drx:policy:ratelimit:searcher_v1 60
```

### Policy Violation Event Stream

Stream of policy violation events for real-time monitoring.

```
Key:     drx:events:policy_violations
Type:    STREAM
TTL:     None (MAXLEN trimmed to 10000 entries)
```

**Event Structure:**
```json
{
  "event_type": "policy_violated",
  "violation_id": "vio-550e8400-e29b-41d4-a716-446655440000",
  "agent_id": "searcher_v1",
  "violation_type": "budget_exceeded|rate_limited|domain_blocked|capability_denied",
  "severity": "warning|error|critical",
  "message": "Human-readable violation description",
  "session_id": "session-123",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example:**
```redis
# Add violation event
XADD drx:events:policy_violations MAXLEN ~ 10000 * \
    event_type "policy_violated" \
    violation_id "vio-550e8400" \
    agent_id "searcher_v1" \
    violation_type "budget_exceeded" \
    severity "critical" \
    message "Agent exceeded budget: $1.05 > $1.00" \
    session_id "session-123" \
    timestamp "2024-01-01T00:00:00Z"

# Read recent violations
XREVRANGE drx:events:policy_violations + - COUNT 10
```

---

## 13. Active State Service (Agentic Metadata - R10.3)

The Active State Service provides real-time agent health monitoring, metrics tracking, and circuit breaker state management. This implements the Agentic Metadata functionality from R10.3 of the DRX spec.

### Key Namespace

All Active State Service keys use the `drx:agent:` prefix:

```
drx:agent:{agent_id}:{resource}
```

### Agent Invocations
Time-series record of agent invocations for metrics computation.

```
Key:     drx:agent:{agent_id}:invocations
Type:    SORTED SET
Score:   Unix timestamp (float)
Value:   JSON-encoded InvocationRecord
TTL:     1 hour (3600 seconds)
```

**InvocationRecord Structure:**
```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "tokens_used": 1500,
  "latency_ms": 250,
  "success": true,
  "error_type": null
}
```

**Example:**
```redis
ZADD drx:agent:planner:invocations 1704067200.123 '{"timestamp":"2024-01-01T00:00:00Z","tokens_used":1500,"latency_ms":250,"success":true,"error_type":null}'

# Query invocations in last 60 seconds
ZRANGEBYSCORE drx:agent:planner:invocations (now-60) +inf
```

### Agent Errors
Time-series record of agent errors for error rate computation.

```
Key:     drx:agent:{agent_id}:errors
Type:    SORTED SET
Score:   Unix timestamp (float)
Value:   {timestamp}:{error_type}
TTL:     1 hour (3600 seconds)
```

**Example:**
```redis
ZADD drx:agent:searcher:errors 1704067200.123 "1704067200.123:timeout"

# Count errors in last 5 minutes
ZCOUNT drx:agent:searcher:errors (now-300) +inf
```

### Agent Health Status
Current health state and failure tracking.

```
Key:     drx:agent:{agent_id}:health
Type:    HASH
TTL:     None (application managed)
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Current status: healthy, degraded, unhealthy |
| `failure_count` | int | Consecutive failure count |
| `last_check` | timestamp | ISO8601 timestamp of last health check |
| `last_success` | timestamp | ISO8601 timestamp of last successful invocation |
| `last_failure` | timestamp | ISO8601 timestamp of last failed invocation |

**Example:**
```redis
HSET drx:agent:planner:health \
    status "healthy" \
    failure_count 0 \
    last_check "2024-01-01T00:00:00Z" \
    last_success "2024-01-01T00:00:00Z"

HINCRBY drx:agent:planner:health failure_count 1
```

### Agent Metrics Cache
Pre-computed metrics from background aggregation.

```
Key:     drx:agent:{agent_id}:metrics
Type:    HASH
TTL:     5 minutes (300 seconds)
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `tokens_1m` | int | Tokens consumed in last 1 minute |
| `tokens_5m` | int | Tokens consumed in last 5 minutes |
| `requests_1m` | int | Requests in last 1 minute |
| `latency_p50` | float | 50th percentile latency (ms) |
| `latency_p99` | float | 99th percentile latency (ms) |
| `error_rate_5m` | float | Error rate in last 5 minutes (0.0-1.0) |
| `last_updated` | timestamp | ISO8601 timestamp of last update |

**Example:**
```redis
HSET drx:agent:planner:metrics \
    tokens_1m 15000 \
    tokens_5m 75000 \
    requests_1m 10 \
    latency_p50 250.5 \
    latency_p99 890.0 \
    error_rate_5m 0.02 \
    last_updated "2024-01-01T00:00:00Z"
```

### Agent Circuit Breaker State
Circuit breaker state for fault tolerance.

```
Key:     drx:agent:{agent_id}:circuit
Type:    STRING
TTL:     None (application managed)
Value:   closed | open | half_open
```

**Example:**
```redis
SET drx:agent:planner:circuit "closed"
GET drx:agent:planner:circuit
```

### Circuit Breaker Opened At
Timestamp when circuit was opened (for recovery timeout calculation).

```
Key:     drx:agent:{agent_id}:circuit_opened_at
Type:    STRING
TTL:     None (deleted when circuit closes)
Value:   Unix timestamp (float)
```

**Example:**
```redis
SET drx:agent:planner:circuit_opened_at "1704067200.123"
```

### Half-Open Request Counter
Tracks requests allowed in half-open state.

```
Key:     drx:agent:{agent_id}:half_open_count
Type:    STRING (integer)
TTL:     recovery_timeout_seconds
Value:   Request count
```

**Example:**
```redis
INCR drx:agent:planner:half_open_count
EXPIRE drx:agent:planner:half_open_count 60
```

### Active State Service Usage Examples

**Recording an Invocation:**
```python
from src.services.active_state import get_active_state_service

async def record_agent_call():
    service = await get_active_state_service()
    await service.record_invocation(
        agent_id="planner",
        tokens=1500,
        latency_ms=250,
        success=True,
        error_type=None
    )
```

**Checking Agent Health:**
```python
async def check_agent_health():
    service = await get_active_state_service()
    health = await service.get_agent_health("planner")
    print(f"Status: {health['status']}")
    print(f"Circuit: {health['circuit_status']}")
    print(f"Error Rate: {health['metrics']['error_rate_5m']:.2%}")
```

**Circuit Breaker Pattern:**
```python
async def call_agent_with_circuit_breaker(agent_id: str):
    service = await get_active_state_service()

    # Check if request should be allowed
    if not await service.should_allow_request(agent_id):
        raise CircuitOpenError(f"Circuit open for {agent_id}")

    try:
        result = await call_agent(agent_id)
        await service.record_circuit_success(agent_id)
        return result
    except Exception as e:
        await service.record_circuit_failure(agent_id, error_type=type(e).__name__)
        raise
```

**Background Metrics Aggregation:**
```python
import asyncio

async def start_background_aggregation():
    service = await get_active_state_service()
    stop_event = asyncio.Event()

    # Run in background task
    task = asyncio.create_task(
        service.run_background_aggregation(
            interval_seconds=30,
            stop_event=stop_event
        )
    )

    # Later, to stop:
    stop_event.set()
    await task
```
