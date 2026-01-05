# Agentic Metadata Implementation Plan

## Overview

This plan implements the missing Agentic Metadata features (R10.x), Active State Tracking, and Deterministic Replay for DRX v1.

## Topological Dependency Graph

```
Wave 1 (Foundation - No Dependencies)
├── WP-M1: Agent Manifest JSON Schema
└── WP-M2: Active State Redis Service

Wave 2 (Depends on Wave 1)
├── WP-M3: Context Propagation System ──────┐
├── WP-M4: Capability-Based Routing ────────┤
└── WP-M5: Circuit Breaker Implementation ──┤
                                            │
Wave 3 (Depends on Wave 2)                  │
└── WP-M6: Metadata Firewall Middleware ────┘

Wave 4 (Final Integration)
├── WP-M7: Deterministic Replay System
└── WP-M8: Metadata-Aware Evaluation
```

---

## Work Packet Specifications

### WP-M1: Agent Manifest JSON Schema
**Dependencies**: None
**Owner**: Schema Agent
**Expertise**: JSON Schema, TypeScript types, Pydantic models

**Files to Create**:
```
v1/schemas/agent_manifest.json          # JSON Schema definition
v1/src/metadata/manifest.py             # Pydantic models for manifest
v1/src/metadata/__init__.py             # Module init
```

**Deliverables**:
1. JSON Schema with: id, version, capabilities, allowed_domains, blocked_domains, max_budget, rate_limits, health_thresholds, model_config
2. Pydantic model `AgentManifest` with validation
3. Factory function `load_manifest_from_registry(agent_id: str)`
4. Validation function `validate_manifest(manifest: dict) -> bool`

---

### WP-M2: Active State Redis Service
**Dependencies**: None
**Owner**: Redis/Infrastructure Agent
**Expertise**: Redis data structures, async Python, real-time metrics

**Files to Create**:
```
v1/src/services/active_state.py         # Active state service
v1/src/services/health_monitor.py       # Health monitoring
v1/schemas/redis_active_state.md        # Redis key documentation
```

**Deliverables**:
1. `ActiveStateService` class with methods:
   - `record_invocation(agent_id, tokens, latency_ms, success: bool)`
   - `get_agent_health(agent_id) -> AgentHealthStatus`
   - `get_token_burn_rate(agent_id, window_seconds=60) -> float`
   - `get_error_rate(agent_id, window_seconds=300) -> float`
   - `set_circuit_status(agent_id, status: CircuitStatus)`
   - `get_circuit_status(agent_id) -> CircuitStatus`

2. Redis key patterns:
   - `drx:agent:{agent_id}:invocations` - Sorted set (timestamp, invocation_data)
   - `drx:agent:{agent_id}:errors` - Sorted set (timestamp, error_type)
   - `drx:agent:{agent_id}:health` - Hash (status, last_check, failure_count)
   - `drx:agent:{agent_id}:circuit` - String (open/closed/half-open)
   - `drx:agent:{agent_id}:metrics` - Hash (tokens_1m, tokens_5m, latency_p50, latency_p99)

3. Background task for metrics aggregation (every 10 seconds)

---

### WP-M3: Context Propagation System
**Dependencies**: WP-M1
**Owner**: State Management Agent
**Expertise**: Distributed systems, context management, memory optimization

**Files to Create**:
```
v1/src/metadata/context.py              # Context propagation system
v1/src/metadata/context_store.py        # Context storage abstraction
```

**Deliverables**:
1. `ResearchContext` TypedDict:
   ```python
   class ResearchContext(TypedDict):
       context_id: str
       session_id: str
       summary: str                    # Compressed context summary
       key_entities: list[str]         # Extracted entities
       relevance_vector: list[float]   # Embedding for relevance check
       chunk_refs: list[str]           # References to full chunks in pgvector
       created_at: str
       ttl_seconds: int
   ```

2. `ContextPropagator` class:
   - `create_context(state: AgentState) -> ResearchContext`
   - `is_relevant(context: ResearchContext, task: SubTask) -> bool`
   - `fetch_relevant_chunks(context: ResearchContext, limit: int) -> list[str]`
   - `compress_context(full_context: str, max_tokens: int) -> str`

3. Context passing in agent invocations (modify `src/orchestrator/nodes.py`)

---

### WP-M4: Capability-Based Routing
**Dependencies**: WP-M2
**Owner**: Routing/Orchestration Agent
**Expertise**: Service discovery, load balancing, capability matching

**Files to Create**:
```
v1/src/metadata/routing.py              # Capability-based router
v1/src/metadata/agent_selector.py       # Agent selection logic
```

**Deliverables**:
1. `AgentCapabilityRouter` class:
   - `find_agent(requirements: AgentRequirements) -> str | None`
   - `get_available_agents(capability: str) -> list[AgentInfo]`
   - `score_agent(agent_id: str, requirements: AgentRequirements) -> float`

2. `AgentRequirements` TypedDict:
   ```python
   class AgentRequirements(TypedDict):
       required_capabilities: list[str]
       preferred_capabilities: list[str]
       compliance_level: Literal["standard", "high", "critical"]
       cost_tier: Literal["free", "standard", "premium"]
       max_latency_ms: int | None
       required_domains: list[str] | None
   ```

3. Integration with orchestrator nodes (replace hardcoded agent selection)

4. Fallback logic when primary agent unavailable

---

### WP-M5: Circuit Breaker Implementation
**Dependencies**: WP-M2
**Owner**: Resilience/Infrastructure Agent
**Expertise**: Circuit breaker patterns, failure detection, graceful degradation

**Files to Create**:
```
v1/src/metadata/circuit_breaker.py      # Circuit breaker implementation
v1/src/metadata/health_checker.py       # Health check logic
```

**Deliverables**:
1. `CircuitBreaker` class implementing standard circuit breaker pattern:
   - States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing)
   - `can_execute(agent_id: str) -> bool`
   - `record_success(agent_id: str)`
   - `record_failure(agent_id: str, error: Exception)`
   - `get_state(agent_id: str) -> CircuitState`

2. Configuration via agent manifest:
   ```python
   class CircuitBreakerConfig(TypedDict):
       failure_threshold: int          # Failures before opening (default: 5)
       success_threshold: int          # Successes to close (default: 3)
       timeout_seconds: int            # Time in open state (default: 30)
       half_open_max_calls: int        # Max calls in half-open (default: 3)
   ```

3. `HealthChecker` class:
   - `check_agent_health(agent_id: str) -> HealthStatus`
   - `is_healthy(agent_id: str) -> bool`
   - Configurable thresholds from manifest

4. Automatic rerouting when circuit opens

---

### WP-M6: Metadata Firewall Middleware
**Dependencies**: WP-M2, WP-M3
**Owner**: Security/Policy Agent
**Expertise**: Middleware patterns, policy enforcement, security

**Files to Create**:
```
v1/src/middleware/__init__.py           # Middleware module
v1/src/middleware/policy_firewall.py    # Policy enforcement middleware
v1/src/middleware/domain_validator.py   # Domain validation
```

**Deliverables**:
1. `PolicyFirewall` middleware class:
   - Intercepts all tool invocations
   - Checks agent manifest for `allowed_domains`, `blocked_domains`
   - Validates `max_budget` not exceeded
   - Enforces `rate_limits`

2. `DomainValidator`:
   - `is_domain_allowed(agent_id: str, domain: str) -> bool`
   - `extract_domain(url: str) -> str`

3. Policy violation recording:
   - Write to `policy_violations` table on violation
   - Emit `policy_violation` event to stream
   - Block or warn based on severity

4. Integration points:
   - Wrap tool execution in `src/tools/base.py`
   - Add middleware to `src/orchestrator/nodes.py`

---

### WP-M7: Deterministic Replay System
**Dependencies**: WP-M1, WP-M2, WP-M6 (all metadata infrastructure)
**Owner**: Replay/Debug Agent
**Expertise**: Event sourcing, replay systems, debugging tools

**Files to Create**:
```
v1/src/replay/__init__.py               # Replay module
v1/src/replay/recorder.py               # Event recording
v1/src/replay/player.py                 # Replay execution
v1/src/api/replay_routes.py             # Replay API endpoints
```

**Deliverables**:
1. `EventRecorder` class:
   - `record_event(session_id: str, event: ReplayEvent)`
   - `get_events(session_id: str, from_checkpoint: str | None) -> list[ReplayEvent]`

2. `ReplayEvent` TypedDict:
   ```python
   class ReplayEvent(TypedDict):
       event_id: str
       session_id: str
       checkpoint_id: str
       event_type: str
       node_name: str
       inputs: dict
       outputs: dict
       tool_calls: list[ToolCallRecord]
       llm_calls: list[LLMCallRecord]
       timestamp: str
       deterministic_seed: int | None
   ```

3. `ReplayPlayer` class:
   - `replay_from_checkpoint(session_id: str, checkpoint_id: str) -> AsyncGenerator`
   - `replay_with_modifications(session_id: str, modifications: dict) -> AsyncGenerator`
   - `compare_runs(original_session: str, replay_session: str) -> DiffReport`

4. API endpoints:
   - `POST /api/v1/interactions/{id}/replay` - Start replay
   - `GET /api/v1/interactions/{id}/events` - Get recorded events
   - `POST /api/v1/interactions/{id}/compare` - Compare original vs replay

---

### WP-M8: Metadata-Aware Evaluation
**Dependencies**: All above
**Owner**: Evaluation Agent
**Expertise**: Testing frameworks, metrics, compliance validation

**Files to Create**:
```
v1/ci/evaluation/metadata_assertions.py  # Metadata assertion helpers
v1/ci/evaluation/test_metadata_compliance.py  # Compliance tests
v1/ci/evaluation/test_circuit_breaker.py      # Circuit breaker tests
```

**Deliverables**:
1. Metadata assertion helpers:
   ```python
   def assert_budget_compliance(trace: Trace, manifest: AgentManifest) -> None
   def assert_domain_compliance(trace: Trace, manifest: AgentManifest) -> None
   def assert_rate_limit_compliance(trace: Trace, manifest: AgentManifest) -> None
   def assert_capability_match(agent_id: str, task: SubTask) -> None
   ```

2. Integration with existing DeepEval tests

3. Test scenarios:
   - Budget exceeded scenario
   - Blocked domain access attempt
   - Circuit breaker activation
   - Capability mismatch routing

4. CI/CD integration (update `.github/workflows/evaluation.yaml`)

---

## Execution Strategy

### Wave 1 (Parallel - No Dependencies)
- WP-M1 and WP-M2 can execute simultaneously
- Estimated: 15-20 minutes each

### Wave 2 (Parallel - After Wave 1)
- WP-M3, WP-M4, WP-M5 can execute in parallel
- WP-M3 depends on WP-M1 (manifest types)
- WP-M4 depends on WP-M2 (active state for health)
- WP-M5 depends on WP-M2 (active state for metrics)
- Estimated: 20-25 minutes each

### Wave 3 (After Wave 2)
- WP-M6 depends on WP-M2 and WP-M3
- Estimated: 20-25 minutes

### Wave 4 (Final - After Wave 3)
- WP-M7 and WP-M8 can execute in parallel
- Both depend on full metadata infrastructure
- Estimated: 25-30 minutes each

---

## Output Chunking Strategy

Due to 32,000 token output limits, each agent must:

1. **Step 1**: Create directory structure and `__init__.py` files
2. **Step 2**: Implement core data types (TypedDicts, Enums)
3. **Step 3**: Implement main class/service (Part 1 - initialization, core methods)
4. **Step 4**: Implement main class/service (Part 2 - remaining methods)
5. **Step 5**: Create integration points and update existing files
6. **Step 6**: Write tests and documentation

Each step outputs ~4000-6000 tokens, staying well under limits.

---

## Git Strategy

After each work packet:
1. Stage new files: `git add src/metadata/ src/middleware/ src/replay/`
2. Verify syntax: `python -m py_compile <files>`
3. Final commit after all waves complete
