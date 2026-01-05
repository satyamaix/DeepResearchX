# DRX Low-Level Design (LLD) Document

## Table of Contents
1. [Class Diagrams](#class-diagrams)
2. [Database Design](#database-design)
3. [Sequence Diagrams](#sequence-diagrams)
4. [API Specifications](#api-specifications)
5. [Component Interfaces](#component-interfaces)
6. [Error Handling](#error-handling)
7. [Configuration Schema](#configuration-schema)

---

## Class Diagrams

### Agent Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BaseAgent (ABC)                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Attributes:                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  - agent_id: str                                                                 │
│  - agent_type: AgentTypeEnum                                                     │
│  - system_prompt: str                                                            │
│  - llm_client: OpenRouterClient                                                  │
│  - tools: list[BaseTool]                                                         │
│  - manifest: AgentManifest | None                                                │
│  - tracer: Tracer                                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Methods:                                                                        │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  + __call__(state: AgentState) -> AgentState                                     │
│  + _process(state: AgentState) -> AgentResponse  [abstract]                      │
│  + _post_process(state: AgentState, response: AgentResponse) -> AgentState       │
│  + _validate_input(state: AgentState) -> bool                                    │
│  + _emit_event(event_type: str, data: dict) -> None                              │
│  + _record_metrics(metrics: AgentMetrics) -> None                                │
│  + _check_budget(estimated_tokens: int) -> bool                                  │
│  + _get_manifest() -> AgentManifest | None                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ extends
        ┌───────────────┬───────────────┼───────────────┬───────────────┬───────────────┐
        │               │               │               │               │               │
        ▼               ▼               ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│PlannerAgent  │ │SearcherAgent │ │ ReaderAgent  │ │SynthesizerAgent│ │ CriticAgent │ │ReporterAgent │
├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────┤
│              │ │              │ │              │ │              │ │              │ │              │
│ Methods:     │ │ Methods:     │ │ Methods:     │ │ Methods:     │ │ Methods:     │ │ Methods:     │
│ _process()   │ │ _process()   │ │ _process()   │ │ _process()   │ │ _process()   │ │ _process()   │
│ _decompose() │ │ _expand_     │ │ _fetch_      │ │ _aggregate() │ │ _evaluate()  │ │ _generate()  │
│ _prioritize()│ │  query()     │ │  content()   │ │ _resolve_    │ │ _find_gaps() │ │ _format()    │
│ _build_dag() │ │ _search()    │ │ _extract()   │ │  conflicts() │ │ _score()     │ │ _cite()      │
│              │ │ _dedupe()    │ │ _parse()     │ │ _build_      │ │ _verify()    │ │              │
│              │ │              │ │              │ │  argument()  │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### State Classes

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            AgentState (TypedDict)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  messages: Annotated[list[AnyMessage], add_messages]                             │
│  session_id: str                                                                 │
│  user_query: str                                                                 │
│  steerability: SteerabilityConfig                                                │
│  plan: ResearchPlan                                                              │
│  findings: list[Finding]                                                         │
│  citations: list[CitationRecord]                                                 │
│  synthesis: str                                                                  │
│  gaps: list[str]                                                                 │
│  policy_violations: list[PolicyViolation]                                        │
│  final_report: str | None                                                        │
│  iteration_count: int                                                            │
│  token_budget: int                                                               │
│  tokens_used: int                                                                │
│  metrics: AgentMetrics                                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
        │
        │ contains
        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ResearchPlan (TypedDict)                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  dag_nodes: list[SubTask]                                                        │
│  current_iteration: int                                                          │
│  max_iterations: int                                                             │
│  coverage_score: float                                                           │
│  execution_order: list[str]                                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
        │
        │ contains
        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             SubTask (TypedDict)                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  id: str                                                                         │
│  description: str                                                                │
│  agent_type: str                                                                 │
│  dependencies: list[str]                                                         │
│  status: Literal["pending", "running", "completed", "failed"]                    │
│  inputs: dict[str, Any]                                                          │
│  outputs: dict[str, Any] | None                                                  │
│  quality_score: float | None                                                     │
│  retry_count: int                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Metadata Infrastructure Classes

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          AgentManifest (Pydantic)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  agent_id: str                                                                   │
│  version: str                                                                    │
│  agent_type: AgentTypeEnum                                                       │
│  capabilities: list[str]                                                         │
│  allowed_domains: list[str]                                                      │
│  blocked_domains: list[str]                                                      │
│  max_budget_usd: float                                                           │
│  rate_limits: RateLimits                                                         │
│  circuit_breaker: CircuitBreakerConfig                                           │
│  model_config: ModelConfig                                                       │
│  metadata: dict[str, Any]                                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Methods:                                                                        │
│  + is_domain_allowed(url: str) -> bool                                           │
│  + check_budget(current_spend: float) -> bool                                    │
│  + to_dict() -> dict[str, Any]                                                   │
│  + from_json(path: str) -> AgentManifest [classmethod]                           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            CircuitBreaker                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Attributes:                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  - agent_id: str                                                                 │
│  - state: CircuitState (CLOSED | OPEN | HALF_OPEN)                               │
│  - failure_count: int                                                            │
│  - success_count: int                                                            │
│  - last_failure_time: datetime | None                                            │
│  - config: CircuitBreakerConfig                                                  │
│  - redis: RedisClient                                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Methods:                                                                        │
│  + call(func: Callable, *args, **kwargs) -> T                                    │
│  + record_success() -> None                                                      │
│  + record_failure(error: Exception) -> None                                      │
│  + is_available() -> bool                                                        │
│  + get_state() -> CircuitState                                                   │
│  + reset() -> None                                                               │
│  - _should_attempt_reset() -> bool                                               │
│  - _transition_to(state: CircuitState) -> None                                   │
│  - _persist_state() -> None                                                      │
│  - _load_state() -> None                                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                             PolicyFirewall                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Attributes:                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  - manifest: AgentManifest                                                       │
│  - domain_validator: DomainValidator                                             │
│  - redis: RedisClient                                                            │
│  - db_pool: AsyncConnectionPool                                                  │
│  - config: dict[str, Any]                                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Methods:                                                                        │
│  + check_tool_invocation(tool_name: str, args: dict) -> PolicyResult             │
│  + check_domain(url: str) -> PolicyResult                                        │
│  + check_budget(estimated_cost: float) -> PolicyResult                           │
│  + check_rate_limit(tokens: int) -> PolicyResult                                 │
│  + log_violation(violation: PolicyViolation) -> None                             │
│  - _extract_urls(args: dict) -> list[str]                                        │
│  - _get_current_spend() -> float                                                 │
│  - _get_token_count(window: str) -> int                                          │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          CapabilityRouter                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Attributes:                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  - registry: AgentRegistry                                                       │
│  - active_state: ActiveStateService                                              │
│  - weights: ScoringWeights                                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Methods:                                                                        │
│  + select_agent(requirements: TaskRequirements) -> AgentManifest                 │
│  + get_fallback_agents(agent_id: str) -> list[AgentManifest]                     │
│  + score_agent(manifest: AgentManifest, requirements: TaskRequirements) -> float │
│  - _calculate_capability_match(capabilities: list, required: list) -> float      │
│  - _calculate_health_score(agent_id: str) -> float                               │
│  - _calculate_load_score(agent_id: str) -> float                                 │
│  - _calculate_cost_penalty(manifest: AgentManifest, tier: str) -> float          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Service Classes

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           OpenRouterClient                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Attributes:                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  - api_key: str                                                                  │
│  - base_url: str = "https://openrouter.ai/api/v1"                                │
│  - http_client: httpx.AsyncClient                                                │
│  - default_model: str                                                            │
│  - tracer: Tracer                                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Methods:                                                                        │
│  + generate(prompt: str, model: str | None, **kwargs) -> LLMResponse             │
│  + generate_stream(prompt: str, model: str | None) -> AsyncIterator[str]         │
│  + count_tokens(text: str) -> int                                                │
│  + get_model_info(model: str) -> ModelInfo                                       │
│  - _prepare_request(prompt: str, **kwargs) -> dict                               │
│  - _handle_response(response: httpx.Response) -> LLMResponse                     │
│  - _handle_error(error: Exception) -> None                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ActiveStateService                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Attributes:                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  - redis: RedisClient                                                            │
│  - key_prefix: str = "drx"                                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Methods:                                                                        │
│  + get_agent_health(agent_id: str) -> AgentHealth                                │
│  + set_agent_health(agent_id: str, health: AgentHealth) -> None                  │
│  + record_invocation(agent_id: str, metrics: InvocationMetrics) -> None          │
│  + get_metrics(agent_id: str, window: str) -> AgentMetrics                       │
│  + get_circuit_state(agent_id: str) -> CircuitState                              │
│  + set_circuit_state(agent_id: str, state: CircuitState) -> None                 │
│  + get_rate_limit_usage(agent_id: str) -> RateLimitUsage                         │
│  + increment_rate_limit(agent_id: str, tokens: int) -> None                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            ReplayRecorder                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Attributes:                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  - session_id: str                                                               │
│  - db_pool: AsyncConnectionPool                                                  │
│  - events: list[ReplayEvent]                                                     │
│  - recording: bool                                                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Methods:                                                                        │
│  + start_recording() -> None                                                     │
│  + stop_recording() -> None                                                      │
│  + record_event(event_type: str, data: dict, deterministic: bool) -> None        │
│  + record_llm_call(request: dict, response: dict) -> None                        │
│  + record_tool_call(tool: str, args: dict, result: Any) -> None                  │
│  + get_events(from_seq: int | None) -> list[ReplayEvent]                         │
│  + persist() -> None                                                             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                             ReplayPlayer                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Attributes:                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  - session_id: str                                                               │
│  - events: list[ReplayEvent]                                                     │
│  - current_index: int                                                            │
│  - modifications: ReplayModifications                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Methods:                                                                        │
│  + load_events(from_checkpoint: str | None) -> None                              │
│  + replay() -> AsyncIterator[ReplayResult]                                       │
│  + get_next_response(event_type: str) -> Any                                     │
│  + compare_with_original(new_result: Any) -> ComparisonResult                    │
│  + set_modifications(mods: ReplayModifications) -> None                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Database Design

### Entity Relationship Diagram

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  Entity Relationship Diagram                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────┐         ┌─────────────────────┐         ┌─────────────────────┐
    │   research_sessions │         │   research_steps    │         │  agent_invocations  │
    ├─────────────────────┤         ├─────────────────────┤         ├─────────────────────┤
    │ PK id: UUID         │◀───────┐│ PK id: UUID         │◀───────┐│ PK id: UUID         │
    │    user_id: UUID    │   1:N ││ FK session_id: UUID  │   1:N ││ FK step_id: UUID    │
    │    query: TEXT      │        ││ FK parent_step_id   │        ││    agent_type: ENUM │
    │    steerability: JSON│       ││    step_type: VARCHAR│       ││    input_tokens: INT│
    │    status: VARCHAR  │        ││    agent_type: ENUM │        ││    output_tokens:INT│
    │    config: JSON     │        ││    status: VARCHAR  │        ││    latency_ms: INT  │
    │    created_at: TS   │        ││    inputs: JSON     │        ││    cost_usd: DECIMAL│
    │    updated_at: TS   │        ││    outputs: JSON    │        ││    error: TEXT      │
    │    completed_at: TS │        ││    metadata: JSON   │        ││    created_at: TS   │
    └─────────────────────┘        ││    error: TEXT      │        │└─────────────────────┘
              │                     ││    created_at: TS   │        │
              │                     ││    completed_at: TS │        │
              │                     │└─────────────────────┘        │
              │                     │          │                     │
              │                     │          │                     │
              │                     │          │ 1:N                 │
              │                     │          ▼                     │
              │                     │┌─────────────────────┐        │
              │                     ││  tool_invocations   │        │
              │                     │├─────────────────────┤        │
              │                     ││ PK id: UUID         │        │
              │                     ││ FK step_id: UUID    │────────┘
              │                     ││    tool_name: VARCHAR│
              │                     ││    tool_args: JSON  │
              │                     ││    tool_result: JSON│
              │                     ││    latency_ms: INT  │
              │                     ││    tokens_used: INT │
              │                     ││    success: BOOL    │
              │                     ││    created_at: TS   │
              │                     │└─────────────────────┘
              │                     │
              │ 1:N                 │
              ▼                     │
    ┌─────────────────────┐        │
    │  policy_violations  │        │
    ├─────────────────────┤        │
    │ PK id: UUID         │        │
    │ FK session_id: UUID │        │
    │ FK step_id: UUID    │────────┘
    │    agent_id: VARCHAR│
    │    violation_type   │
    │    severity: VARCHAR│
    │    details: JSON    │
    │    blocked: BOOL    │
    │    created_at: TS   │
    └─────────────────────┘

    ┌─────────────────────┐         ┌─────────────────────┐         ┌─────────────────────┐
    │   agent_registry    │         │   document_chunks   │         │   replay_events     │
    ├─────────────────────┤         ├─────────────────────┤         ├─────────────────────┤
    │ PK id: VARCHAR      │         │ PK id: UUID         │         │ PK id: UUID         │
    │    version: VARCHAR │         │ FK session_id: UUID │         │ FK session_id: UUID │
    │    agent_type: ENUM │         │    source_url: TEXT │         │    sequence_num: INT│
    │    capabilities: []  │         │    content: TEXT    │         │    event_type: VARCHAR
    │    allowed_domains[]│         │    embedding: vector│         │    event_data: JSON │
    │    blocked_domains[]│         │    metadata: JSON   │         │    deterministic: BOOL
    │    max_budget: DEC  │         │    chunk_index: INT │         │    timestamp: TS    │
    │    rate_limits: JSON│         │    created_at: TS   │         │    created_at: TS   │
    │    circuit_config   │         └─────────────────────┘         └─────────────────────┘
    │    model_config:JSON│                   │
    │    is_active: BOOL  │                   │ vector index
    │    created_at: TS   │                   ▼
    │    updated_at: TS   │         ┌─────────────────────┐
    └─────────────────────┘         │ idx_chunks_embedding│
                                    │ (ivfflat cosine)    │
                                    └─────────────────────┘
```

### Table Definitions Detail

```sql
-- Enum types
CREATE TYPE session_status AS ENUM (
    'queued', 'running', 'paused', 'completed', 'failed', 'cancelled'
);

CREATE TYPE agent_type AS ENUM (
    'orchestrator', 'planner', 'searcher', 'reader',
    'reasoner', 'writer', 'critic', 'synthesizer', 'reporter'
);

CREATE TYPE step_status AS ENUM (
    'pending', 'running', 'completed', 'failed', 'skipped'
);

CREATE TYPE violation_type AS ENUM (
    'domain_blocked', 'budget_exceeded', 'rate_limit_exceeded',
    'content_policy', 'authentication', 'other'
);

CREATE TYPE violation_severity AS ENUM (
    'info', 'warning', 'error', 'critical'
);

-- Core indexes
CREATE INDEX idx_sessions_user ON research_sessions(user_id);
CREATE INDEX idx_sessions_status ON research_sessions(status);
CREATE INDEX idx_sessions_created ON research_sessions(created_at DESC);

CREATE INDEX idx_steps_session ON research_steps(session_id);
CREATE INDEX idx_steps_status ON research_steps(status);
CREATE INDEX idx_steps_parent ON research_steps(parent_step_id);

CREATE INDEX idx_violations_session ON policy_violations(session_id);
CREATE INDEX idx_violations_type ON policy_violations(violation_type);

CREATE INDEX idx_replay_session_seq ON replay_events(session_id, sequence_num);
```

### LangGraph Checkpoint Tables

```sql
-- Tables created by langgraph-checkpoint-postgres
-- Reference: AsyncPostgresSaver.setup()

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value JSONB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
```

---

## Sequence Diagrams

### Research Session Execution

```
┌──────┐      ┌─────────┐      ┌────────────┐      ┌───────────┐      ┌────────┐      ┌───────┐
│Client│      │  API    │      │Orchestrator│      │  Agents   │      │Services│      │  DB   │
└──┬───┘      └────┬────┘      └─────┬──────┘      └─────┬─────┘      └───┬────┘      └───┬───┘
   │               │                 │                   │                │              │
   │ POST /interactions              │                   │                │              │
   │──────────────▶│                 │                   │                │              │
   │               │                 │                   │                │              │
   │               │ validate request│                   │                │              │
   │               │────────────────▶│                   │                │              │
   │               │                 │                   │                │              │
   │               │                 │ INSERT session    │                │              │
   │               │                 │────────────────────────────────────┼─────────────▶│
   │               │                 │                   │                │              │
   │               │                 │                   │                │    session_id│
   │               │                 │◀───────────────────────────────────┼──────────────│
   │               │                 │                   │                │              │
   │               │ queue task      │                   │                │              │
   │               │────────────────▶│                   │                │              │
   │               │                 │                   │                │              │
   │  202 Accepted │                 │                   │                │              │
   │◀──────────────│                 │                   │                │              │
   │               │                 │                   │                │              │
   │               │                 │══════════════════════════════════════════════════│
   │               │                 │         ASYNC WORKER EXECUTION                   │
   │               │                 │══════════════════════════════════════════════════│
   │               │                 │                   │                │              │
   │               │                 │ load state        │                │              │
   │               │                 │────────────────────────────────────┼─────────────▶│
   │               │                 │                   │                │              │
   │               │                 │ run planner       │                │              │
   │               │                 │──────────────────▶│                │              │
   │               │                 │                   │                │              │
   │               │                 │                   │ LLM call       │              │
   │               │                 │                   │───────────────▶│              │
   │               │                 │                   │                │              │
   │               │                 │                   │◀───────────────│              │
   │               │                 │                   │                │              │
   │               │                 │◀──────────────────│                │              │
   │               │                 │                   │                │              │
   │               │                 │ checkpoint        │                │              │
   │               │                 │────────────────────────────────────┼─────────────▶│
   │               │                 │                   │                │              │
   │               │                 │ run searcher      │                │              │
   │               │                 │──────────────────▶│                │              │
   │               │                 │                   │                │              │
   │               │                 │                   │ Tavily search  │              │
   │               │                 │                   │───────────────▶│              │
   │               │                 │                   │                │              │
   │               │                 │                   │◀───────────────│              │
   │               │                 │                   │                │              │
   │               │                 │◀──────────────────│                │              │
   │               │                 │                   │                │              │
   │               │                 │         ... (reader, synthesizer, critic) ...    │
   │               │                 │                   │                │              │
   │               │                 │ run reporter      │                │              │
   │               │                 │──────────────────▶│                │              │
   │               │                 │                   │                │              │
   │               │                 │◀──────────────────│                │              │
   │               │                 │                   │                │              │
   │               │                 │ UPDATE session    │                │              │
   │               │                 │────────────────────────────────────┼─────────────▶│
   │               │                 │                   │                │              │
   └───────────────┴─────────────────┴───────────────────┴────────────────┴──────────────┘
```

### SSE Streaming Flow

```
┌──────┐      ┌─────────┐      ┌────────┐      ┌───────────┐
│Client│      │  API    │      │ Redis  │      │Orchestrator│
└──┬───┘      └────┬────┘      └───┬────┘      └─────┬─────┘
   │               │               │                 │
   │ GET /stream   │               │                 │
   │──────────────▶│               │                 │
   │               │               │                 │
   │               │ SUBSCRIBE     │                 │
   │               │──────────────▶│                 │
   │               │               │                 │
   │  SSE: connected               │                 │
   │◀──────────────│               │                 │
   │               │               │                 │
   │               │               │ PUBLISH event   │
   │               │               │◀────────────────│
   │               │               │                 │
   │               │ event received│                 │
   │               │◀──────────────│                 │
   │               │               │                 │
   │  SSE: thought_summary         │                 │
   │◀──────────────│               │                 │
   │               │               │                 │
   │               │               │ PUBLISH event   │
   │               │               │◀────────────────│
   │               │               │                 │
   │  SSE: content.delta           │                 │
   │◀──────────────│               │                 │
   │               │               │                 │
   │               │               │ PUBLISH event   │
   │               │               │◀────────────────│
   │               │               │                 │
   │  SSE: checkpoint              │                 │
   │◀──────────────│               │                 │
   │               │               │                 │
   │               │               │ PUBLISH complete│
   │               │               │◀────────────────│
   │               │               │                 │
   │  SSE: interaction.complete    │                 │
   │◀──────────────│               │                 │
   │               │               │                 │
   │  Connection closed            │                 │
   │◀──────────────│               │                 │
   └───────────────┴───────────────┴─────────────────┘
```

### Circuit Breaker Flow

```
┌───────┐      ┌───────────────┐      ┌────────┐      ┌───────────┐
│ Agent │      │CircuitBreaker │      │ Redis  │      │ LLM/Tool  │
└───┬───┘      └───────┬───────┘      └───┬────┘      └─────┬─────┘
    │                  │                  │                 │
    │ call(fn, args)   │                  │                 │
    │─────────────────▶│                  │                 │
    │                  │                  │                 │
    │                  │ GET state        │                 │
    │                  │─────────────────▶│                 │
    │                  │                  │                 │
    │                  │◀─────────────────│                 │
    │                  │  state=CLOSED    │                 │
    │                  │                  │                 │
    │                  │ execute fn       │                 │
    │                  │─────────────────────────────────▶ │
    │                  │                  │                 │
    │                  │◀─────────────────────────────────│
    │                  │  success         │                 │
    │                  │                  │                 │
    │                  │ INCR success     │                 │
    │                  │─────────────────▶│                 │
    │                  │                  │                 │
    │◀─────────────────│                  │                 │
    │  result          │                  │                 │
    │                  │                  │                 │
    │═══════════════════════════════════════════════════════│
    │                  FAILURE SCENARIO                     │
    │═══════════════════════════════════════════════════════│
    │                  │                  │                 │
    │ call(fn, args)   │                  │                 │
    │─────────────────▶│                  │                 │
    │                  │                  │                 │
    │                  │ execute fn       │                 │
    │                  │─────────────────────────────────▶ │
    │                  │                  │                 │
    │                  │◀─────────────────────────────────│
    │                  │  ERROR           │                 │
    │                  │                  │                 │
    │                  │ INCR failure     │                 │
    │                  │─────────────────▶│                 │
    │                  │                  │                 │
    │                  │                  │                 │
    │                  │ if failures >= threshold:         │
    │                  │ SET state=OPEN   │                 │
    │                  │─────────────────▶│                 │
    │                  │                  │                 │
    │◀─────────────────│                  │                 │
    │  raise error     │                  │                 │
    │                  │                  │                 │
    │═══════════════════════════════════════════════════════│
    │                  OPEN STATE CALL                      │
    │═══════════════════════════════════════════════════════│
    │                  │                  │                 │
    │ call(fn, args)   │                  │                 │
    │─────────────────▶│                  │                 │
    │                  │                  │                 │
    │                  │ GET state        │                 │
    │                  │─────────────────▶│                 │
    │                  │                  │                 │
    │                  │◀─────────────────│                 │
    │                  │  state=OPEN      │                 │
    │                  │                  │                 │
    │◀─────────────────│                  │                 │
    │  CircuitOpenError│ (no external call made)           │
    └──────────────────┴──────────────────┴─────────────────┘
```

### Policy Firewall Check Flow

```
┌───────┐      ┌──────────────┐      ┌──────────────┐      ┌────────┐      ┌────────┐
│ Agent │      │PolicyFirewall│      │DomainValidator│     │ Redis  │      │   DB   │
└───┬───┘      └──────┬───────┘      └──────┬───────┘      └───┬────┘      └───┬────┘
    │                 │                     │                  │               │
    │ check_tool_invocation(tool, args)    │                  │               │
    │────────────────▶│                     │                  │               │
    │                 │                     │                  │               │
    │                 │ extract_urls(args)  │                  │               │
    │                 │────────────────────▶│                  │               │
    │                 │                     │                  │               │
    │                 │ urls               │                  │               │
    │                 │◀────────────────────│                  │               │
    │                 │                     │                  │               │
    │                 │ for each url:       │                  │               │
    │                 │   is_allowed(url)   │                  │               │
    │                 │────────────────────▶│                  │               │
    │                 │                     │                  │               │
    │                 │◀────────────────────│                  │               │
    │                 │                     │                  │               │
    │                 │ GET current_spend   │                  │               │
    │                 │────────────────────────────────────────┼──────────────▶│
    │                 │                     │                  │               │
    │                 │◀───────────────────────────────────────┼───────────────│
    │                 │                     │                  │               │
    │                 │ check budget        │                  │               │
    │                 │ (spend + estimated <= max)            │               │
    │                 │                     │                  │               │
    │                 │ GET token_count     │                  │               │
    │                 │────────────────────────────────────▶  │               │
    │                 │                     │                  │               │
    │                 │◀───────────────────────────────────── │               │
    │                 │                     │                  │               │
    │                 │ check rate limit    │                  │               │
    │                 │ (count <= limit)    │                  │               │
    │                 │                     │                  │               │
    │ PolicyResult    │                     │                  │               │
    │◀────────────────│                     │                  │               │
    │                 │                     │                  │               │
    │                 │ if violation:       │                  │               │
    │                 │   log_violation()   │                  │               │
    │                 │────────────────────────────────────────┼──────────────▶│
    │                 │                     │                  │               │
    └─────────────────┴─────────────────────┴──────────────────┴───────────────┘
```

---

## API Specifications

### Endpoint Details

#### POST /api/v1/interactions

```yaml
Request:
  Content-Type: application/json
  Body:
    query: string (required)
      description: The research question or task
      min_length: 10
      max_length: 10000
    steerability:
      tone: enum [academic, executive, technical, casual]
        default: academic
      format: enum [markdown, html, json, plain]
        default: markdown
      max_sources: integer
        minimum: 1
        maximum: 100
        default: 20
      language: string
        default: "en"
    config:
      max_iterations: integer
        minimum: 1
        maximum: 10
        default: 5
      thinking_summaries: boolean
        default: true
      token_budget: integer
        minimum: 1000
        maximum: 1000000
        default: 100000

Response:
  201 Created:
    id: string (UUID)
    status: enum [queued]
    created_at: string (ISO 8601)
    estimated_duration: string (optional)

  400 Bad Request:
    error: string
    detail: string
    validation_errors: array

  429 Too Many Requests:
    error: string
    retry_after: integer (seconds)

Example Request:
  POST /api/v1/interactions
  {
    "query": "What are the latest advances in quantum computing?",
    "steerability": {
      "tone": "academic",
      "format": "markdown",
      "max_sources": 15
    },
    "config": {
      "max_iterations": 3,
      "thinking_summaries": true
    }
  }

Example Response:
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "queued",
    "created_at": "2024-12-15T10:30:00Z"
  }
```

#### GET /api/v1/interactions/{id}/stream

```yaml
Request:
  Path Parameters:
    id: string (UUID) - Interaction ID
  Headers:
    Accept: text/event-stream
    Last-Event-ID: string (optional) - For resuming from checkpoint

Response:
  Content-Type: text/event-stream

Events:
  interaction.start:
    data: { "id": string, "status": "running" }

  thought_summary:
    data: { "agent": string, "text": string }

  tool.use:
    data: { "tool": string, "args": object }

  tool.result:
    data: { "tool": string, "result": object }

  content.delta:
    data: { "text": string }

  checkpoint:
    data: { "checkpoint_id": string, "node": string }

  error:
    data: { "error": string, "recoverable": boolean }

  interaction.complete:
    data: { "id": string, "status": "completed" }

Example SSE Stream:
  event: interaction.start
  id: evt_001
  data: {"id": "550e8400...", "status": "running"}

  event: thought_summary
  id: evt_002
  data: {"agent": "planner", "text": "Decomposing into 3 sub-questions..."}

  event: tool.use
  id: evt_003
  data: {"tool": "tavily_search", "args": {"query": "quantum computing 2024"}}

  event: checkpoint
  id: chk_001
  data: {"checkpoint_id": "chk_abc123", "node": "planner"}

  event: content.delta
  id: evt_010
  data: {"text": "## Executive Summary\n\n"}

  event: interaction.complete
  id: evt_final
  data: {"id": "550e8400...", "status": "completed"}
```

#### POST /api/v1/interactions/{id}/replay

```yaml
Request:
  Path Parameters:
    id: string (UUID) - Original interaction ID
  Body:
    from_checkpoint: string (optional)
      description: Checkpoint ID to replay from
    modifications:
      tools: object
        description: Override tool configurations
      agents: object
        description: Override agent configurations
      model: string
        description: Override model selection

Response:
  201 Created:
    replay_id: string (UUID)
    original_id: string (UUID)
    from_checkpoint: string | null
    status: enum [queued]
    created_at: string (ISO 8601)

Example Request:
  POST /api/v1/interactions/550e8400.../replay
  {
    "from_checkpoint": "chk_abc123",
    "modifications": {
      "tools": {
        "tavily_search": {"max_results": 5}
      }
    }
  }

Example Response:
  {
    "replay_id": "660e8400-e29b-41d4-a716-446655440001",
    "original_id": "550e8400-e29b-41d4-a716-446655440000",
    "from_checkpoint": "chk_abc123",
    "status": "queued",
    "created_at": "2024-12-15T11:00:00Z"
  }
```

#### GET /api/v1/interactions/{id}/replay/events

```yaml
Request:
  Path Parameters:
    id: string (UUID) - Original interaction ID
  Query Parameters:
    from_seq: integer (optional) - Start from sequence number
    limit: integer (optional, default: 100)

Response:
  200 OK:
    events: array
      - id: string (UUID)
        sequence_num: integer
        event_type: string
        event_data: object
        deterministic: boolean
        timestamp: string (ISO 8601)
    has_more: boolean
    next_seq: integer | null

Example Response:
  {
    "events": [
      {
        "id": "evt_001",
        "sequence_num": 1,
        "event_type": "llm_call",
        "event_data": {
          "model": "google/gemini-2.0-flash-exp",
          "input_tokens": 1500,
          "output_tokens": 800,
          "request": {...},
          "response": {...}
        },
        "deterministic": true,
        "timestamp": "2024-12-15T10:30:05Z"
      }
    ],
    "has_more": true,
    "next_seq": 101
  }
```

#### GET /api/v1/health

```yaml
Response:
  200 OK:
    status: enum [healthy, degraded, unhealthy]
    version: string
    components:
      database:
        status: enum [healthy, unhealthy]
        latency_ms: integer
      redis:
        status: enum [healthy, unhealthy]
        latency_ms: integer
      phoenix:
        status: enum [healthy, unhealthy]
        latency_ms: integer
      openrouter:
        status: enum [healthy, unhealthy]
        latency_ms: integer
    timestamp: string (ISO 8601)

Example Response:
  {
    "status": "healthy",
    "version": "1.0.0",
    "components": {
      "database": {"status": "healthy", "latency_ms": 5},
      "redis": {"status": "healthy", "latency_ms": 2},
      "phoenix": {"status": "healthy", "latency_ms": 15},
      "openrouter": {"status": "healthy", "latency_ms": 120}
    },
    "timestamp": "2024-12-15T10:35:00Z"
  }
```

---

## Component Interfaces

### Agent Interface

```python
from abc import ABC, abstractmethod
from typing import TypedDict, Any

class AgentInterface(ABC):
    """Interface that all agents must implement."""

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for this agent instance."""
        ...

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Type of agent (planner, searcher, etc.)."""
        ...

    @abstractmethod
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Execute the agent on the given state.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        ...

    @abstractmethod
    async def validate(self, state: AgentState) -> bool:
        """
        Validate that the agent can process the given state.

        Args:
            state: Current workflow state

        Returns:
            True if state is valid for this agent
        """
        ...
```

### Tool Interface

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from pydantic import BaseModel

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)

class ToolInterface(ABC, Generic[TInput, TOutput]):
    """Interface that all tools must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for identification."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description for LLM tool selection."""
        ...

    @abstractmethod
    async def execute(self, input: TInput) -> TOutput:
        """
        Execute the tool with given input.

        Args:
            input: Validated input parameters

        Returns:
            Tool execution result
        """
        ...

    @abstractmethod
    def get_schema(self) -> dict:
        """Return JSON schema for tool parameters."""
        ...
```

### Service Interfaces

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator

class LLMClientInterface(ABC):
    """Interface for LLM client implementations."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """Generate completion for prompt."""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming completion."""
        ...

class CacheInterface(ABC):
    """Interface for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        ...

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None
    ) -> None:
        """Set value in cache with optional TTL."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        ...

class CheckpointerInterface(ABC):
    """Interface for state checkpointing."""

    @abstractmethod
    async def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata
    ) -> RunnableConfig:
        """Save checkpoint."""
        ...

    @abstractmethod
    async def get(
        self,
        config: RunnableConfig
    ) -> CheckpointTuple | None:
        """Load checkpoint."""
        ...

    @abstractmethod
    async def list(
        self,
        config: RunnableConfig | None = None,
        *,
        filter: dict | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints."""
        ...
```

---

## Error Handling

### Error Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DRXError (Base)                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  message: str                                                                    │
│  code: str                                                                       │
│  recoverable: bool                                                               │
│  details: dict[str, Any]                                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        │               │               │               │               │
        ▼               ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ValidationError│ │ AgentError   │ │ PolicyError  │ │ ServiceError │ │ StorageError │
├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────┤
│code: VAL_*   │ │code: AGT_*   │ │code: POL_*   │ │code: SVC_*   │ │code: STR_*   │
│recoverable:  │ │recoverable:  │ │recoverable:  │ │recoverable:  │ │recoverable:  │
│  false       │ │  varies      │ │  true        │ │  varies      │ │  varies      │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │               │               │               │
        │               │               │               │
        ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│InputValidation│ │AgentTimeout  │ │DomainBlocked │ │LLMRateLimit  │
│SchemaError   │ │CircuitOpen   │ │BudgetExceeded│ │LLMTimeout    │
│ConfigError   │ │MaxIterations │ │RateLimited   │ │SearchError   │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### Error Codes

```yaml
Validation Errors (VAL_*):
  VAL_001: Invalid input schema
  VAL_002: Missing required field
  VAL_003: Invalid field value
  VAL_004: Invalid configuration

Agent Errors (AGT_*):
  AGT_001: Agent timeout
  AGT_002: Circuit breaker open
  AGT_003: Max iterations exceeded
  AGT_004: Agent not found
  AGT_005: Agent unavailable

Policy Errors (POL_*):
  POL_001: Domain blocked
  POL_002: Budget exceeded
  POL_003: Rate limit exceeded
  POL_004: Content policy violation
  POL_005: Authentication required

Service Errors (SVC_*):
  SVC_001: LLM rate limited
  SVC_002: LLM timeout
  SVC_003: Search API error
  SVC_004: External service unavailable

Storage Errors (STR_*):
  STR_001: Database connection failed
  STR_002: Redis connection failed
  STR_003: Checkpoint not found
  STR_004: Session not found
```

### Error Response Format

```json
{
  "error": {
    "code": "POL_002",
    "message": "Budget exceeded for agent searcher_v1",
    "recoverable": true,
    "details": {
      "agent_id": "searcher_v1",
      "current_spend": 0.52,
      "max_budget": 0.50,
      "session_id": "550e8400..."
    }
  },
  "request_id": "req_abc123",
  "timestamp": "2024-12-15T10:35:00Z"
}
```

---

## Configuration Schema

### Environment Variables

```yaml
# Core Application
DRX_ENV: enum [development, staging, production]
  default: development
DRX_DEBUG: boolean
  default: false
DRX_LOG_LEVEL: enum [DEBUG, INFO, WARNING, ERROR]
  default: INFO

# Database
DATABASE_URL: string (required)
  format: postgresql://user:pass@host:port/db
DATABASE_POOL_SIZE: integer
  default: 10
DATABASE_POOL_MAX_OVERFLOW: integer
  default: 20

# Redis
REDIS_URL: string (required)
  format: redis://host:port/db
REDIS_POOL_SIZE: integer
  default: 10

# LLM Gateway
OPENROUTER_API_KEY: string (required)
OPENROUTER_BASE_URL: string
  default: https://openrouter.ai/api/v1
DEFAULT_MODEL: string
  default: google/gemini-2.0-flash-exp
REASONING_MODEL: string
  default: deepseek/deepseek-r1

# Search
TAVILY_API_KEY: string (required for search)
TAVILY_MAX_RESULTS: integer
  default: 10

# Observability
PHOENIX_COLLECTOR_ENDPOINT: string
  default: http://localhost:4317
PHOENIX_PROJECT_NAME: string
  default: drx-research
OTEL_EXPORTER_OTLP_ENDPOINT: string
  default: http://localhost:4317

# Workflow Defaults
DEFAULT_MAX_ITERATIONS: integer
  default: 5
DEFAULT_TOKEN_BUDGET: integer
  default: 100000
DEFAULT_RATE_LIMIT_RPM: integer
  default: 60
DEFAULT_RATE_LIMIT_TPM: integer
  default: 100000

# Circuit Breaker Defaults
CIRCUIT_BREAKER_FAILURE_THRESHOLD: integer
  default: 5
CIRCUIT_BREAKER_SUCCESS_THRESHOLD: integer
  default: 3
CIRCUIT_BREAKER_TIMEOUT_SECONDS: integer
  default: 30
```

### Agent Manifest JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["agent_id", "version", "agent_type", "capabilities"],
  "properties": {
    "agent_id": {
      "type": "string",
      "pattern": "^[a-z][a-z0-9_]*$",
      "description": "Unique agent identifier"
    },
    "version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+\\.\\d+$",
      "description": "Semantic version"
    },
    "agent_type": {
      "type": "string",
      "enum": ["planner", "searcher", "reader", "synthesizer", "critic", "reporter"]
    },
    "capabilities": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1,
      "description": "List of capabilities this agent provides"
    },
    "allowed_domains": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Wildcard patterns for allowed domains"
    },
    "blocked_domains": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Wildcard patterns for blocked domains"
    },
    "max_budget_usd": {
      "type": "number",
      "minimum": 0,
      "description": "Maximum spend allowed per session"
    },
    "rate_limits": {
      "type": "object",
      "properties": {
        "requests_per_minute": { "type": "integer", "minimum": 1 },
        "tokens_per_minute": { "type": "integer", "minimum": 1 }
      }
    },
    "circuit_breaker": {
      "type": "object",
      "properties": {
        "failure_threshold": { "type": "integer", "minimum": 1, "default": 5 },
        "success_threshold": { "type": "integer", "minimum": 1, "default": 3 },
        "timeout_seconds": { "type": "integer", "minimum": 1, "default": 30 }
      }
    },
    "model_config": {
      "type": "object",
      "properties": {
        "model": { "type": "string" },
        "temperature": { "type": "number", "minimum": 0, "maximum": 2 },
        "max_tokens": { "type": "integer", "minimum": 1 }
      }
    }
  }
}
```

---

## Appendix: Redis Key Patterns

```yaml
Session State:
  drx:session:{session_id}:
    type: hash
    fields:
      status: string
      current_node: string
      iteration: integer
      created_at: timestamp
    ttl: 24 hours

Agent Metrics:
  drx:agent:{agent_id}:metrics:
    type: hash
    fields:
      tokens_1m: integer
      tokens_5m: integer
      latency_p50: integer
      latency_p99: integer
      error_rate: float
    ttl: 1 hour

  drx:agent:{agent_id}:invocations:
    type: sorted_set
    score: timestamp
    member: JSON {tokens, latency_ms, success}
    ttl: 1 hour

Circuit Breaker:
  drx:agent:{agent_id}:circuit:
    type: hash
    fields:
      state: string (closed|open|half_open)
      failure_count: integer
      success_count: integer
      opened_at: timestamp
      last_failure: timestamp
    ttl: none (persistent)

Rate Limiting:
  drx:agent:{agent_id}:ratelimit:{window}:
    type: sorted_set
    score: timestamp
    member: request_id
    ttl: window duration

Context Propagation:
  drx:context:{session_id}:
    type: hash
    fields:
      trace_id: string
      span_id: string
      user_id: string
      metadata: JSON
    ttl: 24 hours

Policy State:
  drx:policy:{agent_id}:spend:
    type: string (float)
    ttl: 24 hours

Event Streaming:
  drx:events:{session_id}:
    type: stream
    fields: event_type, event_data, timestamp
    ttl: 24 hours
```
