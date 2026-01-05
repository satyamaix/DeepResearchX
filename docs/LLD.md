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

```mermaid
classDiagram
    class BaseAgent {
        <<abstract>>
        -agent_id: str
        -agent_type: AgentTypeEnum
        -system_prompt: str
        -llm_client: OpenRouterClient
        -tools: list~BaseTool~
        -manifest: AgentManifest
        -tracer: Tracer
        +__call__(state: AgentState) AgentState
        +_process(state: AgentState)* AgentResponse
        +_post_process(state: AgentState, response: AgentResponse) AgentState
        +_validate_input(state: AgentState) bool
        +_emit_event(event_type: str, data: dict)
        +_record_metrics(metrics: AgentMetrics)
        +_check_budget(estimated_tokens: int) bool
        +_get_manifest() AgentManifest
    }

    class PlannerAgent {
        +_process() AgentResponse
        -_decompose() list~SubTask~
        -_prioritize(tasks: list) list
        -_build_dag() ResearchPlan
    }

    class SearcherAgent {
        +_process() AgentResponse
        -_expand_query(query: str) list~str~
        -_search(queries: list) list~SearchResult~
        -_dedupe(results: list) list
    }

    class ReaderAgent {
        +_process() AgentResponse
        -_fetch_content(url: str) str
        -_extract(content: str) Finding
        -_parse(html: str) Document
    }

    class SynthesizerAgent {
        +_process() AgentResponse
        -_aggregate(findings: list) Synthesis
        -_resolve_conflicts(findings: list) list
        -_build_argument(synthesis: Synthesis) ArgumentGraph
    }

    class CriticAgent {
        +_process() AgentResponse
        -_evaluate(synthesis: str) QualityScore
        -_find_gaps(synthesis: str) list~str~
        -_score(report: str) float
        -_verify(citations: list) list~bool~
    }

    class ReporterAgent {
        +_process() AgentResponse
        -_generate(synthesis: str) Report
        -_format(report: Report, fmt: str) str
        -_cite(report: Report, citations: list) Report
    }

    BaseAgent <|-- PlannerAgent
    BaseAgent <|-- SearcherAgent
    BaseAgent <|-- ReaderAgent
    BaseAgent <|-- SynthesizerAgent
    BaseAgent <|-- CriticAgent
    BaseAgent <|-- ReporterAgent
```

### State Classes

```mermaid
classDiagram
    class AgentState {
        <<TypedDict>>
        +messages: Annotated~list~AnyMessage~~
        +session_id: str
        +user_query: str
        +steerability: SteerabilityConfig
        +plan: ResearchPlan
        +findings: list~Finding~
        +citations: list~CitationRecord~
        +synthesis: str
        +gaps: list~str~
        +policy_violations: list~PolicyViolation~
        +final_report: str | None
        +iteration_count: int
        +token_budget: int
        +tokens_used: int
        +metrics: AgentMetrics
    }

    class ResearchPlan {
        <<TypedDict>>
        +dag_nodes: list~SubTask~
        +current_iteration: int
        +max_iterations: int
        +coverage_score: float
        +execution_order: list~str~
    }

    class SubTask {
        <<TypedDict>>
        +id: str
        +description: str
        +agent_type: str
        +dependencies: list~str~
        +status: Literal
        +inputs: dict~str, Any~
        +outputs: dict~str, Any~ | None
        +quality_score: float | None
        +retry_count: int
    }

    class CitationRecord {
        <<TypedDict>>
        +url: str
        +title: str
        +snippet: str
        +relevance_score: float
        +retrieved_at: str
    }

    class SteerabilityConfig {
        <<TypedDict>>
        +tone: str
        +format: str
        +max_sources: int
        +language: str
    }

    AgentState *-- ResearchPlan
    AgentState *-- CitationRecord
    AgentState *-- SteerabilityConfig
    ResearchPlan *-- SubTask
```

### Metadata Infrastructure Classes

```mermaid
classDiagram
    class AgentManifest {
        <<Pydantic>>
        +agent_id: str
        +version: str
        +agent_type: AgentTypeEnum
        +capabilities: list~str~
        +allowed_domains: list~str~
        +blocked_domains: list~str~
        +max_budget_usd: float
        +rate_limits: RateLimits
        +circuit_breaker: CircuitBreakerConfig
        +model_config: ModelConfig
        +metadata: dict~str, Any~
        +is_domain_allowed(url: str) bool
        +check_budget(current_spend: float) bool
        +to_dict() dict
        +from_json(path: str)$ AgentManifest
    }

    class CircuitBreaker {
        -agent_id: str
        -state: CircuitState
        -failure_count: int
        -success_count: int
        -last_failure_time: datetime | None
        -config: CircuitBreakerConfig
        -redis: RedisClient
        +call~T~(func: Callable, args, kwargs) T
        +record_success()
        +record_failure(error: Exception)
        +is_available() bool
        +get_state() CircuitState
        +reset()
        -_should_attempt_reset() bool
        -_transition_to(state: CircuitState)
        -_persist_state()
        -_load_state()
    }

    class PolicyFirewall {
        -manifest: AgentManifest
        -domain_validator: DomainValidator
        -redis: RedisClient
        -db_pool: AsyncConnectionPool
        -config: dict~str, Any~
        +check_tool_invocation(tool_name: str, args: dict) PolicyResult
        +check_domain(url: str) PolicyResult
        +check_budget(estimated_cost: float) PolicyResult
        +check_rate_limit(tokens: int) PolicyResult
        +log_violation(violation: PolicyViolation)
        -_extract_urls(args: dict) list~str~
        -_get_current_spend() float
        -_get_token_count(window: str) int
    }

    class CapabilityRouter {
        -registry: AgentRegistry
        -active_state: ActiveStateService
        -weights: ScoringWeights
        +select_agent(requirements: TaskRequirements) AgentManifest
        +get_fallback_agents(agent_id: str) list~AgentManifest~
        +score_agent(manifest: AgentManifest, requirements: TaskRequirements) float
        -_calculate_capability_match(capabilities: list, required: list) float
        -_calculate_health_score(agent_id: str) float
        -_calculate_load_score(agent_id: str) float
        -_calculate_cost_penalty(manifest: AgentManifest, tier: str) float
    }

    class CircuitState {
        <<enumeration>>
        CLOSED
        OPEN
        HALF_OPEN
    }

    CircuitBreaker --> CircuitState
    PolicyFirewall --> AgentManifest
    CapabilityRouter --> AgentManifest
```

### Service Classes

```mermaid
classDiagram
    class OpenRouterClient {
        -api_key: str
        -base_url: str
        -http_client: AsyncClient
        -default_model: str
        -tracer: Tracer
        +generate(prompt: str, model: str, kwargs) LLMResponse
        +generate_stream(prompt: str, model: str) AsyncIterator~str~
        +count_tokens(text: str) int
        +get_model_info(model: str) ModelInfo
        -_prepare_request(prompt: str, kwargs) dict
        -_handle_response(response: Response) LLMResponse
        -_handle_error(error: Exception)
    }

    class ActiveStateService {
        -redis: RedisClient
        -key_prefix: str
        +get_agent_health(agent_id: str) AgentHealth
        +set_agent_health(agent_id: str, health: AgentHealth)
        +record_invocation(agent_id: str, metrics: InvocationMetrics)
        +get_metrics(agent_id: str, window: str) AgentMetrics
        +get_circuit_state(agent_id: str) CircuitState
        +set_circuit_state(agent_id: str, state: CircuitState)
        +get_rate_limit_usage(agent_id: str) RateLimitUsage
        +increment_rate_limit(agent_id: str, tokens: int)
    }

    class ReplayRecorder {
        -session_id: str
        -db_pool: AsyncConnectionPool
        -events: list~ReplayEvent~
        -recording: bool
        +start_recording()
        +stop_recording()
        +record_event(event_type: str, data: dict, deterministic: bool)
        +record_llm_call(request: dict, response: dict)
        +record_tool_call(tool: str, args: dict, result: Any)
        +get_events(from_seq: int) list~ReplayEvent~
        +persist()
    }

    class ReplayPlayer {
        -session_id: str
        -events: list~ReplayEvent~
        -current_index: int
        -modifications: ReplayModifications
        +load_events(from_checkpoint: str)
        +replay() AsyncIterator~ReplayResult~
        +get_next_response(event_type: str) Any
        +compare_with_original(new_result: Any) ComparisonResult
        +set_modifications(mods: ReplayModifications)
    }

    OpenRouterClient --> LLMResponse
    ActiveStateService --> AgentHealth
    ActiveStateService --> CircuitState
    ReplayRecorder --> ReplayEvent
    ReplayPlayer --> ReplayEvent
```

---

## Database Design

### Entity Relationship Diagram

```mermaid
erDiagram
    RESEARCH_SESSIONS ||--o{ RESEARCH_STEPS : contains
    RESEARCH_SESSIONS ||--o{ POLICY_VIOLATIONS : has
    RESEARCH_SESSIONS ||--o{ DOCUMENT_CHUNKS : stores
    RESEARCH_SESSIONS ||--o{ REPLAY_EVENTS : records

    RESEARCH_STEPS ||--o{ TOOL_INVOCATIONS : performs
    RESEARCH_STEPS ||--o{ AGENT_INVOCATIONS : executes
    RESEARCH_STEPS ||--o{ POLICY_VIOLATIONS : triggers
    RESEARCH_STEPS ||--o| RESEARCH_STEPS : parent

    RESEARCH_SESSIONS {
        uuid id PK
        uuid user_id
        text query
        jsonb steerability
        varchar status
        jsonb config
        timestamp created_at
        timestamp updated_at
        timestamp completed_at
    }

    RESEARCH_STEPS {
        uuid id PK
        uuid session_id FK
        uuid parent_step_id FK
        varchar step_type
        enum agent_type
        varchar status
        jsonb inputs
        jsonb outputs
        jsonb metadata
        text error
        timestamp created_at
        timestamp completed_at
    }

    TOOL_INVOCATIONS {
        uuid id PK
        uuid step_id FK
        varchar tool_name
        jsonb tool_args
        jsonb tool_result
        int latency_ms
        int tokens_used
        boolean success
        timestamp created_at
    }

    AGENT_INVOCATIONS {
        uuid id PK
        uuid step_id FK
        enum agent_type
        int input_tokens
        int output_tokens
        int latency_ms
        decimal cost_usd
        text error
        timestamp created_at
    }

    POLICY_VIOLATIONS {
        uuid id PK
        uuid session_id FK
        uuid step_id FK
        varchar agent_id
        enum violation_type
        enum severity
        jsonb details
        boolean blocked
        timestamp created_at
    }

    DOCUMENT_CHUNKS {
        uuid id PK
        uuid session_id FK
        text source_url
        text content
        vector embedding
        jsonb metadata
        int chunk_index
        timestamp created_at
    }

    REPLAY_EVENTS {
        uuid id PK
        uuid session_id FK
        int sequence_num
        varchar event_type
        jsonb event_data
        boolean deterministic
        timestamp timestamp
        timestamp created_at
    }

    AGENT_REGISTRY {
        varchar id PK
        varchar version
        enum agent_type
        text[] capabilities
        text[] allowed_domains
        text[] blocked_domains
        decimal max_budget
        jsonb rate_limits
        jsonb circuit_config
        jsonb model_config
        boolean is_active
        timestamp created_at
        timestamp updated_at
    }
```

### LangGraph Checkpoint Tables

```mermaid
erDiagram
    CHECKPOINTS ||--o{ CHECKPOINT_WRITES : has
    CHECKPOINTS ||--o{ CHECKPOINT_BLOBS : stores

    CHECKPOINTS {
        text thread_id PK
        text checkpoint_ns PK
        text checkpoint_id PK
        text parent_checkpoint_id
        text type
        jsonb checkpoint
        jsonb metadata
    }

    CHECKPOINT_WRITES {
        text thread_id PK
        text checkpoint_ns PK
        text checkpoint_id PK
        text task_id PK
        int idx PK
        text channel
        text type
        jsonb value
    }

    CHECKPOINT_BLOBS {
        text thread_id PK
        text checkpoint_ns PK
        text channel PK
        text version PK
        text type
        bytea blob
    }
```

### Table Indexes

```mermaid
flowchart TB
    subgraph CoreIndexes["Core Table Indexes"]
        SessionIdx["research_sessions<br/>idx_sessions_user (user_id)<br/>idx_sessions_status (status)<br/>idx_sessions_created (created_at DESC)"]

        StepIdx["research_steps<br/>idx_steps_session (session_id)<br/>idx_steps_status (status)<br/>idx_steps_parent (parent_step_id)"]

        ViolationIdx["policy_violations<br/>idx_violations_session (session_id)<br/>idx_violations_type (violation_type)"]

        ReplayIdx["replay_events<br/>idx_replay_session_seq (session_id, sequence_num)"]
    end

    subgraph VectorIndex["Vector Index"]
        ChunkEmbed["document_chunks<br/>idx_chunks_embedding<br/>USING ivfflat (embedding vector_cosine_ops)<br/>WITH (lists = 100)"]
    end
```

---

## Sequence Diagrams

### Research Session Execution

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API
    participant Orchestrator
    participant Agents
    participant Services
    participant DB

    Client->>API: POST /interactions
    API->>API: validate request
    API->>DB: INSERT session
    DB-->>API: session_id
    API-->>Client: 202 Accepted

    rect rgb(240, 240, 240)
        Note over Orchestrator,DB: ASYNC WORKER EXECUTION
        Orchestrator->>DB: load state

        loop For each agent
            Orchestrator->>Agents: run planner
            Agents->>Services: LLM call
            Services-->>Agents: response
            Agents-->>Orchestrator: updated state
            Orchestrator->>DB: checkpoint
        end

        Orchestrator->>Agents: run searcher
        Agents->>Services: Tavily search
        Services-->>Agents: results
        Agents-->>Orchestrator: updated state
        Orchestrator->>DB: checkpoint

        Note over Orchestrator,Agents: ... (reader, synthesizer, critic) ...

        Orchestrator->>Agents: run reporter
        Agents-->>Orchestrator: final report
        Orchestrator->>DB: UPDATE session
    end
```

### SSE Streaming Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Redis
    participant Orchestrator

    Client->>API: GET /stream
    API->>Redis: SUBSCRIBE drx:events:{id}
    API-->>Client: SSE: connected

    loop Event Stream
        Orchestrator->>Redis: PUBLISH event
        Redis-->>API: event received
        API-->>Client: SSE: thought_summary

        Orchestrator->>Redis: PUBLISH event
        Redis-->>API: event received
        API-->>Client: SSE: content.delta

        Orchestrator->>Redis: PUBLISH checkpoint
        Redis-->>API: checkpoint received
        API-->>Client: SSE: checkpoint
    end

    Orchestrator->>Redis: PUBLISH complete
    Redis-->>API: complete received
    API-->>Client: SSE: interaction.complete
    API-->>Client: Connection closed
```

### Circuit Breaker Flow

```mermaid
sequenceDiagram
    participant Agent
    participant CB as CircuitBreaker
    participant Redis
    participant LLM as LLM/Tool

    rect rgb(200, 255, 200)
        Note over Agent,LLM: SUCCESS SCENARIO
        Agent->>CB: call(fn, args)
        CB->>Redis: GET state
        Redis-->>CB: state=CLOSED
        CB->>LLM: execute fn
        LLM-->>CB: success
        CB->>Redis: INCR success
        CB-->>Agent: result
    end

    rect rgb(255, 200, 200)
        Note over Agent,LLM: FAILURE SCENARIO
        Agent->>CB: call(fn, args)
        CB->>LLM: execute fn
        LLM-->>CB: ERROR
        CB->>Redis: INCR failure
        CB->>Redis: SET state=OPEN (if threshold reached)
        CB-->>Agent: raise error
    end

    rect rgb(255, 255, 200)
        Note over Agent,LLM: OPEN STATE CALL
        Agent->>CB: call(fn, args)
        CB->>Redis: GET state
        Redis-->>CB: state=OPEN
        CB-->>Agent: CircuitOpenError (no external call)
    end
```

### Policy Firewall Check Flow

```mermaid
sequenceDiagram
    participant Agent
    participant PF as PolicyFirewall
    participant DV as DomainValidator
    participant Redis
    participant DB

    Agent->>PF: check_tool_invocation(tool, args)

    PF->>DV: extract_urls(args)
    DV-->>PF: urls

    loop For each URL
        PF->>DV: is_allowed(url)
        DV-->>PF: allowed/blocked
    end

    PF->>DB: GET current_spend
    DB-->>PF: spend amount
    PF->>PF: check budget (spend + estimated <= max)

    PF->>Redis: GET token_count
    Redis-->>PF: count
    PF->>PF: check rate limit (count <= limit)

    PF-->>Agent: PolicyResult

    alt Violation occurred
        PF->>DB: log_violation()
    end
```

### Checkpoint Resume Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Orchestrator
    participant Checkpointer
    participant DB

    Client->>API: POST /resume?checkpoint=chk_002

    API->>DB: GET session status
    DB-->>API: status=paused

    API->>Orchestrator: resume(session_id, checkpoint_id)

    Orchestrator->>Checkpointer: get(checkpoint_id)
    Checkpointer->>DB: SELECT FROM checkpoints
    DB-->>Checkpointer: checkpoint data
    Checkpointer-->>Orchestrator: CheckpointTuple

    Orchestrator->>Orchestrator: reconstruct AgentState
    Orchestrator->>Orchestrator: identify next node

    loop Continue from checkpoint
        Orchestrator->>Orchestrator: execute remaining nodes
        Orchestrator->>Checkpointer: save checkpoint
    end

    Orchestrator->>DB: UPDATE session status=completed
    Orchestrator-->>API: complete

    API-->>Client: 200 OK {status: completed}
```

---

## API Specifications

### Endpoint Flow Diagram

```mermaid
flowchart LR
    subgraph Interactions["/api/v1/interactions"]
        POST_Create["POST<br/>Create interaction"]
        GET_List["GET<br/>List interactions"]
        GET_Detail["GET /{id}<br/>Get details"]
        DELETE_Cancel["DELETE /{id}<br/>Cancel"]
    end

    subgraph Stream["/api/v1/interactions/{id}/stream"]
        GET_Stream["GET<br/>SSE stream"]
    end

    subgraph Replay["/api/v1/interactions/{id}/replay"]
        POST_Replay["POST<br/>Start replay"]
        GET_Events["GET /events<br/>Get events"]
        POST_Compare["POST /compare<br/>Compare runs"]
    end

    subgraph Health["/api/v1/health"]
        GET_Health["GET<br/>Health check"]
    end

    Client["Client"] --> Interactions
    Client --> Stream
    Client --> Replay
    Client --> Health
```

### Request/Response Schema

```mermaid
classDiagram
    class CreateInteractionRequest {
        +query: str
        +steerability: SteerabilityConfig
        +config: InteractionConfig
    }

    class SteerabilityConfig {
        +tone: Literal~academic,executive,technical,casual~
        +format: Literal~markdown,html,json,plain~
        +max_sources: int
        +language: str
    }

    class InteractionConfig {
        +max_iterations: int
        +thinking_summaries: bool
        +token_budget: int
    }

    class CreateInteractionResponse {
        +id: str
        +status: Literal~queued~
        +created_at: str
    }

    class SSEEvent {
        +event: str
        +id: str
        +data: dict
    }

    class HealthResponse {
        +status: Literal~healthy,degraded,unhealthy~
        +version: str
        +components: dict~str,ComponentHealth~
        +timestamp: str
    }

    CreateInteractionRequest *-- SteerabilityConfig
    CreateInteractionRequest *-- InteractionConfig
```

### SSE Event Types

```mermaid
flowchart TB
    subgraph EventTypes["SSE Event Types"]
        Start["interaction.start<br/>{id, status}"]
        Thought["thought_summary<br/>{agent, text}"]
        ToolUse["tool.use<br/>{tool, args}"]
        ToolResult["tool.result<br/>{tool, result}"]
        Content["content.delta<br/>{text}"]
        Checkpoint["checkpoint<br/>{checkpoint_id, node}"]
        Error["error<br/>{error, recoverable}"]
        Complete["interaction.complete<br/>{id, status}"]
    end

    Start --> Thought
    Thought --> ToolUse
    ToolUse --> ToolResult
    ToolResult --> Content
    Content --> Checkpoint
    Checkpoint --> Thought
    Thought --> Complete
    Error -.-> Complete
```

---

## Component Interfaces

### Interface Hierarchy

```mermaid
classDiagram
    class AgentInterface {
        <<interface>>
        +agent_id: str*
        +agent_type: str*
        +__call__(state: AgentState)* AgentState
        +validate(state: AgentState)* bool
    }

    class ToolInterface~TInput, TOutput~ {
        <<interface>>
        +name: str*
        +description: str*
        +execute(input: TInput)* TOutput
        +get_schema()* dict
    }

    class LLMClientInterface {
        <<interface>>
        +generate(prompt, model, kwargs)* LLMResponse
        +generate_stream(prompt, model, kwargs)* AsyncIterator~str~
    }

    class CacheInterface {
        <<interface>>
        +get(key: str)* Any
        +set(key: str, value: Any, ttl: int)*
        +delete(key: str)* bool
    }

    class CheckpointerInterface {
        <<interface>>
        +put(config, checkpoint, metadata)* RunnableConfig
        +get(config)* CheckpointTuple
        +list(config, filter, before, limit)* AsyncIterator~CheckpointTuple~
    }

    AgentInterface <|.. BaseAgent
    ToolInterface <|.. BaseTool
    LLMClientInterface <|.. OpenRouterClient
    CacheInterface <|.. RedisClient
    CheckpointerInterface <|.. AsyncPostgresSaver
```

---

## Error Handling

### Error Hierarchy

```mermaid
classDiagram
    class DRXError {
        <<base>>
        +message: str
        +code: str
        +recoverable: bool
        +details: dict~str, Any~
    }

    class ValidationError {
        +code: VAL_*
        +recoverable: false
    }

    class AgentError {
        +code: AGT_*
        +recoverable: varies
    }

    class PolicyError {
        +code: POL_*
        +recoverable: true
    }

    class ServiceError {
        +code: SVC_*
        +recoverable: varies
    }

    class StorageError {
        +code: STR_*
        +recoverable: varies
    }

    DRXError <|-- ValidationError
    DRXError <|-- AgentError
    DRXError <|-- PolicyError
    DRXError <|-- ServiceError
    DRXError <|-- StorageError

    class InputValidationError {
        +code: VAL_001
    }
    class SchemaError {
        +code: VAL_002
    }
    class ConfigError {
        +code: VAL_004
    }

    class AgentTimeoutError {
        +code: AGT_001
    }
    class CircuitOpenError {
        +code: AGT_002
    }
    class MaxIterationsError {
        +code: AGT_003
    }

    class DomainBlockedError {
        +code: POL_001
    }
    class BudgetExceededError {
        +code: POL_002
    }
    class RateLimitedError {
        +code: POL_003
    }

    class LLMRateLimitError {
        +code: SVC_001
    }
    class LLMTimeoutError {
        +code: SVC_002
    }
    class SearchError {
        +code: SVC_003
    }

    ValidationError <|-- InputValidationError
    ValidationError <|-- SchemaError
    ValidationError <|-- ConfigError

    AgentError <|-- AgentTimeoutError
    AgentError <|-- CircuitOpenError
    AgentError <|-- MaxIterationsError

    PolicyError <|-- DomainBlockedError
    PolicyError <|-- BudgetExceededError
    PolicyError <|-- RateLimitedError

    ServiceError <|-- LLMRateLimitError
    ServiceError <|-- LLMTimeoutError
    ServiceError <|-- SearchError
```

### Error Codes Reference

| Category | Code | Description | Recoverable |
|----------|------|-------------|-------------|
| **Validation** | VAL_001 | Invalid input schema | No |
| | VAL_002 | Missing required field | No |
| | VAL_003 | Invalid field value | No |
| | VAL_004 | Invalid configuration | No |
| **Agent** | AGT_001 | Agent timeout | Yes |
| | AGT_002 | Circuit breaker open | Yes |
| | AGT_003 | Max iterations exceeded | No |
| | AGT_004 | Agent not found | No |
| | AGT_005 | Agent unavailable | Yes |
| **Policy** | POL_001 | Domain blocked | Yes |
| | POL_002 | Budget exceeded | Yes |
| | POL_003 | Rate limit exceeded | Yes |
| | POL_004 | Content policy violation | No |
| | POL_005 | Authentication required | No |
| **Service** | SVC_001 | LLM rate limited | Yes |
| | SVC_002 | LLM timeout | Yes |
| | SVC_003 | Search API error | Yes |
| | SVC_004 | External service unavailable | Yes |
| **Storage** | STR_001 | Database connection failed | Yes |
| | STR_002 | Redis connection failed | Yes |
| | STR_003 | Checkpoint not found | No |
| | STR_004 | Session not found | No |

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

### Environment Variable Groups

```mermaid
flowchart TB
    subgraph Core["Core Application"]
        DRX_ENV["DRX_ENV<br/>development|staging|production"]
        DRX_DEBUG["DRX_DEBUG<br/>boolean"]
        DRX_LOG_LEVEL["DRX_LOG_LEVEL<br/>DEBUG|INFO|WARNING|ERROR"]
    end

    subgraph Database["Database"]
        DATABASE_URL["DATABASE_URL<br/>postgresql://..."]
        DATABASE_POOL_SIZE["DATABASE_POOL_SIZE<br/>default: 10"]
        DATABASE_POOL_MAX["DATABASE_POOL_MAX_OVERFLOW<br/>default: 20"]
    end

    subgraph Redis["Redis"]
        REDIS_URL["REDIS_URL<br/>redis://..."]
        REDIS_POOL_SIZE["REDIS_POOL_SIZE<br/>default: 10"]
    end

    subgraph LLM["LLM Gateway"]
        OPENROUTER_API_KEY["OPENROUTER_API_KEY<br/>required"]
        OPENROUTER_BASE_URL["OPENROUTER_BASE_URL<br/>default: openrouter.ai"]
        DEFAULT_MODEL["DEFAULT_MODEL<br/>google/gemini-3-flash-preview"]
        REASONING_MODEL["REASONING_MODEL<br/>deepseek/deepseek-r1"]
    end

    subgraph Search["Search"]
        SEARCH_ENGINE["SEARCH_ENGINE<br/>native|exa|auto"]
        SEARCH_MODEL["SEARCH_MODEL<br/>openai/gpt-oss-20b:free"]
        TAVILY_ENABLED["TAVILY_ENABLED<br/>default: false"]
        TAVILY_API_KEY["TAVILY_API_KEY<br/>optional"]
    end

    subgraph Observability["Observability"]
        PHOENIX_ENDPOINT["PHOENIX_COLLECTOR_ENDPOINT<br/>http://localhost:4317"]
        PHOENIX_PROJECT["PHOENIX_PROJECT_NAME<br/>drx-research"]
    end

    subgraph Workflow["Workflow Defaults"]
        MAX_ITERATIONS["DEFAULT_MAX_ITERATIONS<br/>default: 5"]
        TOKEN_BUDGET["DEFAULT_TOKEN_BUDGET<br/>default: 100000"]
        RATE_LIMIT_RPM["DEFAULT_RATE_LIMIT_RPM<br/>default: 60"]
    end

    subgraph CircuitBreaker["Circuit Breaker"]
        CB_FAILURE["CIRCUIT_BREAKER_FAILURE_THRESHOLD<br/>default: 5"]
        CB_SUCCESS["CIRCUIT_BREAKER_SUCCESS_THRESHOLD<br/>default: 3"]
        CB_TIMEOUT["CIRCUIT_BREAKER_TIMEOUT_SECONDS<br/>default: 30"]
    end
```

### Agent Manifest Schema

```mermaid
classDiagram
    class AgentManifestSchema {
        +agent_id: str
        +version: str
        +agent_type: enum
        +capabilities: array~str~
        +allowed_domains: array~str~
        +blocked_domains: array~str~
        +max_budget_usd: number
        +rate_limits: RateLimitsSchema
        +circuit_breaker: CircuitBreakerSchema
        +model_config: ModelConfigSchema
    }

    class RateLimitsSchema {
        +requests_per_minute: int
        +tokens_per_minute: int
    }

    class CircuitBreakerSchema {
        +failure_threshold: int
        +success_threshold: int
        +timeout_seconds: int
    }

    class ModelConfigSchema {
        +model: str
        +temperature: number
        +max_tokens: int
    }

    AgentManifestSchema *-- RateLimitsSchema
    AgentManifestSchema *-- CircuitBreakerSchema
    AgentManifestSchema *-- ModelConfigSchema
```

---

## Appendix: Redis Key Patterns

```mermaid
flowchart TB
    subgraph SessionKeys["Session State"]
        Session["drx:session:{session_id}<br/>type: hash<br/>ttl: 24 hours<br/>─────────────<br/>status: string<br/>current_node: string<br/>iteration: integer<br/>created_at: timestamp"]
    end

    subgraph AgentKeys["Agent Metrics"]
        Metrics["drx:agent:{agent_id}:metrics<br/>type: hash<br/>ttl: 1 hour<br/>─────────────<br/>tokens_1m: integer<br/>tokens_5m: integer<br/>latency_p50: integer<br/>latency_p99: integer<br/>error_rate: float"]

        Invocations["drx:agent:{agent_id}:invocations<br/>type: sorted_set<br/>ttl: 1 hour<br/>─────────────<br/>score: timestamp<br/>member: {tokens, latency_ms}"]
    end

    subgraph CircuitKeys["Circuit Breaker"]
        Circuit["drx:agent:{agent_id}:circuit<br/>type: hash<br/>ttl: persistent<br/>─────────────<br/>state: closed|open|half_open<br/>failure_count: integer<br/>success_count: integer<br/>opened_at: timestamp<br/>last_failure: timestamp"]
    end

    subgraph RateLimitKeys["Rate Limiting"]
        RateLimit["drx:agent:{agent_id}:ratelimit:{window}<br/>type: sorted_set<br/>ttl: window duration<br/>─────────────<br/>score: timestamp<br/>member: request_id"]
    end

    subgraph ContextKeys["Context Propagation"]
        Context["drx:context:{session_id}<br/>type: hash<br/>ttl: 24 hours<br/>─────────────<br/>trace_id: string<br/>span_id: string<br/>user_id: string<br/>metadata: JSON"]
    end

    subgraph PolicyKeys["Policy State"]
        Spend["drx:policy:{agent_id}:spend<br/>type: string (float)<br/>ttl: 24 hours"]
    end

    subgraph EventKeys["Event Streaming"]
        Events["drx:events:{session_id}<br/>type: stream<br/>ttl: 24 hours<br/>─────────────<br/>event_type<br/>event_data<br/>timestamp"]
    end
```
