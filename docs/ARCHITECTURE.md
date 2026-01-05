# DRX System Architecture

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Agentic Workflow](#agentic-workflow)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [API Gateway](#api-gateway)
6. [External Integrations](#external-integrations)
7. [Instrumentation & Observability](#instrumentation--observability)
8. [Evaluation Pipeline](#evaluation-pipeline)

---

## High-Level Architecture

### System Overview

```mermaid
flowchart TB
    subgraph ClientLayer["Client Layer"]
        WebUI["Web UI<br/>(React/Vue)"]
        CLI["CLI Tool<br/>(Python)"]
        SDK["SDK Client<br/>(Python/JS)"]
        Webhook["Webhook<br/>Consumer"]
    end

    subgraph APIGateway["API Gateway Layer"]
        subgraph Middleware["Middleware Stack"]
            Auth["Auth/RBAC"]
            RateLimit["Rate Limit"]
            CORS["CORS"]
            Validation["Request<br/>Validation"]
        end
        subgraph Routes["Route Handlers"]
            POST_INT["POST /interactions"]
            GET_STREAM["GET /stream"]
            POST_REPLAY["POST /replay"]
            GET_HEALTH["GET /health"]
        end
    end

    subgraph Orchestration["Orchestration Layer"]
        subgraph LangGraph["LangGraph StateGraph"]
            Planner["Planner<br/>Agent"]
            Searcher["Searcher<br/>Agent"]
            Reader["Reader<br/>Agent"]
            Synthesizer["Synthesizer<br/>Agent"]
            Critic["Critic<br/>Agent"]
            Reporter["Reporter<br/>Agent"]
        end
    end

    subgraph MetadataLayer["Metadata Layer"]
        PolicyFW["Policy<br/>Firewall"]
        CircuitB["Circuit<br/>Breaker"]
        CapRouter["Capability<br/>Router"]
        ContextProp["Context<br/>Propagator"]
    end

    subgraph DataLayer["Data Layer"]
        Postgres[("PostgreSQL<br/>+ pgvector")]
        Redis[("Redis<br/>(Valkey)")]
        ObjectStore[("Object Store<br/>(Optional)")]
    end

    subgraph ExternalServices["External Services"]
        OpenRouter["OpenRouter<br/>(LLM API)"]
        Tavily["Tavily<br/>(Web Search)"]
        Phoenix["Phoenix<br/>(Observability)"]
    end

    ClientLayer -->|"HTTPS/WebSocket"| APIGateway
    APIGateway --> Orchestration
    Orchestration --> MetadataLayer
    MetadataLayer --> DataLayer
    Orchestration --> ExternalServices

    Planner --> Searcher --> Reader --> Synthesizer
    Synthesizer --> Critic
    Critic --> Reporter
    Critic -.->|"Gaps Found"| Planner
```

### Component Summary

| Layer | Components | Technology |
|-------|------------|------------|
| **Client** | Web UI, CLI, SDK, Webhooks | React, Python, REST/SSE |
| **API Gateway** | FastAPI, Middleware Stack | FastAPI 0.115+, Pydantic |
| **Orchestration** | LangGraph StateGraph, 6 Agents | LangGraph 1.0+, LangChain |
| **Metadata** | Policy, Circuit Breaker, Routing | Custom Python modules |
| **Data** | PostgreSQL, Redis, Object Store | PostgreSQL 16+, Redis 7+ |
| **External** | LLM, Search, Observability | OpenRouter, Tavily, Phoenix |

---

## Agentic Workflow

### Research DAG (Directed Acyclic Graph)

```mermaid
flowchart TB
    UserQuery["User Request<br/>(Research Query)"]

    subgraph PlannerBox["PLANNER AGENT"]
        P1["Query analysis"]
        P2["Sub-question decomposition"]
        P3["DAG generation"]
        P4["Priority assignment"]
    end

    subgraph SearcherBox["SEARCHER AGENT"]
        S1["Query expansion"]
        S2["Web search (OpenRouter/Tavily)"]
        S3["RAG retrieval (pgvector)"]
        S4["Source deduplication"]
        S5["Relevance filtering"]
    end

    subgraph ReaderFanOut["Parallel Fan-Out"]
        Reader1["READER AGENT<br/>(Source 1)<br/>HTML parsing"]
        Reader2["READER AGENT<br/>(Source 2)<br/>PDF extraction"]
        ReaderN["READER AGENT<br/>(Source N)<br/>Entity extract"]
    end

    subgraph SynthesizerBox["SYNTHESIZER AGENT"]
        SY1["Finding aggregation"]
        SY2["Conflict detection"]
        SY3["Argument graph building"]
        SY4["Consensus formation"]
        SY5["Evidence weighting"]
    end

    subgraph CriticBox["CRITIC AGENT"]
        C1["Quality assessment"]
        C2["Gap identification"]
        C3["Source verification"]
        C4["Hallucination detection"]
        C5["Coverage scoring"]
    end

    CoverageCheck{"Coverage OK?"}

    subgraph ReporterBox["REPORTER AGENT"]
        R1["Report generation"]
        R2["Citation formatting"]
        R3["Executive summary"]
        R4["Multi-format export<br/>(MD, HTML, PDF, JSON)"]
    end

    FinalOutput["Final Output<br/>(Research Report)"]

    UserQuery --> PlannerBox
    PlannerBox --> SearcherBox
    SearcherBox --> ReaderFanOut
    ReaderFanOut --> SynthesizerBox
    SynthesizerBox --> CriticBox
    CriticBox --> CoverageCheck

    CoverageCheck -->|"YES (complete)"| ReporterBox
    CoverageCheck -->|"NO (gaps found)"| PlannerBox

    ReporterBox --> FinalOutput
```

### State Machine Diagram

```mermaid
stateDiagram-v2
    [*] --> PENDING: init

    PENDING --> RUNNING: start()

    state RUNNING {
        [*] --> plan
        plan --> search
        search --> read
        read --> synthesize
        synthesize --> critique

        critique --> report: complete
        critique --> plan: gaps found
        critique --> [*]: fail
    }

    RUNNING --> COMPLETED: report done
    RUNNING --> FAILED: error
    RUNNING --> PAUSED: pause()
    RUNNING --> CANCELLED: cancel()

    PAUSED --> RUNNING: resume()
    PAUSED --> CANCELLED: cancel()

    COMPLETED --> [*]
    FAILED --> [*]
    CANCELLED --> [*]
```

### Agent Communication Pattern

```mermaid
flowchart TB
    subgraph SharedState["Shared State (AgentState)"]
        direction LR
        messages["messages<br/>(chat log)"]
        plan["plan<br/>(DAG nodes)"]
        findings["findings<br/>(extracted)"]
        citations["citations<br/>(sources)"]
        synthesis["synthesis<br/>(combined)"]
        gaps["gaps<br/>(missing)"]
        final_report["final_report<br/>(output)"]
        iteration["iteration<br/>(count)"]
        tokens_used["tokens_used<br/>(budget)"]
        metrics["metrics<br/>(quality)"]
    end

    Agent1["Agent 1<br/>read → process → write"]
    Agent2["Agent 2<br/>read → process → write"]
    AgentN["Agent N<br/>read → process → write"]

    SharedState --> Agent1
    SharedState --> Agent2
    SharedState --> AgentN

    Agent1 --> SharedState
    Agent2 --> SharedState
    AgentN --> SharedState
```

---

## Component Architecture

### API Gateway Components

```mermaid
flowchart TB
    subgraph FastAPI["FastAPI Application"]
        subgraph MiddlewareStack["Middleware Stack"]
            direction LR
            ReqID["Request ID"]
            AuthCheck["Auth Check"]
            RateLim["Rate Limit"]
            TimeTrack["Time Track"]
        end

        subgraph RouteHandlers["Route Handlers"]
            Interactions["/api/v1/interactions<br/>POST - Create<br/>GET - List<br/>GET /{id} - Details<br/>DELETE /{id} - Cancel"]
            Stream["/api/v1/interactions/{id}/stream<br/>GET - SSE stream<br/>Supports Last-Event-ID"]
            Replay["/api/v1/interactions/{id}/replay<br/>POST - Start replay<br/>GET /events - Get events<br/>POST /compare - Compare"]
            Health["/api/v1/health<br/>GET - Health check"]
        end

        subgraph DI["Dependency Injection"]
            DatabaseDep["DatabaseDep<br/>(psycopg)"]
            RedisDep["RedisDep<br/>(redis)"]
            OrchestratorDep["OrchestratorDep<br/>(LangGraph)"]
            CurrentUserDep["CurrentUserDep<br/>(auth)"]
        end
    end

    Request["Request"] --> MiddlewareStack
    MiddlewareStack --> RouteHandlers
    RouteHandlers --> DI
```

### Agent Architecture

```mermaid
classDiagram
    class BaseAgent {
        +agent_id: str
        +agent_type: str
        +system_prompt: str
        +llm_client: LLMClient
        +tools: list[Tool]
        +manifest: AgentManifest
        +__call__(state) AgentState
        +_process(state) AgentResponse
        +_post_process(state, response) AgentState
        +_validate_input(state) bool
        +_emit_event(event_type, data)
        +_record_metrics(metrics)
    }

    class PlannerAgent {
        +_decompose()
        +_prioritize()
        +_build_dag()
    }

    class SearcherAgent {
        +_expand_query()
        +_search()
        +_dedupe()
    }

    class ReaderAgent {
        +_fetch_content()
        +_extract()
        +_parse()
    }

    class SynthesizerAgent {
        +_aggregate()
        +_resolve_conflicts()
        +_build_argument()
    }

    class CriticAgent {
        +_evaluate()
        +_find_gaps()
        +_score()
        +_verify()
    }

    class ReporterAgent {
        +_generate()
        +_format()
        +_cite()
    }

    BaseAgent <|-- PlannerAgent
    BaseAgent <|-- SearcherAgent
    BaseAgent <|-- ReaderAgent
    BaseAgent <|-- SynthesizerAgent
    BaseAgent <|-- CriticAgent
    BaseAgent <|-- ReporterAgent

    class ToolIntegration {
        +OpenRouterSearch
        +TavilySearch
        +RAGRetriever
        +HTMLParser
        +PDFExtractor
    }

    PlannerAgent --> ToolIntegration
    SearcherAgent --> ToolIntegration
    ReaderAgent --> ToolIntegration
```

### Metadata Layer Architecture

```mermaid
flowchart TB
    subgraph MetadataInfra["Metadata Infrastructure"]
        subgraph ManifestSystem["Agent Manifest System"]
            JSONSchema["JSON Schema<br/>(Validation)"]
            PydanticModel["Pydantic Model<br/>(Type Safety)"]
            RegistryLoad["Registry Load<br/>(Database)"]

            JSONSchema --> PydanticModel --> RegistryLoad
        end

        subgraph PolicyFirewall["Policy Firewall"]
            DomainValidator["Domain Validator<br/>URL extraction<br/>Wildcard match<br/>Block/allow"]
            BudgetEnforcer["Budget Enforcer<br/>Spend tracking<br/>Cost estimate<br/>Budget check"]
            RateLimiter["Rate Limiter<br/>Sliding window<br/>Token counting<br/>Burst control"]
            ViolationLogger["Violation Logger<br/>PostgreSQL audit<br/>Redis events"]

            DomainValidator --> ViolationLogger
            BudgetEnforcer --> ViolationLogger
            RateLimiter --> ViolationLogger
        end

        subgraph CircuitBreaker["Circuit Breaker"]
            CLOSED["CLOSED<br/>(normal)"]
            OPEN["OPEN<br/>(failing)"]
            HALF_OPEN["HALF-OPEN<br/>(testing)"]

            CLOSED -->|"failure threshold"| OPEN
            OPEN -->|"timeout"| HALF_OPEN
            HALF_OPEN -->|"success threshold"| CLOSED
            HALF_OPEN -->|"failure"| OPEN
        end

        subgraph CapabilityRouting["Capability-Based Routing"]
            TaskReqs["Task Requirements<br/>Capabilities<br/>Domain needs<br/>Cost tier<br/>Latency req"]
            AgentScoring["Agent Scoring<br/>Capability match: 0.3<br/>Health: 0.2<br/>Load: 0.2<br/>Cost: -0.1"]
            Selection["Selection<br/>Best match<br/>Fallback<br/>Error if none"]

            TaskReqs --> AgentScoring --> Selection
        end
    end
```

---

## Data Flow

### Request Lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as API Gateway
    participant DB as PostgreSQL
    participant R as Redis
    participant O as Orchestrator
    participant A as Agents
    participant LLM as OpenRouter

    C->>API: POST /api/v1/interactions
    API->>API: Validate & Authenticate
    API->>DB: INSERT research_session
    DB-->>API: session_id
    API->>R: SET drx:session:{id}
    API-->>C: 202 Accepted {id, status: queued}

    Note over O,LLM: Async Worker Execution

    O->>DB: Load session state

    loop For each agent in DAG
        O->>A: Run agent
        A->>LLM: Generate completion
        LLM-->>A: Response
        A->>R: Emit SSE event
        A-->>O: Updated state
        O->>DB: Save checkpoint
    end

    O->>DB: UPDATE session status=completed
    O->>R: PUBLISH complete event
```

### SSE Streaming Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant API as API Server
    participant R as Redis
    participant O as Orchestrator

    C->>API: GET /stream
    API->>R: SUBSCRIBE drx:events:{id}
    API-->>C: SSE: connected

    loop Event Stream
        O->>R: PUBLISH event
        R-->>API: event received
        API-->>C: SSE: thought_summary
        O->>R: PUBLISH event
        R-->>API: event received
        API-->>C: SSE: content.delta
        O->>R: PUBLISH checkpoint
        R-->>API: checkpoint received
        API-->>C: SSE: checkpoint
    end

    O->>R: PUBLISH complete
    R-->>API: complete received
    API-->>C: SSE: interaction.complete
    API-->>C: Connection closed
```

### Checkpoint & Resume Flow

```mermaid
flowchart TB
    subgraph NormalExecution["Normal Execution"]
        Node1["Node 1<br/>(Planner)"]
        Node2["Node 2<br/>(Searcher)"]
        Node3["Node 3<br/>(Reader)"]

        Node1 --> Node2 --> Node3
    end

    subgraph Checkpointing["AsyncPostgresSaver"]
        CP["Checkpoint saved after each node:<br/>checkpoint_id<br/>thread_id<br/>state<br/>metadata"]
    end

    Node1 --> CP
    Node2 --> CP
    Node3 --> CP

    CP --> Interruption["INTERRUPTION<br/>(error, timeout, pause)"]
    Interruption --> Paused["Session Paused<br/>State preserved"]

    Paused --> Resume["RESUME REQUEST<br/>POST /resume?checkpoint=chk_002"]

    Resume --> LoadState["Resume from Checkpoint<br/>1. Load state from PostgreSQL<br/>2. Reconstruct AgentState<br/>3. Identify next node<br/>4. Continue execution"]

    LoadState --> Node3Continue["Node 3<br/>(continue)"]
    Node3Continue --> Node4["Node 4<br/>(Synthesize)"]
    Node4 --> Node5["Node 5<br/>(Report)"]
```

---

## API Gateway

### Authentication Flow

```mermaid
flowchart TB
    Request["Incoming Request<br/>Headers:<br/>Authorization: Bearer<br/>X-API-Key: drx_..."]

    CheckHeaders["Check Auth Headers"]

    BearerToken["Bearer Token<br/>JWT Validation<br/>Signature<br/>Expiration<br/>Claims"]
    APIKey["API Key<br/>Key Lookup<br/>Redis cache<br/>DB fallback<br/>Permissions"]
    NoAuth["No Auth<br/>Dev Mode Only<br/>is_dev check<br/>Default user"]

    UserObject["User Object<br/>id: str<br/>email: str | None<br/>is_active: bool<br/>is_admin: bool<br/>rate_limit_tier: str"]

    Request --> CheckHeaders
    CheckHeaders --> BearerToken
    CheckHeaders --> APIKey
    CheckHeaders --> NoAuth

    BearerToken --> UserObject
    APIKey --> UserObject
    NoAuth --> UserObject
```

### Rate Limiting Architecture

```mermaid
sequenceDiagram
    participant Req as Request
    participant RL as Rate Limiter
    participant Redis as Redis

    Req->>RL: Check Limit
    RL->>Redis: ZREMRANGEBYSCORE (remove old)
    RL->>Redis: ZCARD (count current window)
    Redis-->>RL: current_count

    alt count < limit
        RL->>Redis: ZADD request, Set TTL
        RL-->>Req: Allow request
    else count >= limit
        RL-->>Req: Return 429 + Retry-After
    end
```

**Rate Limit Tiers:**

| Tier | Requests/min | Requests/hour | Burst Limit |
|------|--------------|---------------|-------------|
| standard | 60 | 1,000 | 10 |
| premium | 300 | 5,000 | 50 |
| unlimited | 10,000 | 100,000 | 1,000 |

---

## External Integrations

### OpenRouter LLM Integration

```mermaid
sequenceDiagram
    participant Agent as DRX Agent
    participant Client as OpenRouter Client
    participant API as OpenRouter API

    Agent->>Client: generate(prompt, model)
    Client->>API: POST /api/v1/chat/completions
    Note right of API: Headers:<br/>Authorization: Bearer<br/>HTTP-Referer: app-url<br/>X-Title: DRX
    Note right of API: Body:<br/>model: google/gemini...<br/>messages: [...]<br/>temperature: 0.7<br/>max_tokens: 4096
    API-->>Client: Streaming response
    Client-->>Agent: Parsed response
```

**Supported Models:**

| Model ID | Context | Use Case |
|----------|---------|----------|
| google/gemini-3-flash-preview | 1M | Default, fast |
| google/gemini-3-pro-preview | 1M | Complex reasoning |
| anthropic/claude-3.5-sonnet | 200K | High quality |
| deepseek/deepseek-r1 | 128K | Reasoning tasks |
| openai/gpt-4o | 128K | General purpose |
| openai/gpt-oss-20b:free | 131K | Free search model |

### Web Search Integration

```mermaid
sequenceDiagram
    participant Agent as Searcher Agent
    participant Tool as OpenRouter Search
    participant API as OpenRouter API

    Agent->>Tool: search(query)
    Tool->>API: POST /chat/completions
    Note right of API: model: model:online<br/>plugins: [{id: "web"}]
    API-->>Tool: {results: [...], annotations: [...]}
    Tool-->>Agent: Parsed SearchResults
```

**SearchResult Schema:**
```json
{
  "url": "https://...",
  "title": "Article Title",
  "content": "Extracted content...",
  "score": 0.95,
  "metadata": {
    "source": "openrouter",
    "engine": "native"
  }
}
```

---

## Instrumentation & Observability

### Tracing Architecture

```mermaid
flowchart TB
    subgraph Application["Application Layer"]
        RootSpan["Create Root Span<br/>research_session"]

        subgraph AgentExecution["Agent Execution"]
            PlannerSpan["Child Span: planner"]
            LLMSpan1["Child Span: llm_call"]
            SearcherSpan["Child Span: searcher"]
            ToolSpan["Child Span: tool_use"]
        end
    end

    subgraph OTel["OpenTelemetry"]
        Exporter["OTLP Exporter"]
    end

    subgraph Phoenix["Phoenix Collector"]
        Storage["Phoenix Storage<br/>Traces<br/>Spans<br/>Metrics"]
    end

    RootSpan --> AgentExecution
    AgentExecution --> Exporter
    Exporter -->|"OTLP/gRPC"| Storage
```

**Span Hierarchy:**

```mermaid
flowchart TB
    Root["research_session (root)"]

    Planner["planner_agent"]
    PlannerLLM["llm_call (gemini-flash)"]
    PlannerState["state_update"]

    Searcher["searcher_agent"]
    SearcherLLM["llm_call (query expansion)"]
    SearcherTool1["tool_call (web_search)"]
    SearcherTool2["tool_call (rag_retrieve)"]

    Reader0["reader_agent[0]"]
    Reader0Tool["tool_call (html_parse)"]
    Reader0LLM["llm_call (extraction)"]

    Reader1["reader_agent[1]"]

    Synthesizer["synthesizer_agent"]
    SynthesizerLLM["llm_call (synthesis)"]

    Critic["critic_agent"]
    CriticLLM["llm_call (evaluation)"]

    Reporter["reporter_agent"]
    ReporterLLM["llm_call (report generation)"]

    Root --> Planner
    Planner --> PlannerLLM
    Planner --> PlannerState

    Root --> Searcher
    Searcher --> SearcherLLM
    Searcher --> SearcherTool1
    Searcher --> SearcherTool2

    Root --> Reader0
    Reader0 --> Reader0Tool
    Reader0 --> Reader0LLM

    Root --> Reader1
    Root --> Synthesizer
    Synthesizer --> SynthesizerLLM

    Root --> Critic
    Critic --> CriticLLM

    Root --> Reporter
    Reporter --> ReporterLLM
```

### Metrics Collection

```mermaid
flowchart TB
    subgraph RealTimeRedis["Real-time Metrics (Redis)"]
        AgentHealth["drx:agent:{id}:health<br/>status: healthy|degraded<br/>last_check: timestamp<br/>failure_count: 0"]

        RateMetrics["drx:agent:{id}:metrics<br/>tokens_1m: 45000<br/>tokens_5m: 180000<br/>latency_p50: 1200<br/>latency_p99: 4500<br/>error_rate: 0.02"]

        CircuitStatus["drx:agent:{id}:circuit<br/>state: closed<br/>opened_at: null"]

        Invocations["drx:agent:{id}:invocations<br/>(timestamp, {tokens, ms})"]
    end

    subgraph PersistentPostgres["Persistent Metrics (PostgreSQL)"]
        Sessions["research_sessions<br/>tokens_used: 125000<br/>cost_usd: 0.0125<br/>latency_ms: 45000<br/>iteration_count: 3"]

        ToolInvocations["tool_invocations<br/>tool_name: search<br/>latency_ms: 1250<br/>tokens_used: 500<br/>success: true"]

        AgentInvocations["agent_invocations<br/>agent_type: searcher<br/>input_tokens: 2500<br/>output_tokens: 1500<br/>latency_ms: 3200"]

        Violations["policy_violations<br/>violation_type: domain<br/>severity: warning<br/>blocked: true<br/>agent_id: searcher_v1"]
    end
```

---

## Evaluation Pipeline

### Evaluation Workflow

```mermaid
flowchart TB
    subgraph Trigger["CI/CD Pipeline Trigger"]
        GHA["GitHub Actions<br/>on: push [main, develop]<br/>on: pull_request [main]"]
    end

    subgraph Tests["Test Execution"]
        Unit["Unit Tests<br/>pytest tests/unit"]
        Integration["Integration Tests<br/>pytest tests/integ"]
        Eval["Evaluation Tests<br/>pytest ci/eval"]
    end

    subgraph EvalFramework["Evaluation Framework"]
        subgraph DeepEval["DeepEval Metrics"]
            Faithfulness["Faithfulness<br/>Threshold: >= 0.8"]
            AnswerRelevancy["Answer Relevancy<br/>Threshold: >= 0.7"]
            Hallucination["Hallucination<br/>Threshold: <= 0.2"]
        end

        subgraph Ragas["Ragas Metrics"]
            ContextPrecision["Context Precision<br/>Threshold: >= 0.6"]
            ContextRecall["Context Recall<br/>Threshold: >= 0.6"]
            AnswerCorrectness["Answer Correctness<br/>Threshold: >= 0.7"]
        end

        subgraph MetadataCompliance["Metadata Compliance"]
            BudgetCheck["Budget Compliance<br/>spend <= max_budget"]
            DomainCheck["Domain Compliance<br/>no blocked domains"]
            RateLimitCheck["Rate Limit Compliance<br/>within rpm/tpm"]
        end
    end

    subgraph Gate["Gate Decision"]
        HardGates["Hard Gates (block):<br/>Faithfulness >= 0.8<br/>Hallucination <= 0.2<br/>Policy Violations == 0<br/>Task Completion >= 0.7"]
        SoftGates["Soft Gates (warn):<br/>Answer Relevancy >= 0.7<br/>Context Precision >= 0.6"]

        Decision{"All Gates Pass?"}
        DeployAllowed["Deploy Allowed"]
        DeployBlocked["Deploy Blocked"]
    end

    Trigger --> Tests
    Tests --> EvalFramework
    EvalFramework --> Gate
    HardGates --> Decision
    SoftGates --> Decision
    Decision -->|YES| DeployAllowed
    Decision -->|NO| DeployBlocked
```

### Phoenix Dashboard

```mermaid
flowchart TB
    subgraph PhoenixUI["Phoenix UI (http://localhost:6006)"]
        subgraph TraceExplorer["Trace Explorer"]
            SessionInfo["Session: int_abc123<br/>Duration: 45.2s<br/>Tokens: 125,000<br/>Cost: $0.0125"]

            Timeline["Timeline View<br/>planner (8.2s)<br/>searcher (12.1s)<br/>reader[0] (9.5s)<br/>reader[1] (6.2s)<br/>synthesizer (7.8s)<br/>critic (2.1s)<br/>reporter (4.3s)"]
        end

        subgraph EvalResults["Evaluation Results"]
            Metrics["Metric | Score | Threshold<br/>Faithfulness | 0.92 | >= 0.8<br/>Hallucination | 0.08 | <= 0.2<br/>Relevancy | 0.88 | >= 0.7<br/>Completeness | 0.85 | >= 0.7"]
        end
    end
```

---

## Appendix

### Technology Stack Summary

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Language** | Python | 3.11+ | Core implementation |
| **Orchestration** | LangGraph | 1.0+ | Agent workflow DAG |
| **Framework** | FastAPI | 0.115+ | REST API |
| **Database** | PostgreSQL | 16+ | Persistent storage |
| **Vector DB** | pgvector | 0.8+ | Embedding storage |
| **Cache** | Redis | 7+ | State, rate limiting |
| **Queue** | Celery | 5.4+ | Async task execution |
| **LLM Gateway** | OpenRouter | - | Multi-model access |
| **Search** | OpenRouter Native | - | Web search (free) |
| **Observability** | Phoenix | 12+ | Tracing, eval |
| **Telemetry** | OpenTelemetry | 1.28+ | Distributed tracing |
| **Eval** | DeepEval | 1.0+ | LLM evaluation |
| **Eval** | Ragas | 0.1+ | RAG evaluation |
| **Guardrails** | NeMo | 0.19+ | Safety rails |

### Port Allocations

| Service | Port | Protocol |
|---------|------|----------|
| FastAPI | 8000 | HTTP |
| Phoenix | 6006 | HTTP |
| Phoenix Collector | 4317 | gRPC |
| PostgreSQL | 5432 | TCP |
| Redis | 6379 | TCP |
