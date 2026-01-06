# DRX - Deep Research X

**Enterprise-grade, self-hosted multi-agent research system with full observability and deterministic replay.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)

---

## Table of Contents

- [What is DRX?](#what-is-drx)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Development](#development)
- [Evaluation](#evaluation)
- [Roadmap](#roadmap)
- [License](#license)

---

## What is DRX?

DRX (Deep Research X) is a **self-hosted, multi-agent research orchestration platform** that performs comprehensive research tasks by coordinating specialized AI agents in a directed acyclic graph (DAG) workflow. Unlike consumer deep research products, DRX gives you full control over your research infrastructure, data sovereignty, and the ability to customize every aspect of the research pipeline.

### Why DRX Over Consumer Products?

| Feature | DRX | Gemini Deep Research | OpenAI | Perplexity |
|---------|-----|---------------------|--------|------------|
| **Self-Hosted** | Full control | Google Cloud | OpenAI servers | SaaS only |
| **Data Sovereignty** | Your infrastructure | Data sent to Google | Data sent to OpenAI | Data retained |
| **Open Source** | MIT License | Proprietary | Proprietary | Proprietary |
| **Custom Agents** | Add/modify agents | Fixed pipeline | Fixed pipeline | Fixed pipeline |
| **Model Agnostic** | Any LLM via OpenRouter | Gemini only | GPT only | Limited models |
| **Checkpoint/Resume** | Pause & continue | No | No | No |
| **Deterministic Replay** | Debug & reproduce | No | No | No |
| **Full Observability** | Phoenix/OpenTelemetry | Limited | Limited | None |

---

## Key Features

### Multi-Agent DAG Architecture

DRX uses a transparent, auditable pipeline of six specialized agents:

| Agent | Role | Description |
|-------|------|-------------|
| **Planner** | Query Decomposition | Breaks complex queries into research sub-tasks with dependencies |
| **Searcher** | Source Discovery | Discovers sources via web search (OpenRouter native or Tavily) |
| **Reader** | Content Extraction | Extracts and structures information from sources |
| **Synthesizer** | Information Synthesis | Aggregates findings, resolves conflicts, builds argument graphs |
| **Critic** | Quality Assurance | Reviews quality, identifies gaps, triggers re-research iterations |
| **Reporter** | Report Generation | Generates final deliverables with citations and references |

### Iterative Refinement Loop

The Critic agent analyzes research quality after each iteration and identifies knowledge gaps. Research continues until:
- Coverage score meets the threshold (default: 0.7)
- Maximum iterations reached (default: 5)
- Token budget exhausted

### Enterprise Features

- **Checkpoint & Resume**: Pause long research tasks and resume from any checkpoint
- **Deterministic Replay**: Debug and reproduce any research run for auditing
- **Circuit Breakers**: Automatic failover when agents or services degrade
- **Policy Firewall**: Enforce domain restrictions, rate limits, and token budgets
- **Full Observability**: Trace every LLM call with Phoenix/OpenTelemetry

---

## Architecture

### Research Workflow

```mermaid
flowchart TD
    subgraph Input
        A[User Query] --> B[FastAPI]
    end

    subgraph Orchestrator["LangGraph Orchestrator"]
        B --> C[Planner]
        C --> D[Searcher]
        D --> E[Reader]
        E --> F[Synthesizer]
        F --> G[Critic]
        G --> H{Gaps Found?}
        H -->|Yes| I[Continue Loop]
        I --> D
        H -->|No| J[Reporter]
        J --> K[Final Report]
    end

    subgraph SharedState["Shared State (Redis)"]
        L[(interaction:id)]
        M[(events:id)]
        N[(rate_limits)]
    end

    C <--> L
    D <--> L
    E <--> L
    F <--> L
    G <--> L
    J <--> L
    B <--> M

    style SharedState fill:#f9f,stroke:#333,stroke-width:2px
    style Orchestrator fill:#e1f5fe,stroke:#333,stroke-width:2px
```

### Infrastructure Architecture

```mermaid
flowchart TB
    subgraph Client
        U[User/Browser]
    end

    subgraph API["FastAPI Server :8000"]
        R[REST Endpoints]
        S[SSE Streaming]
    end

    subgraph Orchestration["LangGraph Workflow"]
        W[ResearchOrchestrator]
        A1[Planner Agent]
        A2[Searcher Agent]
        A3[Reader Agent]
        A4[Synthesizer Agent]
        A5[Critic Agent]
        A6[Reporter Agent]
    end

    subgraph Storage
        PG[(PostgreSQL 16<br/>+ pgvector)]
        RD[(Valkey/Redis 7)]
    end

    subgraph External
        OR[OpenRouter API<br/>100+ LLMs]
        TV[Tavily Search<br/>Optional]
    end

    subgraph Observability
        PH[Phoenix :6006<br/>LLM Tracing]
    end

    U <-->|HTTP/SSE| R
    U <-->|Events| S
    R --> W
    S <-->|Pub/Sub| RD
    W --> A1 & A2 & A3 & A4 & A5 & A6
    A1 & A2 & A3 & A4 & A5 & A6 -->|LLM Calls| OR
    A2 -->|Web Search| TV
    W <-->|Checkpoints| PG
    W <-->|State| RD
    A1 & A2 & A3 & A4 & A5 & A6 -->|Traces| PH

    style Storage fill:#e8f5e9,stroke:#333
    style Observability fill:#fff3e0,stroke:#333
    style External fill:#fce4ec,stroke:#333
```

### Redis Shared State Model

```mermaid
erDiagram
    INTERACTION_HASH {
        string id PK "UUID"
        string status "queued|running|completed|failed"
        string query "Research question"
        json result "Final report + findings"
        string error "Error message if failed"
        timestamp created_at "ISO 8601"
    }

    EVENT_CHANNEL {
        string interaction_id FK
        string event_type "start|delta|complete|error"
        json payload "Event data"
    }

    RATE_LIMIT_ZSET {
        string identifier PK "user:id or ip:addr"
        float timestamp "Request timestamp"
        int cost "Request cost"
    }

    CIRCUIT_BREAKER {
        string agent_id PK
        string state "closed|open|half_open"
        int failure_count
        timestamp last_failure
    }

    INTERACTION_HASH ||--o{ EVENT_CHANNEL : "publishes"
```

### Infrastructure Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Database** | PostgreSQL 16 + pgvector | Session state, checkpoints, vector embeddings |
| **Cache/Queue** | Valkey 7 (Redis-compatible) | Real-time state, pub/sub, Celery broker |
| **Observability** | Arize Phoenix | LLM tracing, token monitoring, latency tracking |
| **API** | FastAPI + SSE | REST endpoints with real-time streaming |
| **Orchestration** | LangGraph | DAG workflow execution with checkpointing |
| **LLM Gateway** | OpenRouter | Access to 100+ models (Gemini, Claude, GPT, DeepSeek) |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose v2+
- OpenRouter API key ([Get one here](https://openrouter.ai/keys))
- (Optional) Tavily API key for enhanced web search

### Step 1: Clone and Configure

```bash
# Clone the repository
git clone https://github.com/your-org/drx.git
cd drx/v1

# Create environment file
cp .env.example .env

# Edit .env and add your API key
# OPENROUTER_API_KEY=sk-or-v1-...
```

### Step 2: Launch Services

```bash
cd deployment

# Start all services
docker compose up -d

# Verify services are healthy
docker compose ps

# View logs
docker compose logs -f api
```

**Services Started:**
| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | FastAPI REST endpoints |
| Phoenix | http://localhost:6006 | Observability dashboard |
| PostgreSQL | localhost:5432 | Database |
| Redis | localhost:6379 | Cache & queue |

### Step 3: Run Your First Research

```bash
# Create a research interaction
curl -X POST http://localhost:8000/api/v1/interactions \
  -H "Content-Type: application/json" \
  -d '{"input": "What are the latest advancements in quantum computing and their impact on cryptography?"}'
```

**Response:**
```json
{
  "id": "bee2698d-7146-47f8-810d-2ac6bd4f6b8c",
  "status": "queued",
  "created_at": "2026-01-05T19:53:34.405942Z",
  "query": "What are the latest advancements in quantum computing...",
  "result": null
}
```

### Step 4: Get Results

```bash
# Poll for results (typically 30-60 seconds)
curl http://localhost:8000/api/v1/interactions/{id}

# Or stream real-time progress
curl -N http://localhost:8000/api/v1/interactions/{id}/stream
```

**Completed Response:**
```json
{
  "id": "bee2698d-7146-47f8-810d-2ac6bd4f6b8c",
  "status": "completed",
  "result": {
    "final_report": "# Research Report: Quantum Computing...",
    "findings": [...],
    "citations": [...],
    "tokens_used": 696875,
    "iteration_count": 5
  }
}
```

---

## API Reference

### Core Endpoints

#### Create Research Interaction

```http
POST /api/v1/interactions
Content-Type: application/json

{
  "input": "Your research question (10-10000 chars)",
  "steerability": {
    "tone": "technical",
    "format": "markdown",
    "max_sources": 20,
    "focus_areas": ["security", "performance"],
    "exclude_topics": ["marketing"],
    "preferred_domains": ["arxiv.org", "*.edu"],
    "language": "en",
    "custom_instructions": "Focus on peer-reviewed sources"
  },
  "config": {
    "max_iterations": 5,
    "token_budget": 500000,
    "timeout_seconds": 600,
    "enable_citations": true,
    "enable_quality_checks": true
  }
}
```

**Response:** `202 Accepted`
```json
{
  "id": "uuid",
  "status": "queued",
  "created_at": "ISO8601",
  "query": "..."
}
```

#### Get Interaction Status

```http
GET /api/v1/interactions/{interaction_id}
```

#### List Interactions

```http
GET /api/v1/interactions?page=1&page_size=20&status=completed
```

#### Stream Progress (SSE)

```http
GET /api/v1/interactions/{interaction_id}/stream
Accept: text/event-stream
```

**Event Types:**
- `interaction.start` - Research initiated
- `thought_summary` - Agent thinking updates
- `content.delta` - Incremental content
- `tool.use` - Tool invocations
- `interaction.complete` - Research finished
- `interaction.error` - Error occurred

#### Cancel Interaction

```http
DELETE /api/v1/interactions/{interaction_id}
```

#### Resume from Checkpoint

```http
POST /api/v1/interactions/{interaction_id}/resume
Content-Type: application/json

{
  "checkpoint_id": "chk_abc123"
}
```

### Replay Endpoints

#### Start Replay

```http
POST /api/v1/interactions/{interaction_id}/replay
Content-Type: application/json

{
  "from_checkpoint": "chk_abc123",
  "speed_multiplier": 1.0,
  "modifications": {
    "max_iterations": 3
  }
}
```

#### Get Recorded Events

```http
GET /api/v1/interactions/{interaction_id}/events?limit=100&offset=0
```

#### Compare Original vs Replay

```http
POST /api/v1/interactions/{interaction_id}/compare
Content-Type: application/json

{
  "replay_session_id": "replay_xyz"
}
```

### Health Check

```http
GET /api/v1/health
```

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "ISO8601",
  "checks": {
    "database": "healthy",
    "redis": "healthy"
  }
}
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# ============================================================================
# REQUIRED
# ============================================================================

# OpenRouter API Key (get from https://openrouter.ai/keys)
OPENROUTER_API_KEY=sk-or-v1-...

# ============================================================================
# DATABASE (PostgreSQL)
# ============================================================================

DATABASE_URL=postgresql://drx:drx_password@localhost:5432/drx
DB_POOL_SIZE=10
DB_POOL_MIN_SIZE=5
DB_POOL_MAX_SIZE=20

# ============================================================================
# CACHE (Redis/Valkey)
# ============================================================================

REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=50

# ============================================================================
# LLM MODELS
# ============================================================================

# Default model for general tasks
DEFAULT_MODEL=google/gemini-3-flash-preview

# Model for complex reasoning
REASONING_MODEL=deepseek/deepseek-r1

# Model for synthesis tasks
SYNTHESIS_MODEL=anthropic/claude-3.5-sonnet

# Temperature for generation (0.0-2.0)
DEFAULT_TEMPERATURE=0.7

# Max tokens per request
MAX_TOKENS_PER_REQUEST=100000

# ============================================================================
# WEB SEARCH
# ============================================================================

# Search engine: native (OpenRouter :online), exa, or auto
SEARCH_ENGINE=native

# Model for search (append :online for native search)
SEARCH_MODEL=google/gemini-3-flash-preview:online

# Max results per search query
SEARCH_MAX_RESULTS=5

# ============================================================================
# TAVILY (Optional Fallback)
# ============================================================================

TAVILY_ENABLED=false
TAVILY_API_KEY=tvly-...
TAVILY_SEARCH_DEPTH=advanced
TAVILY_MAX_RESULTS=10

# ============================================================================
# RESEARCH DEFAULTS
# ============================================================================

# Maximum research iterations (1-10)
MAX_RESEARCH_ITERATIONS=5

# Token budget per session (10k-2M)
TOKEN_BUDGET_PER_SESSION=500000

# Max sources to retrieve per query
MAX_SOURCES_PER_QUERY=20

# Minimum coverage score to complete (0.0-1.0)
MIN_COVERAGE_SCORE=0.7

# Research timeout in seconds
DEFAULT_RESEARCH_TIMEOUT=600

# ============================================================================
# RATE LIMITING
# ============================================================================

RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_TOKENS_PER_MINUTE=1000000

# ============================================================================
# OBSERVABILITY (Phoenix)
# ============================================================================

PHOENIX_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317
PHOENIX_PROJECT_NAME=drx-research

# ============================================================================
# APPLICATION
# ============================================================================

APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# CORS allowed origins (comma-separated)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
```

### Steerability Options

Control research output through the `steerability` parameter:

| Option | Values | Description |
|--------|--------|-------------|
| `tone` | `executive`, `technical`, `casual` | Writing style of the report |
| `format` | `markdown`, `markdown_table`, `json` | Output format |
| `max_sources` | `1-100` | Maximum sources to cite |
| `focus_areas` | `string[]` | Topics to emphasize |
| `exclude_topics` | `string[]` | Topics to avoid |
| `preferred_domains` | `string[]` | Preferred source domains |
| `language` | `string` | Output language code |
| `custom_instructions` | `string` | Additional guidance |

---

## Project Structure

```
v1/
├── src/
│   ├── agents/                 # Specialized research agents
│   │   ├── base.py             # BaseAgent class & LLM client adapter
│   │   ├── planner.py          # Query decomposition & DAG planning
│   │   ├── searcher.py         # Web search & source discovery
│   │   ├── reader.py           # Content extraction & structuring
│   │   ├── synthesizer.py      # Information synthesis
│   │   ├── critic.py           # Quality assessment & gap analysis
│   │   └── reporter.py         # Report generation
│   │
│   ├── orchestrator/           # LangGraph workflow
│   │   ├── workflow.py         # ResearchOrchestrator & DAG execution
│   │   ├── state.py            # AgentState TypedDict definitions
│   │   ├── nodes.py            # Node functions & agent registry
│   │   └── checkpointer.py     # AsyncPostgresSaver integration
│   │
│   ├── api/                    # FastAPI REST API
│   │   ├── main.py             # Application factory & lifespan
│   │   ├── routes.py           # Core research endpoints
│   │   ├── replay_routes.py    # Replay & debugging endpoints
│   │   ├── dependencies.py     # DI: DB, Redis, Orchestrator
│   │   └── models.py           # Pydantic request/response models
│   │
│   ├── services/               # External service clients
│   │   ├── openrouter_client.py# LLM API with retry & streaming
│   │   └── active_state.py     # Real-time metrics
│   │
│   ├── metadata/               # Agentic metadata infrastructure
│   │   ├── manifest.py         # Agent manifest schema
│   │   ├── routing.py          # Capability-based routing
│   │   ├── circuit_breaker.py  # Fault tolerance
│   │   └── context.py          # Context propagation
│   │
│   ├── middleware/             # Policy enforcement
│   │   ├── policy_firewall.py  # Domain, budget, rate limits
│   │   └── domain_validator.py # URL validation
│   │
│   ├── replay/                 # Deterministic replay
│   │   ├── recorder.py         # Event recording
│   │   └── player.py           # Replay execution
│   │
│   ├── tools/                  # Agent tools
│   │   ├── tavily_search.py    # Tavily web search
│   │   ├── openrouter_search.py# Native OpenRouter search
│   │   └── rag_retriever.py    # Vector retrieval
│   │
│   ├── db/                     # Database utilities
│   │   └── connection.py       # Async connection pool
│   │
│   ├── config.py               # Pydantic settings
│   └── worker.py               # Celery worker
│
├── schemas/                    # Data definitions
│   ├── postgres_schema.sql     # Database DDL
│   └── agent_manifest.json     # Agent manifest JSON Schema
│
├── deployment/                 # Docker & infrastructure
│   ├── docker-compose.yaml     # Full stack definition
│   ├── Dockerfile.api          # API server image
│   ├── Dockerfile.worker       # Celery worker image
│   └── init.sql                # Database initialization
│
├── ci/evaluation/              # Evaluation pipeline
│   ├── test_agent_evals.py     # DeepEval tests
│   └── conftest.py             # Pytest fixtures
│
├── pyproject.toml              # Dependencies & build config
├── .env.example                # Environment template
└── README.md                   # This file
```

---

## Development

### Local Development Setup

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev,eval]"

# Start infrastructure only
cd deployment
docker compose up -d postgres redis phoenix

# Initialize database
docker exec -i drx-postgres psql -U drx -d drx < ../schemas/postgres_schema.sql

# Run API server with hot reload
cd ..
uvicorn src.api.main:app --reload --port 8000
```

### Code Quality

```bash
# Linting
ruff check .
ruff check . --fix

# Type checking
mypy src/

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Adding a New Agent

1. Create `src/agents/my_agent.py`:

```python
from .base import BaseAgent, AgentResponse

class MyAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "my_agent"

    @property
    def description(self) -> str:
        return "Description of what this agent does"

    @property
    def agent_type(self) -> str:
        return "my_agent"

    @property
    def system_prompt(self) -> str:
        return "You are an expert at..."

    async def _process(self, state: AgentState) -> AgentResponse:
        # Your agent logic here
        result = await self._call_llm(user_message)
        return AgentResponse.success_response(
            data=result.data,
            agent_name=self.name,
            tokens_used=result.tokens_used
        )
```

2. Register in `src/orchestrator/nodes.py`:

```python
elif agent_type == "my_agent":
    from src.agents.my_agent import MyAgent
    _agents[agent_type] = MyAgent(llm_client=llm_client)
```

3. Add to workflow in `src/orchestrator/workflow.py`

---

## Evaluation

DRX includes an evaluation pipeline using DeepEval and Ragas:

```bash
# Run all evaluations
pytest ci/evaluation/ -v

# Run CI gate tests only
pytest ci/evaluation/ -m ci_gate -v

# Run with detailed output
pytest ci/evaluation/ -v --tb=long
```

### Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Faithfulness** | >= 0.8 | Claims supported by sources |
| **Task Completion** | >= 0.7 | Query fully addressed |
| **Hallucination Rate** | <= 0.2 | Unsupported claims |
| **Policy Violations** | == 0 | Domain/budget violations |
| **Coverage Score** | >= 0.7 | Research completeness |

---

## Troubleshooting

### Common Issues

**1. Database connection errors**
```bash
# Check PostgreSQL is running
docker compose ps postgres
docker compose logs postgres

# Verify connection
docker exec -it drx-postgres psql -U drx -d drx -c "SELECT 1"
```

**2. Redis connection errors**
```bash
# Check Valkey is running
docker compose ps redis
docker exec -it drx-redis valkey-cli ping
```

**3. OpenRouter API errors**
```bash
# Verify API key
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

**4. Workflow not executing**
```bash
# Check API logs
docker compose logs api -f

# Verify orchestrator initialized
curl http://localhost:8000/api/v1/health
```

### Useful Commands

```bash
# Restart all services
docker compose restart

# Rebuild after code changes
docker compose build api --no-cache
docker compose up -d api

# View real-time logs
docker compose logs -f api

# Access PostgreSQL
docker exec -it drx-postgres psql -U drx -d drx

# Access Redis CLI
docker exec -it drx-redis valkey-cli

# Check Phoenix traces
open http://localhost:6006
```

---

## Roadmap

- [ ] **v1.1**: Enhanced Tavily integration with fallback chain
- [ ] **v1.2**: Interactive DAG editing UI
- [ ] **v1.3**: Multi-modal research (images, PDFs)
- [ ] **v1.4**: RLHF fine-tuning pipeline export
- [ ] **v2.0**: Hierarchical multi-agent orchestration

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting PRs.

---

## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [Phoenix](https://github.com/Arize-ai/phoenix) - LLM observability
- [DeepEval](https://github.com/confident-ai/deepeval) - Evaluation framework
- [OpenRouter](https://openrouter.ai) - LLM gateway
- [FastAPI](https://fastapi.tiangolo.com) - API framework
- [Pydantic](https://docs.pydantic.dev) - Data validation
