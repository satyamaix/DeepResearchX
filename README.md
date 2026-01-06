# DRX - Deep Research X

**Enterprise-grade, self-hosted multi-agent research system with full observability, deterministic replay, and continuous learning.**

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
- [Frontend Interface](#frontend-interface)
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
| **Full Observability** | Phoenix/Prometheus/Grafana | Limited | Limited | None |
| **Knowledge Graph** | Entity-Relation visualization | No | No | No |
| **Dataset Flywheel** | Continuous learning | No | No | No |

---

## Key Features

### Multi-Agent DAG Architecture

DRX uses a transparent, auditable pipeline of six specialized agents with parallel execution:

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        Query["User Query"]
    end

    subgraph Orchestration["LangGraph Orchestrator"]
        Planner["Planner Agent"]

        subgraph ParallelSearch["Parallel Fan-Out"]
            S1["Searcher 1"]
            S2["Searcher 2"]
            S3["Searcher N"]
        end

        subgraph ParallelRead["Parallel Fan-Out"]
            R1["Reader 1"]
            R2["Reader 2"]
            R3["Reader N"]
        end

        Synthesizer["Synthesizer Agent"]
        Critic["Critic Agent"]
        Reporter["Reporter Agent"]
    end

    subgraph Output["Output Layer"]
        Report["Final Report"]
        KG["Knowledge Graph"]
        Export["HTML/PDF Export"]
    end

    Query --> Planner
    Planner --> ParallelSearch
    ParallelSearch --> ParallelRead
    ParallelRead --> Synthesizer
    Synthesizer --> Critic
    Critic -->|"Gaps"| Planner
    Critic -->|"Complete"| Reporter
    Reporter --> Report & KG & Export
```

| Agent | Role | Key Tools |
|-------|------|-----------|
| **Planner** | Query decomposition & DAG planning | - |
| **Searcher** | Source discovery via web search | OpenRouter Native, Tavily, RAG |
| **Reader** | Content extraction & structuring | PDF Extractor, HTML Parser |
| **Synthesizer** | Information synthesis & conflict resolution | Knowledge Graph Builder |
| **Critic** | Quality assurance & gap analysis | Citation Verifier, Bias Detector |
| **Reporter** | Report generation & export | HTML/PDF Exporter |

### Enterprise Features

- **Parallel DAG Execution**: Fan-out/fan-in patterns for concurrent task processing
- **Knowledge Graph**: Entity-Relation-Claim visualization with Cytoscape.js
- **Checkpoint & Resume**: Pause long research tasks and resume from any checkpoint
- **Deterministic Replay**: Debug and reproduce any research run for auditing
- **Circuit Breakers**: Automatic failover when agents or services degrade
- **Policy Firewall**: Enforce domain restrictions, rate limits, and token budgets
- **Budget Tracking**: Real-time token/cost monitoring with configurable limits
- **Full Observability**: Phoenix traces, Prometheus metrics, Grafana dashboards
- **Citation Verification**: URL accessibility and quote fuzzy matching
- **Bias Detection**: Source diversity analysis and bias indicators
- **Dataset Flywheel**: Continuous learning from user feedback
- **Multi-format Export**: Markdown, HTML, PDF, JSON reports

---

## Architecture

### System Overview

```mermaid
flowchart TB
    subgraph Clients["Client Layer"]
        WebUI["Web UI"]
        API_Client["REST/SSE Client"]
    end

    subgraph Gateway["API Gateway"]
        FastAPI["FastAPI :8000"]
        SSE["SSE Streaming"]
        Feedback["Feedback Endpoint"]
    end

    subgraph Workers["Worker Layer"]
        Celery["Celery Workers"]
        Progress["Progress Publisher"]
    end

    subgraph Core["Core Orchestration"]
        LangGraph["LangGraph StateGraph"]
        ParallelExec["Parallel Executor"]
        Budget["Budget Tracker"]
    end

    subgraph Agents["Agent Pool"]
        Planner["Planner"]
        Searcher["Searcher"]
        Reader["Reader"]
        Synthesizer["Synthesizer"]
        Critic["Critic"]
        Reporter["Reporter"]
    end

    subgraph Tools["Tool Layer"]
        Search["Web Search"]
        PDF["PDF Extractor"]
        RAG["RAG Retriever"]
        Citation["Citation Verifier"]
        Bias["Bias Detector"]
    end

    subgraph Knowledge["Knowledge Layer"]
        KGraph["Knowledge Graph"]
        VectorStore["Vector Store"]
        ReportExport["Report Exporter"]
    end

    subgraph Storage["Data Layer"]
        Postgres[("PostgreSQL + pgvector")]
        Redis[("Valkey/Redis")]
    end

    subgraph External["External Services"]
        OpenRouter["OpenRouter API"]
        Tavily["Tavily Search"]
    end

    subgraph Observability["Observability"]
        Phoenix["Phoenix :6006"]
        Prometheus["Prometheus"]
        Grafana["Grafana :3000"]
    end

    subgraph Flywheel["Dataset Flywheel"]
        Collector["Dataset Collector"]
        FeedbackStore["Feedback Store"]
    end

    Clients --> Gateway
    Gateway --> Workers
    Workers --> Core
    Core --> Agents
    Agents --> Tools
    Tools --> Knowledge
    Knowledge --> Storage
    Agents --> External
    Core --> Observability
    Gateway --> Flywheel
    Flywheel --> Storage
```

### Infrastructure Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Database** | PostgreSQL 16 + pgvector | Session state, checkpoints, vector embeddings |
| **Cache/Queue** | Valkey 7 (Redis-compatible) | Real-time state, pub/sub, Celery broker |
| **API** | FastAPI + SSE | REST endpoints with real-time streaming |
| **Workers** | Celery | Async task execution with progress streaming |
| **Orchestration** | LangGraph | DAG workflow execution with checkpointing |
| **LLM Gateway** | OpenRouter | Access to 100+ models (Gemini, Claude, GPT, DeepSeek) |
| **Observability** | Phoenix, Prometheus, Grafana | LLM tracing, metrics, dashboards |
| **Frontend** | Vanilla JS + D3.js + Cytoscape.js | DAG & Knowledge Graph visualization |

### Data Flow

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as FastAPI
    participant W as Celery Worker
    participant O as Orchestrator
    participant A as Agents
    participant LLM as OpenRouter
    participant DB as PostgreSQL
    participant R as Redis

    C->>API: POST /interactions
    API->>DB: INSERT session
    API->>W: Enqueue task
    API-->>C: 202 Accepted {id}

    C->>API: GET /stream
    API->>R: SUBSCRIBE events:{id}

    rect rgb(240, 248, 255)
        Note over W,LLM: Async Worker Execution
        W->>O: Start orchestration
        O->>DB: Load checkpoint

        loop For each DAG level
            O->>A: Execute parallel tasks
            A->>LLM: Generate completions
            LLM-->>A: Responses
            A->>R: PUBLISH progress events
            R-->>API: Forward events
            API-->>C: SSE: progress
            A-->>O: Updated state
            O->>DB: Save checkpoint
        end

        O->>DB: UPDATE status=completed
        O->>R: PUBLISH complete
    end

    R-->>API: Complete event
    API-->>C: SSE: complete
    C->>API: GET /interactions/{id}
    API->>DB: SELECT result
    API-->>C: Final report + knowledge graph
```

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
| Frontend | http://localhost:8000 | Research UI |
| Phoenix | http://localhost:6006 | LLM Observability |
| Grafana | http://localhost:3000 | Metrics Dashboards |
| PostgreSQL | localhost:5432 | Database |
| Redis | localhost:6379 | Cache & Queue |

### Step 3: Open the Frontend

Navigate to http://localhost:8000 to access the research interface with:
- Real-time DAG visualization
- Knowledge graph explorer
- Research history
- Feedback collection

### Step 4: Run Your First Research (API)

```bash
# Create a research interaction
curl -X POST http://localhost:8000/api/v1/interactions \
  -H "Content-Type: application/json" \
  -d '{"input": "What are the latest advancements in quantum computing?"}'
```

---

## Frontend Interface

### DAG Visualization

Real-time directed acyclic graph showing research workflow execution:

```mermaid
flowchart LR
    subgraph DAGViz["DAG Visualization Component"]
        direction TB
        D3["D3.js Rendering"]
        Nodes["Node States"]
        Edges["Edge Animation"]
        Controls["Zoom/Pan/Export"]
    end

    subgraph States["Node States"]
        Pending["Pending (gray)"]
        Running["Running (blue pulse)"]
        Complete["Completed (green)"]
        Failed["Failed (red)"]
    end

    subgraph SSE["Real-time Updates"]
        Events["SSE Events"]
        Update["State Updates"]
    end

    SSE --> DAGViz
    DAGViz --> States
```

### Knowledge Graph

Interactive argument graph showing entities, relations, and claims:

```mermaid
flowchart TB
    subgraph GraphViz["Knowledge Graph Component"]
        Cytoscape["Cytoscape.js"]
        Filter["Entity Type Filters"]
        Detail["Detail Panel"]
        Tooltip["Citation Hovercards"]
    end

    subgraph Entities["Entity Types"]
        Person["Person (blue)"]
        Org["Organization (green)"]
        Concept["Concept (purple)"]
        Claim["Claim (orange)"]
    end

    subgraph Claims["Claim Status"]
        Supported["Supported (green)"]
        Contested["Contested (yellow)"]
        Refuted["Refuted (red)"]
    end

    GraphViz --> Entities
    GraphViz --> Claims
```

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/interactions` | Create research interaction |
| `GET` | `/api/v1/interactions` | List interactions |
| `GET` | `/api/v1/interactions/{id}` | Get interaction details |
| `GET` | `/api/v1/interactions/{id}/stream` | SSE progress stream |
| `DELETE` | `/api/v1/interactions/{id}` | Cancel interaction |
| `POST` | `/api/v1/interactions/{id}/resume` | Resume from checkpoint |
| `POST` | `/api/v1/interactions/{id}/feedback` | Submit feedback |
| `GET` | `/api/v1/interactions/{id}/feedback` | Get feedback |

### Create Interaction

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
    "language": "en"
  },
  "config": {
    "max_iterations": 5,
    "token_budget": 500000,
    "timeout_seconds": 600
  }
}
```

### Submit Feedback

```http
POST /api/v1/interactions/{id}/feedback
Content-Type: application/json

{
  "rating": 5,
  "comment": "Comprehensive and well-cited",
  "labels": ["accurate", "comprehensive", "well-structured"]
}
```

### SSE Event Types

```mermaid
flowchart LR
    Start["interaction.start"] --> Thought["thought_summary"]
    Thought --> Tool["tool.use"]
    Tool --> Content["content.delta"]
    Content --> DAG["dag_state"]
    DAG --> Checkpoint["checkpoint"]
    Checkpoint --> Thought
    Thought --> Complete["interaction.complete"]
```

---

## Configuration

### Environment Variables

```env
# ============================================================================
# REQUIRED
# ============================================================================
OPENROUTER_API_KEY=sk-or-v1-...

# ============================================================================
# DATABASE
# ============================================================================
DATABASE_URL=postgresql://drx:drx_password@localhost:5432/drx

# ============================================================================
# CACHE
# ============================================================================
REDIS_URL=redis://localhost:6379/0

# ============================================================================
# LLM MODELS
# ============================================================================
DEFAULT_MODEL=google/gemini-3-flash-preview
REASONING_MODEL=deepseek/deepseek-r1
SEARCH_MODEL=google/gemini-3-flash-preview:online

# ============================================================================
# BUDGET TRACKING
# ============================================================================
TOKEN_BUDGET_PER_SESSION=500000
COST_BUDGET_PER_SESSION=1.00

# ============================================================================
# RESEARCH DEFAULTS
# ============================================================================
MAX_RESEARCH_ITERATIONS=5
MAX_SOURCES_PER_QUERY=20
MIN_COVERAGE_SCORE=0.7

# ============================================================================
# OBSERVABILITY
# ============================================================================
PHOENIX_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317
PROMETHEUS_ENABLED=true
```

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
│   │   ├── parallel.py         # DAG parallel execution engine
│   │   ├── budget.py           # Token/cost budget tracking
│   │   └── checkpointer.py     # AsyncPostgresSaver integration
│   │
│   ├── models/                 # Data models
│   │   ├── __init__.py         # Package exports
│   │   └── knowledge.py        # Knowledge Graph (Entity, Relation, Claim)
│   │
│   ├── api/                    # FastAPI REST API
│   │   ├── main.py             # Application factory & lifespan
│   │   ├── routes.py           # Core research endpoints + feedback
│   │   ├── replay_routes.py    # Replay & debugging endpoints
│   │   └── dependencies.py     # DI: DB, Redis, Orchestrator
│   │
│   ├── services/               # Service layer
│   │   ├── openrouter_client.py# LLM API with retry & streaming
│   │   ├── vectorstore.py      # pgvector-backed vector store
│   │   ├── report_exporter.py  # HTML/PDF report generation
│   │   └── progress_publisher.py# Redis pub/sub progress streaming
│   │
│   ├── tools/                  # Agent tools
│   │   ├── base.py             # BaseTool interface
│   │   ├── openrouter_search.py# Native OpenRouter search
│   │   ├── tavily_search.py    # Tavily web search
│   │   ├── rag_retriever.py    # Vector retrieval
│   │   ├── pdf_extractor.py    # PDF extraction (pypdf)
│   │   ├── citation_verifier.py# Citation verification
│   │   └── bias_detector.py    # Bias detection & diversity analysis
│   │
│   ├── observability/          # Observability
│   │   ├── phoenix.py          # Phoenix/OpenTelemetry setup
│   │   └── metrics.py          # Prometheus metrics
│   │
│   ├── metadata/               # Agentic metadata infrastructure
│   │   ├── manifest.py         # Agent manifest schema
│   │   ├── routing.py          # Capability-based routing
│   │   ├── circuit_breaker.py  # Fault tolerance
│   │   └── context.py          # Context propagation
│   │
│   ├── config.py               # Pydantic settings
│   └── worker.py               # Celery worker
│
├── frontend/                   # Web UI
│   ├── index.html              # Main HTML
│   ├── js/
│   │   ├── main.js             # Application entry
│   │   ├── api.js              # API client
│   │   ├── sse.js              # SSE handler
│   │   ├── dag.js              # D3.js DAG visualization
│   │   ├── graph.js            # Cytoscape.js knowledge graph
│   │   └── state.js            # State management
│   └── css/
│       ├── styles.css          # Main styles
│       ├── dag.css             # DAG visualization styles
│       └── graph.css           # Knowledge graph styles
│
├── ci/evaluation/              # Evaluation & dataset flywheel
│   ├── conftest.py             # Pytest fixtures
│   ├── dataset_collector.py    # Training data collection
│   ├── feedback_store.py       # User feedback storage
│   ├── test_agent_evals.py     # DeepEval tests
│   └── metadata_assertions.py  # Metadata compliance
│
├── deployment/                 # Docker & infrastructure
│   ├── docker-compose.yaml     # Development stack
│   ├── docker-compose.prod.yaml# Production stack
│   ├── Dockerfile.api          # API server image
│   ├── Dockerfile.worker       # Celery worker image
│   ├── init.sql                # Database initialization
│   └── grafana/                # Grafana dashboards
│       ├── dashboards/
│       │   ├── drx-overview.json
│       │   └── drx-agents.json
│       └── provisioning/
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md         # System architecture
│   └── LLD.md                  # Low-level design
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

# Run API server with hot reload
cd ..
uvicorn src.api.main:app --reload --port 8000
```

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy src/

# Run tests
pytest tests/ -v

# Run evaluations
pytest ci/evaluation/ -v
```

---

## Observability

DRX provides comprehensive observability through Arize Phoenix for LLM tracing and Prometheus/Grafana for metrics.

### Arize Phoenix Tracing

```mermaid
flowchart TB
    subgraph Application["DRX Application"]
        subgraph Spans["Trace Spans"]
            Root["research_session<br/>(root span)"]
            Planner["planner_agent"]
            Searcher["searcher_agent"]
            Reader["reader_agent"]
            Synthesizer["synthesizer_agent"]
            Critic["critic_agent"]
            Reporter["reporter_agent"]
        end

        subgraph LLMSpans["LLM Call Spans"]
            LLM1["llm_call (prompt, completion)"]
            LLM2["token counts, latency"]
            LLM3["model, temperature"]
        end
    end

    subgraph Phoenix["Phoenix Collector :6006"]
        OTLP["OTLP Receiver :4317"]
        Storage["Trace Storage"]
        UI["Web UI Dashboard"]
        Analysis["Token Analysis"]
    end

    Root --> Planner & Searcher & Reader & Synthesizer & Critic & Reporter
    Planner & Searcher --> LLMSpans
    LLMSpans -->|"OTLP/gRPC"| OTLP
    OTLP --> Storage --> UI & Analysis
```

### Key Observability Features

| Feature | Tool | Purpose |
|---------|------|---------|
| **LLM Tracing** | Phoenix | Track prompts, completions, tokens per agent |
| **Latency Analysis** | Phoenix | Identify slow agents and LLM calls |
| **Cost Tracking** | Phoenix + Custom | Real-time token/cost per session |
| **Error Tracing** | Phoenix | Distributed trace debugging |
| **Metrics** | Prometheus | Time-series performance metrics |
| **Dashboards** | Grafana | Visual monitoring & alerting |

### Accessing Observability Tools

```bash
# Phoenix UI - LLM Traces
open http://localhost:6006

# Grafana Dashboards
open http://localhost:3000

# Prometheus Metrics
curl http://localhost:8000/metrics
```

---

## Evaluation

DRX includes a comprehensive evaluation pipeline using industry-standard frameworks:

### Evaluation Frameworks

```mermaid
flowchart TB
    subgraph Evaluation["Evaluation Pipeline"]
        Runner["Evaluation Runner<br/>ci/evaluation/run_evaluation.py"]
        Cases["Test Cases<br/>curated_test_cases.yaml"]
    end

    subgraph Frameworks["Evaluation Frameworks"]
        subgraph DeepEval["DeepEval v3.7+"]
            Faith["FaithfulnessMetric"]
            Halluc["HallucinationMetric"]
            Relevancy["AnswerRelevancyMetric"]
        end

        subgraph Ragas["Ragas v0.4+"]
            CtxPrec["context_precision"]
            CtxRecall["context_recall"]
        end

        subgraph Custom["Custom Metrics"]
            TaskComp["task_completion"]
            Policy["policy_compliance"]
        end
    end

    subgraph Output["Outputs"]
        Results["eval_results.json"]
        Metrics["metrics_results.json"]
        Report["EVALUATION_REPORT.md"]
    end

    Cases --> Runner
    Runner --> Frameworks
    Frameworks --> Output
```

### Evaluation Metrics

#### Hard Gates (Must Pass)

| Metric | Framework | Target | Description |
|--------|-----------|--------|-------------|
| **task_completion** | Custom | >= 0.7 | Percentage of queries producing output |
| **faithfulness** | DeepEval | >= 0.8 | Claims supported by retrieved sources |
| **hallucination** | DeepEval | <= 0.2 | Rate of unsupported claims |
| **policy_violations** | Custom | 100% blocked | Harmful/PII requests must be blocked |

#### Soft Gates (Warnings)

| Metric | Framework | Target | Description |
|--------|-----------|--------|-------------|
| **answer_relevancy** | DeepEval | >= 0.7 | Output relevance to query |
| **context_precision** | Ragas | >= 0.6 | Relevance of retrieved context |
| **context_recall** | Ragas | >= 0.6 | Coverage of required information |

### Running Evaluations

```bash
# Smoke test (2 scenarios)
python ci/evaluation/run_evaluation.py \
  --scenarios ci/evaluation/curated_test_cases.yaml \
  --group smoke_test \
  --output ci/evaluation/smoke_test_results.json \
  --verbose

# Full evaluation (10 scenarios)
python ci/evaluation/run_evaluation.py \
  --scenarios ci/evaluation/curated_test_cases.yaml \
  --group full_evaluation \
  --output ci/evaluation/eval_results.json \
  --verbose

# Compute DeepEval/Ragas metrics
python ci/evaluation/compute_metrics.py \
  --input ci/evaluation/eval_results.json \
  --output ci/evaluation/metrics_results.json

# Generate report
python ci/evaluation/generate_report.py \
  --metrics ci/evaluation/metrics_results.json \
  --output ci/evaluation/EVALUATION_REPORT.md
```

### Test Case Categories

| Category | Count | Description |
|----------|-------|-------------|
| **Competitor Analysis** | 1 | Market research queries |
| **Technical Research** | 1 | Technical/scientific queries |
| **Market Sizing** | 1 | Business intelligence queries |
| **Regulatory Research** | 1 | Compliance-focused queries |
| **Product Comparison** | 1 | Comparative analysis |
| **Executive Summary** | 1 | Synthesis queries |
| **Quick Fact Check** | 1 | Fact verification |
| **News Synthesis** | 1 | Current events analysis |
| **Policy Violation (PII)** | 1 | Should be blocked |
| **Policy Violation (Harmful)** | 1 | Should be blocked |

### Dataset Flywheel

User feedback is collected and used for continuous improvement:

```mermaid
flowchart LR
    Research["Research Session"] --> Feedback["User Feedback"]
    Feedback --> Collector["Dataset Collector"]
    Collector --> Quality["Quality Tiers"]
    Quality --> Export["Training Export"]
    Export --> FineTune["Fine-tuning"]
    FineTune --> Improve["Improved Models"]
    Improve --> Research
```

### Evaluation Files

| File | Purpose |
|------|---------|
| `ci/evaluation/curated_test_cases.yaml` | 10 curated test scenarios |
| `ci/evaluation/run_evaluation.py` | Main evaluation runner |
| `ci/evaluation/compute_metrics.py` | DeepEval/Ragas metric computation |
| `ci/evaluation/generate_report.py` | Markdown report generator |
| `ci/evaluation/eval_results.json` | Raw evaluation outputs |
| `ci/evaluation/metrics_results.json` | Computed metrics |
| `ci/evaluation/EVALUATION_REPORT.md` | Human-readable report |

---

## Roadmap

- [x] **v1.0**: Core multi-agent research system
- [x] **v1.1**: Knowledge Graph & parallel execution
- [x] **v1.2**: DAG visualization & argument graph UI
- [x] **v1.3**: PDF extraction & citation verification
- [x] **v1.4**: Dataset flywheel & feedback collection
- [ ] **v1.5**: RLHF fine-tuning pipeline export
- [ ] **v2.0**: Hierarchical multi-agent orchestration

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [Phoenix](https://github.com/Arize-ai/phoenix) - LLM observability
- [DeepEval](https://github.com/confident-ai/deepeval) - Evaluation framework
- [OpenRouter](https://openrouter.ai) - LLM gateway
- [FastAPI](https://fastapi.tiangolo.com) - API framework
- [D3.js](https://d3js.org) - DAG visualization
- [Cytoscape.js](https://js.cytoscape.org) - Knowledge graph visualization
