# DRX - Deep Research X

**Enterprise-grade, self-hosted multi-agent research system with full observability and deterministic replay.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-green.svg)](https://github.com/langchain-ai/langgraph)

---

## What is DRX?

DRX (Deep Research X) is a **self-hosted, multi-agent research orchestration platform** that performs comprehensive research tasks by coordinating specialized AI agents in a directed acyclic graph (DAG) workflow. Unlike consumer deep research products, DRX gives you full control over your research infrastructure, data sovereignty, and the ability to customize every aspect of the research pipeline.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DRX Architecture                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   User Query                                                             │
│       │                                                                  │
│       ▼                                                                  │
│   ┌─────────┐    ┌──────────┐    ┌────────┐    ┌─────────────┐          │
│   │ Planner │───▶│ Searcher │───▶│ Reader │───▶│ Synthesizer │          │
│   └─────────┘    └──────────┘    └────────┘    └─────────────┘          │
│       │              │               │               │                   │
│       │              ▼               ▼               ▼                   │
│       │         ┌─────────────────────────────────────┐                 │
│       │         │         Shared State (Redis)         │                 │
│       │         └─────────────────────────────────────┘                 │
│       │                          │                                       │
│       ▼                          ▼                                       │
│   ┌────────┐              ┌──────────┐                                  │
│   │ Critic │◀────────────▶│ Reporter │───▶ Final Report                 │
│   └────────┘              └──────────┘                                  │
│       │                                                                  │
│       └──────── Iterative Refinement Loop ────────┘                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Why DRX Over Consumer Products?

### Comparison Matrix

| Feature | DRX | Gemini Deep Research | OpenAI | Perplexity | Kimi/Tongyi |
|---------|-----|---------------------|--------|------------|-------------|
| **Self-Hosted** | ✅ Full control | ❌ Google Cloud | ❌ OpenAI servers | ❌ SaaS only | ❌ Cloud only |
| **Data Sovereignty** | ✅ Your infrastructure | ❌ Data sent to Google | ❌ Data sent to OpenAI | ❌ Data retained | ❌ Regional restrictions |
| **Open Source** | ✅ MIT License | ❌ Proprietary | ❌ Proprietary | ❌ Proprietary | ❌ Proprietary |
| **Custom Agents** | ✅ Add/modify agents | ❌ Fixed pipeline | ❌ Fixed pipeline | ❌ Fixed pipeline | ❌ Fixed pipeline |
| **Model Agnostic** | ✅ Any LLM via OpenRouter | ❌ Gemini only | ❌ GPT only | ❌ Limited models | ❌ Vendor locked |
| **Checkpoint/Resume** | ✅ Pause & continue | ❌ | ❌ | ❌ | ❌ |
| **Deterministic Replay** | ✅ Debug & reproduce | ❌ | ❌ | ❌ | ❌ |
| **Circuit Breakers** | ✅ Fault tolerance | ❌ | ❌ | ❌ | ❌ |
| **Policy Firewall** | ✅ Domain/budget controls | ❌ | ❌ | ❌ | ❌ |
| **Full Observability** | ✅ Phoenix/OpenTelemetry | ❌ Limited | ❌ Limited | ❌ None | ❌ None |
| **Evaluation Pipeline** | ✅ DeepEval/Ragas | ❌ | ❌ | ❌ | ❌ |
| **Enterprise SSO** | ✅ Customizable auth | ❌ Google Workspace | ❌ Enterprise tier | ❌ Enterprise tier | ❌ Limited |
| **Cost Control** | ✅ Per-agent budgets | ❌ Usage-based | ❌ Usage-based | ❌ Subscription | ❌ Subscription |

### Key Differentiators

#### 1. **True Self-Hosting & Data Sovereignty**
Unlike Gemini Deep Research, OpenAI, or Perplexity, DRX runs entirely on your infrastructure. Your research queries, source documents, and synthesized reports never leave your network. Critical for:
- Regulated industries (healthcare, finance, legal)
- Government and defense applications
- Proprietary research with trade secrets
- GDPR/CCPA compliance requirements

#### 2. **Multi-Agent DAG Architecture**
DRX uses a transparent, auditable pipeline of specialized agents:
- **Planner**: Decomposes complex queries into research sub-tasks
- **Searcher**: Discovers sources via web search and RAG retrieval
- **Reader**: Extracts and structures information from sources
- **Synthesizer**: Aggregates findings, resolves conflicts, builds argument graphs
- **Critic**: Reviews quality, identifies gaps, triggers re-research
- **Reporter**: Generates final deliverables with citations

Consumer products are black boxes—you can't see or modify the research process.

#### 3. **Checkpoint & Resume**
Long research tasks (hours or days) can be paused and resumed:
```bash
# Start research
curl -X POST /api/v1/interactions -d '{"query": "..."}'
# Returns interaction_id

# Resume from checkpoint after system restart
curl -X POST /api/v1/interactions/{id}/resume?checkpoint_id=chk_abc123
```

Consumer products lose all progress if you close the tab or hit a timeout.

#### 4. **Deterministic Replay**
Debug and reproduce any research run:
```bash
# Replay a research session
curl -X POST /api/v1/interactions/{id}/replay

# Compare original vs replay
curl -X POST /api/v1/interactions/{id}/compare
```

Essential for:
- Debugging unexpected research outputs
- Training data generation for fine-tuning
- Audit trails and compliance
- A/B testing agent improvements

#### 5. **Agentic Metadata Infrastructure**
Enterprise-grade reliability features:

- **Circuit Breakers**: Automatic failover when agents degrade
- **Policy Firewall**: Enforce domain restrictions, rate limits, budgets
- **Capability Routing**: Dynamic agent selection based on requirements
- **Health Monitoring**: Real-time agent health tracking

#### 6. **Model Agnostic via OpenRouter**
Use any LLM without code changes:
```env
OPENROUTER_API_KEY=sk-or-...
DEFAULT_MODEL=google/gemini-2.0-flash-exp
# or anthropic/claude-3.5-sonnet, deepseek/deepseek-r1, etc.
```

#### 7. **Full Observability Stack**
Self-hosted Phoenix for complete visibility:
- Trace every LLM call, tool invocation, and agent decision
- Visualize research DAG execution
- Monitor token usage and latency
- Debug issues with distributed tracing

---

## Quick Launch

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- API keys: OpenRouter, Tavily (optional)

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/drx.git
cd drx/v1

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   OPENROUTER_API_KEY=sk-or-...
#   TAVILY_API_KEY=tvly-...

# Launch all services
docker compose up -d

# Services started:
#   - API:      http://localhost:8000
#   - Phoenix:  http://localhost:6006
#   - Postgres: localhost:5432
#   - Redis:    localhost:6379
```

### Option 2: Local Development

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev,eval]"

# Or use requirements.txt
pip install -r requirements.txt

# Start infrastructure
docker compose up -d postgres redis phoenix

# Initialize database
psql $DATABASE_URL < schemas/postgres_schema.sql

# Run API server
uvicorn src.api.main:app --reload --port 8000

# Run Celery worker (in another terminal)
celery -A src.worker worker -l info
```

### Option 3: Quick Test

```bash
# After services are running, test the API
curl -X POST http://localhost:8000/api/v1/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key differences between transformer and state-space models for sequence modeling?",
    "config": {
      "max_iterations": 3,
      "thinking_summaries": true
    }
  }'

# Stream results
curl -N http://localhost:8000/api/v1/interactions/{interaction_id}/stream
```

---

## Project Structure

```
v1/
├── src/
│   ├── agents/           # Specialized research agents
│   │   ├── planner.py    # Query decomposition & DAG planning
│   │   ├── searcher.py   # Web search & source discovery
│   │   ├── reader.py     # Content extraction & structuring
│   │   ├── synthesizer.py# Information synthesis & conflict resolution
│   │   ├── critic.py     # Quality assessment & gap analysis
│   │   └── reporter.py   # Report generation & citation management
│   ├── orchestrator/     # LangGraph workflow orchestration
│   │   ├── workflow.py   # DAG definition & execution
│   │   ├── state.py      # TypedDict state definitions
│   │   └── checkpointer.py# AsyncPostgresSaver integration
│   ├── api/              # FastAPI REST API
│   │   ├── routes.py     # Main endpoints
│   │   ├── streaming.py  # SSE implementation
│   │   └── replay_routes.py# Replay & debugging endpoints
│   ├── metadata/         # Agentic metadata infrastructure
│   │   ├── manifest.py   # Agent manifest schema
│   │   ├── routing.py    # Capability-based routing
│   │   ├── circuit_breaker.py# Fault tolerance
│   │   └── context.py    # Context propagation
│   ├── middleware/       # Policy enforcement
│   │   ├── policy_firewall.py# Domain, budget, rate limits
│   │   └── domain_validator.py# URL validation
│   ├── replay/           # Deterministic replay
│   │   ├── recorder.py   # Event recording
│   │   └── player.py     # Replay execution
│   ├── services/         # External service clients
│   │   ├── openrouter_client.py# LLM API
│   │   ├── redis_client.py# Cache & pubsub
│   │   └── active_state.py# Real-time metrics
│   ├── tools/            # Agent tools
│   │   ├── tavily_search.py# Web search
│   │   └── rag_retriever.py# Vector retrieval
│   └── observability/    # Tracing & metrics
│       ├── phoenix.py    # Phoenix integration
│       └── tracing.py    # OpenTelemetry spans
├── schemas/              # Data definitions
│   ├── postgres_schema.sql# Database DDL
│   ├── agent_manifest.json# Agent manifest JSON Schema
│   └── redis_keys.md     # Redis key patterns
├── ci/evaluation/        # Evaluation pipeline
│   ├── test_agent_evals.py# DeepEval tests
│   └── conftest.py       # Pytest fixtures
├── deployment/           # Docker & infrastructure
│   ├── docker-compose.yaml
│   ├── Dockerfile.api
│   └── Dockerfile.worker
└── .github/workflows/    # CI/CD
    ├── ci.yaml
    └── evaluation.yaml
```

---

## Configuration

### Environment Variables

```env
# Required
OPENROUTER_API_KEY=sk-or-v1-...        # LLM API access
DATABASE_URL=postgresql://...           # PostgreSQL connection
REDIS_URL=redis://localhost:6379       # Redis connection

# Optional - Search
TAVILY_API_KEY=tvly-...                # Web search API

# Optional - Observability
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317
PHOENIX_PROJECT_NAME=drx-research

# Optional - Model Selection
DEFAULT_MODEL=google/gemini-2.0-flash-exp
REASONING_MODEL=deepseek/deepseek-r1

# Optional - Limits
MAX_ITERATIONS=5                       # Max research cycles
TOKEN_BUDGET=100000                    # Max tokens per session
DEFAULT_RATE_LIMIT_RPM=60             # Requests per minute
```

### Agent Manifest

Configure per-agent behavior in `schemas/agent_manifest.json`:

```json
{
  "agent_id": "searcher_v1",
  "capabilities": ["web_search", "source_discovery"],
  "allowed_domains": ["*.gov", "*.edu", "arxiv.org"],
  "blocked_domains": ["facebook.com", "twitter.com"],
  "max_budget_usd": 0.50,
  "rate_limits": {
    "requests_per_minute": 30,
    "tokens_per_minute": 50000
  }
}
```

---

## API Reference

### Create Research Interaction
```http
POST /api/v1/interactions
Content-Type: application/json

{
  "query": "Your research question",
  "steerability": {
    "tone": "academic",
    "format": "markdown",
    "max_sources": 20
  },
  "config": {
    "max_iterations": 5,
    "thinking_summaries": true
  }
}
```

### Stream Progress (SSE)
```http
GET /api/v1/interactions/{id}/stream
Accept: text/event-stream
Last-Event-ID: {checkpoint_id}  # Optional, for resume
```

### Replay Session
```http
POST /api/v1/interactions/{id}/replay
Content-Type: application/json

{
  "from_checkpoint": "chk_abc123",
  "modifications": {
    "tools": {"web_search": {"max_results": 5}}
  }
}
```

### Health Check
```http
GET /api/v1/health
```

---

## Evaluation

Run the evaluation pipeline:

```bash
# Run all evaluations
pytest ci/evaluation/ -v

# Run CI gate tests only
pytest ci/evaluation/ -m ci_gate -v

# Run with coverage
pytest ci/evaluation/ --cov=src --cov-report=html
```

Evaluation metrics:
- **Faithfulness** (Ragas): >= 0.8
- **Task Completion** (DeepEval): >= 0.7
- **Hallucination Rate**: <= 0.2
- **Policy Violations**: == 0

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,eval]"

# Run linting
ruff check .

# Run type checking
mypy src/

# Run tests
pytest tests/ -v

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

---

## Roadmap

- [ ] **v1.1**: NeMo Guardrails integration for safety
- [ ] **v1.2**: Interactive DAG editing UI
- [ ] **v1.3**: RLHF fine-tuning pipeline export
- [ ] **v1.4**: Multi-modal research (images, PDFs)
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
- [Phoenix](https://github.com/Arize-ai/phoenix) - Observability
- [DeepEval](https://github.com/confident-ai/deepeval) - Evaluation
- [OpenRouter](https://openrouter.ai) - LLM gateway
