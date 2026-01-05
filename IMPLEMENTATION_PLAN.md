# DRX v1 Implementation Plan

## Topological Order of Work Packets

```
WP1: Foundation ─────────────────────────────────────────────────────────────┐
     │                                                                        │
     ▼                                                                        │
WP2: Database Schema ────────────┬───────────────────────────────────────────┤
     │                           │                                            │
     ▼                           ▼                                            │
WP3: State Definitions      WP4: Infrastructure                               │
     │                           │                                            │
     └───────────┬───────────────┘                                            │
                 ▼                                                            │
WP5: Core Services ──────────────────────────────────────────────────────────┤
     │                                                                        │
     ▼                                                                        │
WP6: Tools ──────────────────────────────────────────────────────────────────┤
     │                                                                        │
     ▼                                                                        │
WP7: Agents ─────────────────────────────────────────────────────────────────┤
     │                                                                        │
     ▼                                                                        │
WP8: Orchestrator ───────────────────────────────────────────────────────────┤
     │                                                                        │
     ├───────────────────────────┐                                            │
     ▼                           ▼                                            │
WP9: API                    WP10: Observability                               │
     │                           │                                            │
     └───────────┬───────────────┘                                            │
                 ▼                                                            │
WP11: Evaluation ────────────────────────────────────────────────────────────┤
     │                                                                        │
     ▼                                                                        │
WP12: CI/CD ─────────────────────────────────────────────────────────────────┘
```

## Work Packet Specifications

### WP1: Foundation & Configuration
**Dependencies**: None
**Owner**: Foundation Agent
**Files**:
- `v1/pyproject.toml`
- `v1/.env.example`
- `v1/src/__init__.py` (and all submodule __init__.py)
- `v1/README.md`

### WP2: Database Schema & Models
**Dependencies**: WP1
**Owner**: Schema Agent
**Files**:
- `v1/schemas/postgres_schema.sql`
- `v1/schemas/redis_keys.md`
- `v1/schemas/agent_events.json`

### WP3: State Definitions
**Dependencies**: WP2
**Owner**: State Agent
**Files**:
- `v1/src/orchestrator/state.py`
- `v1/src/api/models.py`
- `v1/src/config.py`

### WP4: Infrastructure
**Dependencies**: WP2
**Owner**: Infrastructure Agent
**Files**:
- `v1/deployment/docker-compose.yaml`
- `v1/deployment/docker-compose.prod.yaml`
- `v1/deployment/Dockerfile.api`
- `v1/deployment/Dockerfile.worker`

### WP5: Core Services
**Dependencies**: WP3, WP4
**Owner**: Services Agent
**Files**:
- `v1/src/db/connection.py`
- `v1/src/orchestrator/checkpointer.py`
- `v1/src/services/redis_client.py`
- `v1/src/services/openrouter_client.py`

### WP6: Tools
**Dependencies**: WP5
**Owner**: Tools Agent
**Files**:
- `v1/src/tools/base.py`
- `v1/src/tools/tavily_search.py`
- `v1/src/tools/openrouter_search.py`
- `v1/src/tools/rag_retriever.py`

### WP7: Agents
**Dependencies**: WP6
**Owner**: Agents Agent
**Files**:
- `v1/src/agents/base.py`
- `v1/src/agents/planner.py`
- `v1/src/agents/searcher.py`
- `v1/src/agents/reader.py`
- `v1/src/agents/synthesizer.py`
- `v1/src/agents/critic.py`
- `v1/src/agents/reporter.py`

### WP8: Orchestrator
**Dependencies**: WP7
**Owner**: Orchestrator Agent
**Files**:
- `v1/src/orchestrator/workflow.py`
- `v1/src/orchestrator/nodes.py`

### WP9: API
**Dependencies**: WP8
**Owner**: API Agent
**Files**:
- `v1/src/api/main.py`
- `v1/src/api/routes.py`
- `v1/src/api/streaming.py`
- `v1/src/api/dependencies.py`

### WP10: Observability
**Dependencies**: WP8
**Owner**: Observability Agent
**Files**:
- `v1/src/observability/phoenix.py`
- `v1/src/observability/tracing.py`

### WP11: Evaluation
**Dependencies**: WP9, WP10
**Owner**: Evaluation Agent
**Files**:
- `v1/ci/evaluation/conftest.py`
- `v1/ci/evaluation/test_agent_evals.py`
- `v1/ci/evaluation/scenarios/*.yaml`

### WP12: CI/CD
**Dependencies**: WP11
**Owner**: CI/CD Agent
**Files**:
- `v1/.github/workflows/ci.yaml`
- `v1/.github/workflows/evaluation.yaml`
- `v1/.pre-commit-config.yaml`

## Parallelization Strategy

**Wave 1** (Parallel):
- WP1: Foundation

**Wave 2** (Parallel after WP1):
- WP2: Database Schema

**Wave 3** (Parallel after WP2):
- WP3: State Definitions
- WP4: Infrastructure

**Wave 4** (After WP3 + WP4):
- WP5: Core Services

**Wave 5** (After WP5):
- WP6: Tools

**Wave 6** (After WP6):
- WP7: Agents

**Wave 7** (After WP7):
- WP8: Orchestrator

**Wave 8** (Parallel after WP8):
- WP9: API
- WP10: Observability

**Wave 9** (After WP9 + WP10):
- WP11: Evaluation

**Wave 10** (After WP11):
- WP12: CI/CD
