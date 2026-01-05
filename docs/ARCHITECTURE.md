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

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    CLIENT LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Web UI     │    │   CLI Tool   │    │  SDK Client  │    │  Webhook     │       │
│  │  (React/Vue) │    │   (Python)   │    │ (Python/JS)  │    │  Consumer    │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                   │               │
│         └───────────────────┴───────────────────┴───────────────────┘               │
│                                      │                                               │
│                              HTTPS / WebSocket                                       │
│                                      │                                               │
├──────────────────────────────────────┼──────────────────────────────────────────────┤
│                                API GATEWAY LAYER                                     │
├──────────────────────────────────────┼──────────────────────────────────────────────┤
│                                      ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                           FastAPI Application                                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │
│  │  │ Auth/RBAC   │  │ Rate Limit  │  │   CORS      │  │  Request Validation │   │  │
│  │  │ Middleware  │  │ Middleware  │  │ Middleware  │  │     Middleware      │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘   │  │
│  │                                                                                │  │
│  │  ┌──────────────────────────────────────────────────────────────────────────┐ │  │
│  │  │                            Route Handlers                                 │ │  │
│  │  │  POST /interactions  │  GET /stream  │  POST /replay  │  GET /health    │ │  │
│  │  └──────────────────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                               │
├──────────────────────────────────────┼──────────────────────────────────────────────┤
│                              ORCHESTRATION LAYER                                     │
├──────────────────────────────────────┼──────────────────────────────────────────────┤
│                                      ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                        LangGraph StateGraph Orchestrator                       │  │
│  │                                                                                │  │
│  │    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────────┐        │  │
│  │    │ Planner  │────▶│ Searcher │────▶│  Reader  │────▶│ Synthesizer  │        │  │
│  │    │  Agent   │     │  Agent   │     │  Agent   │     │    Agent     │        │  │
│  │    └──────────┘     └──────────┘     └──────────┘     └──────────────┘        │  │
│  │         │                                                    │                 │  │
│  │         │           ┌──────────┐     ┌──────────┐           │                 │  │
│  │         └──────────▶│  Critic  │◀───▶│ Reporter │◀──────────┘                 │  │
│  │                     │  Agent   │     │  Agent   │                             │  │
│  │                     └──────────┘     └──────────┘                             │  │
│  │                          │                │                                    │  │
│  │                          └────────────────┴──▶ Final Output                   │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                               │
├──────────────────────────────────────┼──────────────────────────────────────────────┤
│                               METADATA LAYER                                         │
├──────────────────────────────────────┼──────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   Policy    │  │  Circuit    │  │ Capability  │  │  Context    │                 │
│  │  Firewall   │  │  Breaker    │  │   Router    │  │ Propagator  │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                      │                                               │
├──────────────────────────────────────┼──────────────────────────────────────────────┤
│                                 DATA LAYER                                           │
├──────────────────────────────────────┼──────────────────────────────────────────────┤
│                                      ▼                                               │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐                │
│  │     PostgreSQL    │  │       Redis       │  │   Object Store    │                │
│  │   + pgvector      │  │    (Valkey)       │  │    (Optional)     │                │
│  │                   │  │                   │  │                   │                │
│  │ • Sessions        │  │ • Active State    │  │ • Large Files     │                │
│  │ • Steps           │  │ • Rate Limits     │  │ • Attachments     │                │
│  │ • Checkpoints     │  │ • Circuit Status  │  │ • Exports         │                │
│  │ • Embeddings      │  │ • Pub/Sub Events  │  │                   │                │
│  │ • Violations      │  │ • Session Cache   │  │                   │                │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘                │
│                                                                                      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                            EXTERNAL SERVICES LAYER                                   │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐                │
│  │    OpenRouter     │  │      Tavily       │  │      Phoenix      │                │
│  │    (LLM API)      │  │   (Web Search)    │  │  (Observability)  │                │
│  │                   │  │                   │  │                   │                │
│  │ • Gemini Flash    │  │ • Search API      │  │ • Trace Storage   │                │
│  │ • DeepSeek R1     │  │ • Extract API     │  │ • Span Viewer     │                │
│  │ • Claude          │  │                   │  │ • Eval Dashboard  │                │
│  │ • GPT-4           │  │                   │  │                   │                │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘                │
└──────────────────────────────────────────────────────────────────────────────────────┘
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

```
                                    ┌─────────────────────┐
                                    │    User Request     │
                                    │   (Research Query)  │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                              ┌────────────────────────────────┐
                              │         PLANNER AGENT          │
                              │                                │
                              │  • Query analysis              │
                              │  • Sub-question decomposition  │
                              │  • DAG generation              │
                              │  • Priority assignment         │
                              └────────────────┬───────────────┘
                                               │
                                               ▼
                              ┌────────────────────────────────┐
                              │        SEARCHER AGENT          │
                              │                                │
                              │  • Query expansion             │
                              │  • Web search (Tavily)         │
                              │  • RAG retrieval (pgvector)    │
                              │  • Source deduplication        │
                              │  • Relevance filtering         │
                              └────────────────┬───────────────┘
                                               │
                          ┌────────────────────┴────────────────────┐
                          │           Parallel Fan-Out              │
                          ▼                                         ▼
               ┌──────────────────┐                     ┌──────────────────┐
               │   READER AGENT   │        ...          │   READER AGENT   │
               │   (Source 1)     │                     │   (Source N)     │
               │                  │                     │                  │
               │ • Content fetch  │                     │ • Content fetch  │
               │ • HTML parsing   │                     │ • PDF extraction │
               │ • Entity extract │                     │ • Entity extract │
               │ • Citation build │                     │ • Citation build │
               └────────┬─────────┘                     └────────┬─────────┘
                        │                                        │
                        └────────────────┬───────────────────────┘
                                         │
                                         ▼
                              ┌────────────────────────────────┐
                              │      SYNTHESIZER AGENT         │
                              │                                │
                              │  • Finding aggregation         │
                              │  • Conflict detection          │
                              │  • Argument graph building     │
                              │  • Consensus formation         │
                              │  • Evidence weighting          │
                              └────────────────┬───────────────┘
                                               │
                                               ▼
                              ┌────────────────────────────────┐
                              │         CRITIC AGENT           │
                              │                                │
                              │  • Quality assessment          │
                              │  • Gap identification          │
                              │  • Source verification         │
                              │  • Hallucination detection     │
                              │  • Coverage scoring            │
                              └────────────────┬───────────────┘
                                               │
                                    ┌──────────┴──────────┐
                                    │    Coverage OK?     │
                                    └──────────┬──────────┘
                                               │
                           ┌───────────────────┴───────────────────┐
                           │                                       │
                     NO (gaps found)                         YES (complete)
                           │                                       │
                           ▼                                       ▼
              ┌─────────────────────┐               ┌────────────────────────────┐
              │  Generate new sub-  │               │       REPORTER AGENT       │
              │  questions for gaps │               │                            │
              └──────────┬──────────┘               │  • Report generation       │
                         │                          │  • Citation formatting     │
                         │                          │  • Executive summary       │
                         ▼                          │  • Multi-format export     │
              ┌─────────────────────┐               │    (MD, HTML, PDF, JSON)   │
              │   Back to PLANNER   │               └────────────────┬───────────┘
              │  (iteration += 1)   │                                │
              └─────────────────────┘                                ▼
                                                    ┌────────────────────────────┐
                                                    │       Final Output         │
                                                    │   (Research Report)        │
                                                    └────────────────────────────┘
```

### State Machine Diagram

```
                                    ┌─────────────┐
                                    │   PENDING   │
                                    │   (init)    │
                                    └──────┬──────┘
                                           │ start()
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              RUNNING STATE                                    │
│                                                                               │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │  plan    │───▶│  search  │───▶│   read   │───▶│synthesize│              │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│        │                                               │                      │
│        │              ┌──────────┐                     │                      │
│        └─────────────▶│ critique │◀────────────────────┘                      │
│                       └────┬─────┘                                            │
│                            │                                                  │
│              ┌─────────────┼─────────────┐                                   │
│              │             │             │                                    │
│              ▼             ▼             ▼                                    │
│        ┌──────────┐  ┌──────────┐  ┌──────────┐                              │
│        │  report  │  │ continue │  │   fail   │                              │
│        └────┬─────┘  └────┬─────┘  └────┬─────┘                              │
│             │             │             │                                     │
└─────────────┼─────────────┼─────────────┼─────────────────────────────────────┘
              │             │             │
              ▼             │             ▼
       ┌───────────┐        │      ┌───────────┐
       │ COMPLETED │        │      │  FAILED   │
       └───────────┘        │      └───────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       ┌───────────┐              ┌───────────┐
       │  PAUSED   │◀────────────▶│ CANCELLED │
       │ (resume)  │   cancel()   └───────────┘
       └───────────┘
```

### Agent Communication Pattern

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Shared State (AgentState)                             │
│                                                                                  │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐  │
│  │   messages   │    plan      │   findings   │  citations   │   synthesis  │  │
│  │  (chat log)  │  (DAG nodes) │ (extracted)  │  (sources)   │  (combined)  │  │
│  └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘  │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐  │
│  │     gaps     │ final_report │ iteration    │ tokens_used  │   metrics    │  │
│  │  (missing)   │   (output)   │   (count)    │   (budget)   │  (quality)   │  │
│  └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
            ▼                           ▼                           ▼
     ┌─────────────┐             ┌─────────────┐             ┌─────────────┐
     │   Agent 1   │             │   Agent 2   │             │   Agent N   │
     │             │             │             │             │             │
     │ read state  │             │ read state  │             │ read state  │
     │     ↓       │             │     ↓       │             │     ↓       │
     │  process    │             │  process    │             │  process    │
     │     ↓       │             │     ↓       │             │     ↓       │
     │ write state │             │ write state │             │ write state │
     └─────────────┘             └─────────────┘             └─────────────┘
```

---

## Component Architecture

### API Gateway Components

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Application                                 │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                           Middleware Stack                                  │ │
│  │                                                                             │ │
│  │  Request ──▶ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ──▶ Handler  │ │
│  │              │Request  │ │  Auth   │ │  Rate   │ │ Timing  │               │ │
│  │              │   ID    │ │  Check  │ │  Limit  │ │  Track  │               │ │
│  │              └─────────┘ └─────────┘ └─────────┘ └─────────┘               │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                            Route Handlers                                   │ │
│  │                                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ /api/v1/interactions                                                │   │ │
│  │  │   POST   - Create new research interaction                          │   │ │
│  │  │   GET    - List interactions                                        │   │ │
│  │  │   GET /{id} - Get interaction details                               │   │ │
│  │  │   DELETE /{id} - Cancel interaction                                 │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ /api/v1/interactions/{id}/stream                                    │   │ │
│  │  │   GET    - SSE stream for real-time updates                         │   │ │
│  │  │           Supports Last-Event-ID for resume                         │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ /api/v1/interactions/{id}/replay                                    │   │ │
│  │  │   POST   - Start replay from checkpoint                             │   │ │
│  │  │   GET /events - Get recorded events                                 │   │ │
│  │  │   POST /compare - Compare original vs replay                        │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ /api/v1/health                                                      │   │ │
│  │  │   GET    - Health check endpoint                                    │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         Dependency Injection                                │ │
│  │                                                                             │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │ │
│  │  │  DatabaseDep │  │   RedisDep   │  │OrchestratorDep│  │CurrentUserDep│   │ │
│  │  │  (psycopg)   │  │   (redis)    │  │ (LangGraph)  │  │   (auth)     │   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               BaseAgent                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Properties:                          Methods:                                   │
│  ┌────────────────────────┐          ┌─────────────────────────────────────┐   │
│  │ • agent_id: str        │          │ • __call__(state) -> AgentState     │   │
│  │ • agent_type: str      │          │ • _process(state) -> AgentResponse  │   │
│  │ • system_prompt: str   │          │ • _post_process(state, response)    │   │
│  │ • llm_client: LLMClient│          │ • _validate_input(state) -> bool    │   │
│  │ • tools: list[Tool]    │          │ • _emit_event(event_type, data)     │   │
│  │ • manifest: AgentManifest│        │ • _record_metrics(metrics)          │   │
│  └────────────────────────┘          └─────────────────────────────────────┘   │
│                                                                                  │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ extends
        ┌────────────────┬───────────────┼───────────────┬────────────────┐
        │                │               │               │                │
        ▼                ▼               ▼               ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│PlannerAgent  │ │SearcherAgent │ │ ReaderAgent  │ │SynthesizerAgent│ │ CriticAgent │
├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────────┤
│              │ │              │ │              │ │              │ │              │
│Capabilities: │ │Capabilities: │ │Capabilities: │ │Capabilities: │ │Capabilities: │
│• decompose   │ │• web_search  │ │• html_parse  │ │• aggregate   │ │• evaluate    │
│• prioritize  │ │• rag_search  │ │• pdf_extract │ │• conflict_   │ │• verify      │
│• dag_build   │ │• query_      │ │• entity_     │ │  resolve     │ │• gap_find    │
│              │ │  expand      │ │  extract     │ │• argument_   │ │• score       │
│              │ │              │ │              │ │  graph       │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │                │               │               │                │
        │                │               │               │                │
        ▼                ▼               ▼               ▼                ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              Tool Integration                                     │
│                                                                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │  TavilySearch  │  │ RAGRetriever   │  │  HTMLParser    │  │  PDFExtractor  │ │
│  │    Tool        │  │    Tool        │  │    Tool        │  │    Tool        │ │
│  └────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Metadata Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Metadata Infrastructure                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                          Agent Manifest System                               ││
│  │                                                                              ││
│  │  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    ││
│  │  │  JSON Schema     │────▶│  Pydantic Model  │────▶│  Registry Load   │    ││
│  │  │  (Validation)    │     │  (Type Safety)   │     │  (Database)      │    ││
│  │  └──────────────────┘     └──────────────────┘     └──────────────────┘    ││
│  │                                                                              ││
│  │  Manifest Contents:                                                          ││
│  │  • agent_id, version, capabilities                                           ││
│  │  • allowed_domains, blocked_domains                                          ││
│  │  • max_budget_usd, rate_limits                                              ││
│  │  • circuit_breaker config                                                    ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                          Policy Firewall                                     ││
│  │                                                                              ││
│  │            Tool Invocation                                                   ││
│  │                  │                                                           ││
│  │                  ▼                                                           ││
│  │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       ││
│  │  │ Domain Validator │───▶│ Budget Enforcer  │───▶│ Rate Limiter     │       ││
│  │  │                  │    │                  │    │                  │       ││
│  │  │ • URL extraction │    │ • Spend tracking │    │ • Sliding window │       ││
│  │  │ • Wildcard match │    │ • Cost estimate  │    │ • Token counting │       ││
│  │  │ • Block/allow    │    │ • Budget check   │    │ • Burst control  │       ││
│  │  └──────────────────┘    └──────────────────┘    └──────────────────┘       ││
│  │            │                      │                      │                   ││
│  │            └──────────────────────┴──────────────────────┘                   ││
│  │                                   │                                          ││
│  │                                   ▼                                          ││
│  │                    ┌──────────────────────────────┐                         ││
│  │                    │     Violation Logger         │                         ││
│  │                    │                              │                         ││
│  │                    │ • PostgreSQL audit table     │                         ││
│  │                    │ • Redis event stream         │                         ││
│  │                    └──────────────────────────────┘                         ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                          Circuit Breaker                                     ││
│  │                                                                              ││
│  │      ┌─────────┐         ┌─────────┐         ┌─────────────┐                ││
│  │      │ CLOSED  │────────▶│  OPEN   │────────▶│ HALF-OPEN   │                ││
│  │      │(normal) │ failure │(failing)│ timeout │  (testing)  │                ││
│  │      └────┬────┘ threshold└─────────┘         └──────┬──────┘                ││
│  │           │                                          │                       ││
│  │           └─────────────── success ──────────────────┘                       ││
│  │                          threshold                                           ││
│  │                                                                              ││
│  │  Integrations:                                                               ││
│  │  • Redis state storage (drx:agent:{id}:circuit)                             ││
│  │  • Health checker (latency, error rates)                                     ││
│  │  • Automatic rerouting to alternative agents                                 ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                       Capability-Based Routing                               ││
│  │                                                                              ││
│  │  Task Requirements          Agent Scoring                 Selection          ││
│  │  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐       ││
│  │  │• Capabilities│────────▶ │• Capability  │────────▶ │• Best match  │       ││
│  │  │• Domain needs│          │  match: 0.3  │          │• Fallback    │       ││
│  │  │• Cost tier   │          │• Health: 0.2 │          │  selection   │       ││
│  │  │• Latency req │          │• Load: 0.2   │          │• Error if    │       ││
│  │  └──────────────┘          │• Cost: -0.1  │          │  none avail  │       ││
│  │                            └──────────────┘          └──────────────┘       ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Request Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Request Lifecycle                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

1. CLIENT REQUEST
   │
   │  POST /api/v1/interactions
   │  {
   │    "query": "Research quantum computing advances in 2024",
   │    "config": {"max_iterations": 5}
   │  }
   │
   ▼
2. API GATEWAY
   │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │  │ Validate    │─▶│ Authenticate│─▶│ Rate Limit  │
   │  │ Request     │  │ User        │  │ Check       │
   │  └─────────────┘  └─────────────┘  └─────────────┘
   │
   ▼
3. SESSION CREATION
   │
   │  ┌────────────────────────────────────────────┐
   │  │ PostgreSQL: INSERT INTO research_sessions  │
   │  │ Redis: SET drx:session:{id} (cache)        │
   │  └────────────────────────────────────────────┘
   │
   │  Return: { "id": "int_abc123", "status": "queued" }
   │
   ▼
4. ASYNC EXECUTION (Celery Worker)
   │
   │  ┌─────────────────────────────────────────────────────────────────┐
   │  │                    LangGraph Execution                          │
   │  │                                                                 │
   │  │  State: AgentState                                              │
   │  │  ┌─────────────────────────────────────────────────────────┐   │
   │  │  │ messages, plan, findings, citations, synthesis, gaps... │   │
   │  │  └─────────────────────────────────────────────────────────┘   │
   │  │                           │                                     │
   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
   │  │  │ Planner │─▶│Searcher │─▶│ Reader  │─▶│Synthesizer          │
   │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘           │
   │  │                                              │                  │
   │  │  Each agent:                                 │                  │
   │  │  1. Policy check (firewall)                  ▼                  │
   │  │  2. LLM call (OpenRouter)            ┌─────────┐               │
   │  │  3. Tool invocations                 │ Critic  │               │
   │  │  4. State update                     └────┬────┘               │
   │  │  5. Checkpoint save                       │                    │
   │  │  6. Event emission                        ▼                    │
   │  │                                   ┌─────────────┐              │
   │  │                                   │  Reporter   │              │
   │  │                                   └─────────────┘              │
   │  └─────────────────────────────────────────────────────────────────┘
   │
   ▼
5. EVENT STREAMING (SSE)
   │
   │  GET /api/v1/interactions/{id}/stream
   │
   │  Events emitted:
   │  ┌────────────────────────────────────────────────────┐
   │  │ event: interaction.start                           │
   │  │ event: thought_summary                             │
   │  │ event: tool.use                                    │
   │  │ event: content.delta                               │
   │  │ event: checkpoint                                  │
   │  │ event: interaction.complete                        │
   │  └────────────────────────────────────────────────────┘
   │
   ▼
6. FINAL OUTPUT
   │
   │  {
   │    "id": "int_abc123",
   │    "status": "completed",
   │    "report": "# Research Report\n\n...",
   │    "citations": [...],
   │    "metrics": { "coverage": 0.92, "confidence": 0.88 }
   │  }
   │
   ▼
7. STORAGE & CLEANUP
   │
   │  ┌────────────────────────────────────────────────────┐
   │  │ PostgreSQL: UPDATE research_sessions SET status    │
   │  │ PostgreSQL: Store findings, citations              │
   │  │ Redis: EXPIRE session cache                        │
   │  │ Phoenix: Complete trace span                       │
   │  └────────────────────────────────────────────────────┘
```

### Checkpoint & Resume Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Checkpoint & Resume Flow                               │
└─────────────────────────────────────────────────────────────────────────────────┘

                    NORMAL EXECUTION
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
┌─────────┐         ┌─────────┐           ┌─────────┐
│  Node 1 │────────▶│  Node 2 │──────────▶│  Node 3 │
│(Planner)│         │(Searcher)│          │(Reader) │
└────┬────┘         └────┬────┘           └────┬────┘
     │                   │                     │
     ▼                   ▼                     ▼
┌─────────────────────────────────────────────────────┐
│               AsyncPostgresSaver                     │
│                                                      │
│  Checkpoint saved after each node:                   │
│  ┌────────────────────────────────────────────────┐ │
│  │ checkpoint_id: "chk_001"                       │ │
│  │ thread_id: "session_abc123"                    │ │
│  │ state: { messages, plan, findings, ... }      │ │
│  │ metadata: { node: "planner", timestamp: ... } │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
                          │
                          │ INTERRUPTION
                          │ (error, timeout, pause)
                          ▼
               ┌────────────────────┐
               │  Session Paused    │
               │  State preserved   │
               └────────────────────┘
                          │
                          │ RESUME REQUEST
                          │ POST /interactions/{id}/resume?checkpoint=chk_002
                          ▼
┌─────────────────────────────────────────────────────┐
│               Resume from Checkpoint                 │
│                                                      │
│  1. Load checkpoint state from PostgreSQL            │
│  2. Reconstruct AgentState                           │
│  3. Identify next node in DAG                        │
│  4. Continue execution                               │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────┐
    │                     │                   │
    ▼                     ▼                   ▼
┌─────────┐         ┌─────────┐         ┌─────────┐
│  Node 3 │────────▶│  Node 4 │────────▶│  Node 5 │
│(continue)│        │(Synthesize)│      │(Report) │
└─────────┘         └─────────┘         └─────────┘
```

---

## API Gateway

### Authentication Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Authentication Flow                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

                     ┌────────────────────────────┐
                     │       Incoming Request     │
                     │                            │
                     │  Headers:                  │
                     │  • Authorization: Bearer   │
                     │  • X-API-Key: drx_...      │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │     Check Auth Headers     │
                     └─────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
     ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
     │  Bearer Token  │   │   API Key      │   │   No Auth      │
     │                │   │                │   │                │
     │ JWT Validation │   │ Key Lookup     │   │ Dev Mode Only  │
     │ • Signature    │   │ • Redis cache  │   │ • is_dev check │
     │ • Expiration   │   │ • DB fallback  │   │ • Default user │
     │ • Claims       │   │ • Permissions  │   │                │
     └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
             │                    │                    │
             └────────────────────┴────────────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │        User Object         │
                     │                            │
                     │  • id: str                 │
                     │  • email: str | None       │
                     │  • is_active: bool         │
                     │  • is_admin: bool          │
                     │  • rate_limit_tier: str    │
                     └────────────────────────────┘
```

### Rate Limiting Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Rate Limiting Architecture                               │
└─────────────────────────────────────────────────────────────────────────────────┘

     Request                     Rate Limiter                    Redis Backend
        │                             │                               │
        │   Check Limit               │                               │
        │────────────────────────────▶│                               │
        │                             │   ZREMRANGEBYSCORE            │
        │                             │──────────────────────────────▶│
        │                             │   (remove old entries)        │
        │                             │                               │
        │                             │   ZCARD                       │
        │                             │──────────────────────────────▶│
        │                             │   (count current window)      │
        │                             │                               │
        │                             │◀──────────────────────────────│
        │                             │   current_count               │
        │                             │                               │
        │                    ┌────────┴────────┐                      │
        │                    │ count < limit ? │                      │
        │                    └────────┬────────┘                      │
        │                             │                               │
        │             ┌───────────────┴───────────────┐               │
        │             │                               │               │
        │            YES                             NO               │
        │             │                               │               │
        │             ▼                               ▼               │
        │   ┌─────────────────┐           ┌─────────────────┐        │
        │   │ ZADD request    │           │ Return 429      │        │
        │   │ Set TTL         │           │ Retry-After     │        │
        │   │ Allow request   │           │ header          │        │
        │   └─────────────────┘           └─────────────────┘        │
        │             │                                               │
        │◀────────────┴───────────────────────────────────────────────│


   Rate Limit Tiers:
   ┌─────────────────────────────────────────────────────────────────┐
   │  Tier      │  Requests/min  │  Requests/hour  │  Burst Limit   │
   ├────────────┼────────────────┼─────────────────┼────────────────┤
   │  standard  │      60        │     1,000       │      10        │
   │  premium   │     300        │     5,000       │      50        │
   │  unlimited │  10,000        │   100,000       │   1,000        │
   └─────────────────────────────────────────────────────────────────┘
```

---

## External Integrations

### OpenRouter LLM Integration

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           OpenRouter Integration                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

     DRX Agent                  OpenRouter Client               OpenRouter API
         │                            │                              │
         │  generate(prompt, model)   │                              │
         │───────────────────────────▶│                              │
         │                            │                              │
         │                            │   POST /api/v1/chat/completions
         │                            │─────────────────────────────▶│
         │                            │                              │
         │                            │   Headers:                   │
         │                            │   • Authorization: Bearer    │
         │                            │   • HTTP-Referer: app-url    │
         │                            │   • X-Title: DRX             │
         │                            │                              │
         │                            │   Body:                      │
         │                            │   {                          │
         │                            │     "model": "google/gemini.."│
         │                            │     "messages": [...]        │
         │                            │     "temperature": 0.7       │
         │                            │     "max_tokens": 4096       │
         │                            │   }                          │
         │                            │                              │
         │                            │◀─────────────────────────────│
         │                            │   Streaming response         │
         │                            │                              │
         │◀───────────────────────────│   Parsed response            │
         │                            │                              │


   Supported Models:
   ┌────────────────────────────────────────────────────────────────────────┐
   │  Model ID                        │  Context  │  Use Case              │
   ├──────────────────────────────────┼───────────┼────────────────────────┤
   │  google/gemini-2.0-flash-exp     │   1M      │  Default, fast         │
   │  google/gemini-2.5-pro-preview   │   1M      │  Complex reasoning     │
   │  anthropic/claude-3.5-sonnet     │  200K     │  High quality          │
   │  deepseek/deepseek-r1            │  128K     │  Reasoning tasks       │
   │  openai/gpt-4o                   │  128K     │  General purpose       │
   └────────────────────────────────────────────────────────────────────────┘
```

### Tavily Search Integration

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Tavily Search Integration                             │
└─────────────────────────────────────────────────────────────────────────────────┘

     Searcher Agent              Tavily Tool                   Tavily API
          │                          │                              │
          │  search(query)           │                              │
          │─────────────────────────▶│                              │
          │                          │                              │
          │                          │  POST /search                │
          │                          │─────────────────────────────▶│
          │                          │                              │
          │                          │  {                           │
          │                          │    "query": "...",           │
          │                          │    "search_depth": "advanced"│
          │                          │    "max_results": 10,        │
          │                          │    "include_raw_content": true│
          │                          │  }                           │
          │                          │                              │
          │                          │◀─────────────────────────────│
          │                          │  { "results": [...] }        │
          │                          │                              │
          │◀─────────────────────────│                              │
          │  Parsed SearchResults    │                              │


   SearchResult Schema:
   ┌────────────────────────────────────────────────────────────────┐
   │  {                                                             │
   │    "url": "https://...",                                       │
   │    "title": "Article Title",                                   │
   │    "content": "Extracted content...",                          │
   │    "raw_content": "Full HTML...",                              │
   │    "score": 0.95,                                              │
   │    "published_date": "2024-01-15"                              │
   │  }                                                             │
   └────────────────────────────────────────────────────────────────┘
```

---

## Instrumentation & Observability

### Tracing Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Tracing Architecture                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

   Application Layer                 OpenTelemetry               Phoenix Collector
         │                                │                            │
         │                                │                            │
   ┌─────┴─────┐                          │                            │
   │  Request  │                          │                            │
   │  Arrives  │                          │                            │
   └─────┬─────┘                          │                            │
         │                                │                            │
         ▼                                │                            │
   ┌───────────────────┐                  │                            │
   │  Create Root Span │                  │                            │
   │  "research_session"│                 │                            │
   └─────────┬─────────┘                  │                            │
             │                            │                            │
             ▼                            │                            │
   ┌───────────────────┐                  │                            │
   │  Agent Execution  │                  │                            │
   │                   │                  │                            │
   │  ┌─────────────┐  │                  │                            │
   │  │ Child Span  │  │   Export Span    │                            │
   │  │ "planner"   │──┼─────────────────▶│                            │
   │  └─────────────┘  │                  │                            │
   │                   │                  │     OTLP/gRPC              │
   │  ┌─────────────┐  │                  │────────────────────────────▶
   │  │ Child Span  │  │                  │                            │
   │  │ "llm_call"  │──┼─────────────────▶│                            │
   │  └─────────────┘  │                  │                            │
   │                   │                  │                            │
   │  ┌─────────────┐  │                  │                            │
   │  │ Child Span  │  │                  │                            │
   │  │ "tool_use"  │──┼─────────────────▶│                            │
   │  └─────────────┘  │                  │                            │
   └───────────────────┘                  │                            │
                                          │                            │
                                          │                     ┌──────┴──────┐
                                          │                     │   Phoenix   │
                                          │                     │   Storage   │
                                          │                     │             │
                                          │                     │  • Traces   │
                                          │                     │  • Spans    │
                                          │                     │  • Metrics  │
                                          │                     └─────────────┘


   Span Hierarchy:
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │  research_session (root)                                                     │
   │  ├── planner_agent                                                          │
   │  │   ├── llm_call (model: gemini-flash)                                     │
   │  │   └── state_update                                                       │
   │  ├── searcher_agent                                                         │
   │  │   ├── llm_call (query expansion)                                         │
   │  │   ├── tool_call (tavily_search)                                          │
   │  │   └── tool_call (rag_retrieve)                                           │
   │  ├── reader_agent[0]                                                        │
   │  │   ├── tool_call (html_parse)                                             │
   │  │   └── llm_call (extraction)                                              │
   │  ├── reader_agent[1]                                                        │
   │  │   └── ...                                                                │
   │  ├── synthesizer_agent                                                      │
   │  │   └── llm_call (synthesis)                                               │
   │  ├── critic_agent                                                           │
   │  │   └── llm_call (evaluation)                                              │
   │  └── reporter_agent                                                         │
   │      └── llm_call (report generation)                                       │
   └─────────────────────────────────────────────────────────────────────────────┘
```

### Metrics Collection

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Metrics Collection                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                          Real-time Metrics (Redis)                           │
   │                                                                              │
   │  Agent Health:                        Rate Metrics:                          │
   │  ┌──────────────────────────────┐    ┌──────────────────────────────┐       │
   │  │ drx:agent:{id}:health        │    │ drx:agent:{id}:metrics       │       │
   │  │ • status: healthy|degraded   │    │ • tokens_1m: 45000           │       │
   │  │ • last_check: timestamp      │    │ • tokens_5m: 180000          │       │
   │  │ • failure_count: 0           │    │ • latency_p50: 1200          │       │
   │  └──────────────────────────────┘    │ • latency_p99: 4500          │       │
   │                                       │ • error_rate: 0.02           │       │
   │  Circuit Status:                      └──────────────────────────────┘       │
   │  ┌──────────────────────────────┐                                           │
   │  │ drx:agent:{id}:circuit       │    Invocations (Sorted Set):              │
   │  │ • state: closed              │    ┌──────────────────────────────┐       │
   │  │ • opened_at: null            │    │ drx:agent:{id}:invocations   │       │
   │  └──────────────────────────────┘    │ • (timestamp, {tokens, ms})  │       │
   │                                       └──────────────────────────────┘       │
   └─────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                        Persistent Metrics (PostgreSQL)                       │
   │                                                                              │
   │  research_sessions:                   tool_invocations:                      │
   │  ┌──────────────────────────────┐    ┌──────────────────────────────┐       │
   │  │ • tokens_used: 125000        │    │ • tool_name: tavily_search   │       │
   │  │ • cost_usd: 0.0125           │    │ • latency_ms: 1250           │       │
   │  │ • latency_ms: 45000          │    │ • tokens_used: 500           │       │
   │  │ • iteration_count: 3         │    │ • success: true              │       │
   │  └──────────────────────────────┘    └──────────────────────────────┘       │
   │                                                                              │
   │  agent_invocations:                   policy_violations:                     │
   │  ┌──────────────────────────────┐    ┌──────────────────────────────┐       │
   │  │ • agent_type: searcher       │    │ • violation_type: domain     │       │
   │  │ • input_tokens: 2500         │    │ • severity: warning          │       │
   │  │ • output_tokens: 1500        │    │ • blocked: true              │       │
   │  │ • latency_ms: 3200           │    │ • agent_id: searcher_v1      │       │
   │  └──────────────────────────────┘    └──────────────────────────────┘       │
   └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Pipeline

### Evaluation Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Evaluation Workflow                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                          CI/CD Pipeline Trigger                              │
   │                                                                              │
   │  GitHub Actions                                                              │
   │  ┌───────────────────────────────────────────────────────────────────────┐  │
   │  │  on:                                                                   │  │
   │  │    push: [main, develop]                                               │  │
   │  │    pull_request: [main]                                                │  │
   │  └───────────────────────────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                           Test Execution                                     │
   │                                                                              │
   │  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
   │  │   Unit Tests      │  │ Integration Tests │  │  Evaluation Tests │       │
   │  │                   │  │                   │  │                   │       │
   │  │ pytest tests/unit │  │pytest tests/integ │  │pytest ci/eval     │       │
   │  └─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘       │
   │            │                      │                      │                  │
   │            └──────────────────────┴──────────────────────┘                  │
   │                                   │                                          │
   └───────────────────────────────────┼──────────────────────────────────────────┘
                                       │
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                         Evaluation Framework                                 │
   │                                                                              │
   │  ┌─────────────────────────────────────────────────────────────────────┐   │
   │  │                         DeepEval Metrics                             │   │
   │  │                                                                      │   │
   │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
   │  │  │ Faithfulness │  │   Answer     │  │ Hallucination│               │   │
   │  │  │    Metric    │  │  Relevancy   │  │    Metric    │               │   │
   │  │  │              │  │   Metric     │  │              │               │   │
   │  │  │ Threshold:   │  │ Threshold:   │  │ Threshold:   │               │   │
   │  │  │   >= 0.8     │  │   >= 0.7     │  │   <= 0.2     │               │   │
   │  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
   │  └─────────────────────────────────────────────────────────────────────┘   │
   │                                                                              │
   │  ┌─────────────────────────────────────────────────────────────────────┐   │
   │  │                          Ragas Metrics                               │   │
   │  │                                                                      │   │
   │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
   │  │  │   Context    │  │   Context    │  │   Answer     │               │   │
   │  │  │  Precision   │  │   Recall     │  │ Correctness  │               │   │
   │  │  │              │  │              │  │              │               │   │
   │  │  │ Threshold:   │  │ Threshold:   │  │ Threshold:   │               │   │
   │  │  │   >= 0.6     │  │   >= 0.6     │  │   >= 0.7     │               │   │
   │  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
   │  └─────────────────────────────────────────────────────────────────────┘   │
   │                                                                              │
   │  ┌─────────────────────────────────────────────────────────────────────┐   │
   │  │                      Metadata Compliance                             │   │
   │  │                                                                      │   │
   │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
   │  │  │   Budget     │  │   Domain     │  │  Rate Limit  │               │   │
   │  │  │ Compliance   │  │ Compliance   │  │  Compliance  │               │   │
   │  │  │              │  │              │  │              │               │   │
   │  │  │ spend <=     │  │ no blocked   │  │ within       │               │   │
   │  │  │ max_budget   │  │ domains      │  │ rpm/tpm      │               │   │
   │  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
   │  └─────────────────────────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                            Gate Decision                                     │
   │                                                                              │
   │  ┌───────────────────────────────────────────────────────────────────────┐  │
   │  │                        Threshold Checks                                │  │
   │  │                                                                        │  │
   │  │   Hard Gates (block deployment):        Soft Gates (warn only):        │  │
   │  │   • Faithfulness >= 0.8                 • Answer Relevancy >= 0.7      │  │
   │  │   • Hallucination <= 0.2                • Context Precision >= 0.6     │  │
   │  │   • Policy Violations == 0                                             │  │
   │  │   • Task Completion >= 0.7                                             │  │
   │  └───────────────────────────────────────────────────────────────────────┘  │
   │                                                                              │
   │                         ┌─────────────────────┐                             │
   │                         │   All Gates Pass?   │                             │
   │                         └──────────┬──────────┘                             │
   │                                    │                                         │
   │                    ┌───────────────┴───────────────┐                        │
   │                    │                               │                         │
   │                   YES                             NO                         │
   │                    │                               │                         │
   │                    ▼                               ▼                         │
   │           ┌────────────────┐             ┌────────────────┐                 │
   │           │ Deploy Allowed │             │ Deploy Blocked │                 │
   │           └────────────────┘             └────────────────┘                 │
   └─────────────────────────────────────────────────────────────────────────────┘
```

### Phoenix Integration

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Phoenix Dashboard                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │  Phoenix UI (http://localhost:6006)                                          │
   │                                                                              │
   │  ┌─────────────────────────────────────────────────────────────────────┐   │
   │  │                        Trace Explorer                                │   │
   │  │                                                                      │   │
   │  │  Session: int_abc123                                                 │   │
   │  │  Duration: 45.2s                                                     │   │
   │  │  Tokens: 125,000                                                     │   │
   │  │  Cost: $0.0125                                                       │   │
   │  │                                                                      │   │
   │  │  ┌──────────────────────────────────────────────────────────────┐   │   │
   │  │  │  Timeline View                                               │   │   │
   │  │  │                                                              │   │   │
   │  │  │  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ planner (8.2s)    │   │   │
   │  │  │          ████████████░░░░░░░░░░░░░░░░░░░░ searcher (12.1s)  │   │   │
   │  │  │                      ██████████░░░░░░░░░░ reader[0] (9.5s)  │   │   │
   │  │  │                      ██████░░░░░░░░░░░░░░ reader[1] (6.2s)  │   │   │
   │  │  │                                ████████░░ synthesizer (7.8s)│   │   │
   │  │  │                                        ██ critic (2.1s)     │   │   │
   │  │  │                                          ████ reporter (4.3s)   │   │
   │  │  └──────────────────────────────────────────────────────────────┘   │   │
   │  └─────────────────────────────────────────────────────────────────────┘   │
   │                                                                              │
   │  ┌─────────────────────────────────────────────────────────────────────┐   │
   │  │                       Evaluation Results                             │   │
   │  │                                                                      │   │
   │  │  ┌────────────────┬──────────┬────────────┐                         │   │
   │  │  │    Metric      │  Score   │  Threshold │                         │   │
   │  │  ├────────────────┼──────────┼────────────┤                         │   │
   │  │  │ Faithfulness   │   0.92   │   >= 0.8   │ ✅                      │   │
   │  │  │ Hallucination  │   0.08   │   <= 0.2   │ ✅                      │   │
   │  │  │ Relevancy      │   0.88   │   >= 0.7   │ ✅                      │   │
   │  │  │ Completeness   │   0.85   │   >= 0.7   │ ✅                      │   │
   │  │  └────────────────┴──────────┴────────────┘                         │   │
   │  └─────────────────────────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────────────────────────┘
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
| **Search** | Tavily | - | Web search |
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
