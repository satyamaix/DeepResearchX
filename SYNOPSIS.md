# DRX: Deep Research Orchestration

## Synopsis

We built a self-hosted, multi-agent research system that outperforms consumer deep research products on transparency, control, and auditability.

### The Problem

Existing deep research tools (Gemini, Perplexity, OpenAI) are black boxes. You can't inspect the reasoning, checkpoint long-running tasks, replay failures, or customize the pipeline. For regulated industries—finance, healthcare, legal—this is unacceptable.

### Our Solution

DRX orchestrates six specialized agents (Planner → Searcher → Reader → Synthesizer → Critic → Reporter) in a directed acyclic graph with:

- **Full observability**: Every LLM call traced via Phoenix/OpenTelemetry
- **Checkpoint/resume**: PostgreSQL-backed state persistence for hour-long research tasks
- **Iterative refinement**: Critic agent identifies gaps, loops back until coverage threshold met
- **Knowledge graphs**: Entity-Relation-Claim structures with provenance tracking
- **Model agnostic**: Any LLM via OpenRouter (Gemini, Claude, GPT, DeepSeek)

### Technical Stack

LangGraph for cyclic workflows, FastAPI with SSE streaming, pgvector for RAG, Celery for async execution, DeepEval/Ragas for quality gates.

### Key Metric

85% implementation coverage against our architectural specification, with 10 production-ready core features including citation verification, bias detection, and multi-format export.

### What's Next

Agent registry with capability-based routing, RLHF pipeline from traces, and benchmark integration against HLE/BrowseComp.

---

*Three commands to run it. Full trace of every decision. Your infrastructure, your data.*
