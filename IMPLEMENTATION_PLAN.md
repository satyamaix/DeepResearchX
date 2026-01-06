# DRX Full Coverage Implementation Plan
## Gap Closure Strategy - Topologically Sorted

**Generated**: 2026-01-06
**Baseline**: DRX.md Architectural Specification
**Current Coverage**: ~75% | **Target**: 100%

---

## Topological Dependency Graph

```
LEVEL 0 (Foundation - No Dependencies) ═══════════════════════════════════════
├── WP-0A: Knowledge Graph Models
├── WP-0B: PDF Extraction Tool
├── WP-0C: Prometheus Metrics Foundation
├── WP-0D: Citation Verification Tool
└── WP-0E: Budget Tracking Enhancement

LEVEL 1 (Core Execution) ═════════════════════════════════════════════════════
├── WP-1A: Worker Implementation ──────────► depends on WP-0E
├── WP-1B: Internal RAG Search ────────────► depends on WP-0A
└── WP-1C: Bias Detection ─────────────────► depends on WP-0D

LEVEL 2 (Advanced Features) ══════════════════════════════════════════════════
├── WP-2A: DAG Parallel Execution ─────────► depends on WP-1A
├── WP-2B: HTML/PDF Report Export ─────────► depends on WP-0A
└── WP-2C: Grafana Dashboard Templates ────► depends on WP-0C

LEVEL 3 (Integration & UI) ═══════════════════════════════════════════════════
├── WP-3A: Dataset Flywheel ───────────────► depends on WP-1C
├── WP-3B: Frontend DAG Visualization ─────► depends on WP-2A
└── WP-3C: Frontend Argument Graph ────────► depends on WP-0A, WP-2B
```

---

## LEVEL 0: Foundation Layer (No Dependencies - Run in Parallel)

### WP-0A: Knowledge Graph Models
**Priority**: P0 | **Complexity**: Medium
**Files to Create**:
- `src/models/__init__.py`
- `src/models/knowledge.py`

**Specification**:
```python
# TypedDict definitions for LangGraph compatibility
class Entity(TypedDict):
    id: str
    name: str
    entity_type: Literal["person", "org", "concept", "event", "location"]
    properties: dict[str, Any]
    embedding: list[float] | None
    source_ids: list[str]

class Relation(TypedDict):
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    confidence: float
    evidence: str

class Claim(TypedDict):
    id: str
    statement: str
    supporting_entity_ids: list[str]
    evidence_ids: list[str]
    confidence: float
    status: Literal["supported", "contested", "refuted"]

class KnowledgeGraph:
    """In-memory knowledge graph with pgvector persistence."""
    async def add_entity(self, entity: Entity) -> str
    async def add_relation(self, relation: Relation) -> str
    async def add_claim(self, claim: Claim) -> str
    async def query_similar(self, query: str, k: int = 10) -> list[Entity]
    async def get_subgraph(self, entity_id: str, depth: int = 2) -> dict
    def export_cytoscape(self) -> dict  # For frontend
    def export_jsonld(self) -> dict  # For interop
```

---

### WP-0B: PDF Extraction Tool
**Priority**: P1 | **Complexity**: Low
**Files to Create**:
- `src/tools/pdf_extractor.py`
**Files to Modify**:
- `src/agents/reader.py` (add PDF tool)
- `pyproject.toml` (add pypdf2 dependency)

**Specification**:
```python
class PDFExtractor:
    """Extract text and metadata from PDF documents."""

    async def extract(self, source: str | bytes) -> ExtractedDocument:
        """Extract from URL, file path, or bytes."""

    def extract_tables(self, pdf_bytes: bytes) -> list[Table]
    def get_metadata(self, pdf_bytes: bytes) -> PDFMetadata

class ExtractedDocument(TypedDict):
    text: str
    pages: list[PageContent]
    tables: list[Table]
    metadata: PDFMetadata
    extraction_method: str

class PDFMetadata(TypedDict):
    title: str | None
    author: str | None
    creation_date: str | None
    page_count: int
```

---

### WP-0C: Prometheus Metrics Foundation
**Priority**: P1 | **Complexity**: Medium
**Files to Create**:
- `src/observability/metrics.py`
**Files to Modify**:
- `src/api/main.py` (add /metrics endpoint)

**Specification**:
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Core metrics
TOKENS_TOTAL = Counter(
    "drx_tokens_total",
    "Total tokens consumed",
    ["model", "agent", "direction"]  # direction: input/output
)

REQUEST_LATENCY = Histogram(
    "drx_request_latency_seconds",
    "Request latency",
    ["endpoint", "method"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

SESSIONS_ACTIVE = Gauge(
    "drx_sessions_active",
    "Currently active research sessions"
)

AGENT_EXECUTIONS = Counter(
    "drx_agent_executions_total",
    "Agent execution count",
    ["agent", "status"]  # status: success/error
)

class MetricsMiddleware:
    """FastAPI middleware for automatic metrics collection."""

def get_metrics() -> bytes:
    """Return Prometheus-formatted metrics."""
    return generate_latest()
```

---

### WP-0D: Citation Verification Tool
**Priority**: P1 | **Complexity**: Medium
**Files to Create**:
- `src/tools/citation_verifier.py`
**Files to Modify**:
- `src/agents/critic.py` (integrate verification)

**Specification**:
```python
class CitationVerifier:
    """Verify citations against source content."""

    async def verify_url_accessible(self, url: str) -> URLStatus
    async def verify_quote(
        self,
        quote: str,
        source_content: str,
        threshold: float = 0.85
    ) -> QuoteVerification
    async def batch_verify(
        self,
        citations: list[CitationRecord]
    ) -> list[VerificationResult]

class VerificationResult(TypedDict):
    citation_id: str
    url_accessible: bool
    url_status_code: int | None
    quote_found: bool
    quote_similarity: float  # Fuzzy match 0-1
    issues: list[str]
    verified_at: str

class QuoteVerification(TypedDict):
    found: bool
    similarity: float
    best_match: str | None
    position: int | None  # Character offset if found
```

---

### WP-0E: Budget Tracking Enhancement
**Priority**: P0 | **Complexity**: Medium
**Files to Create**:
- `src/orchestrator/budget.py`
**Files to Modify**:
- `src/orchestrator/state.py` (add cost fields)
- `src/config.py` (add cost config)

**Specification**:
```python
# Model cost configuration (per 1M tokens)
MODEL_COSTS = {
    "google/gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
    "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
    "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
}

class BudgetTracker:
    """Track and enforce token/cost budgets."""

    def __init__(
        self,
        token_budget: int,
        cost_budget: float | None = None
    ):
        self.token_budget = token_budget
        self.cost_budget = cost_budget
        self._tokens_used = 0
        self._cost_used = 0.0

    def track(self, model: str, input_tokens: int, output_tokens: int) -> None
    def estimate(self, model: str, estimated_tokens: int) -> float
    def can_afford(self, model: str, estimated_tokens: int) -> bool
    def enforce(self) -> None  # Raises BudgetExceededError

    @property
    def status(self) -> BudgetStatus

class BudgetExceededError(Exception):
    """Raised when budget is exceeded."""
    def __init__(self, budget_type: str, used: float, limit: float): ...
```

---

## LEVEL 1: Core Execution (Depends on Level 0)

### WP-1A: Worker Implementation
**Depends on**: WP-0E
**Priority**: P0 | **Complexity**: High
**Files to Modify**:
- `src/worker.py` (complete implementation)
- `src/services/redis_client.py` (add pub/sub)

**Specification**:
```python
@app.task(bind=True, name="research.execute", max_retries=3)
def execute_research(
    self,
    session_id: str,
    query: str,
    config: dict | None = None
) -> dict:
    """
    Execute research workflow asynchronously.

    - Initializes BudgetTracker from config
    - Creates ResearchOrchestrator
    - Streams events via Redis pub/sub
    - Handles graceful cancellation
    - Persists final state
    """

    # Implementation requirements:
    # 1. Initialize budget tracker
    # 2. Create orchestrator with checkpointer
    # 3. Subscribe to cancellation channel
    # 4. Stream events to Redis channel
    # 5. Handle exceptions with retry
    # 6. Return final state summary

class ProgressPublisher:
    """Publish progress events to Redis."""

    def __init__(self, redis_client: Redis, session_id: str): ...
    async def publish(self, event: StreamEvent) -> None
    def get_channel(self) -> str
```

---

### WP-1B: Internal RAG Search
**Depends on**: WP-0A
**Priority**: P1 | **Complexity**: High
**Files to Create**:
- `src/services/vectorstore.py`
**Files to Modify**:
- `src/tools/rag_retriever.py`
- `src/agents/searcher.py`

**Specification**:
```python
class VectorStore:
    """pgvector-backed vector store for RAG."""

    def __init__(self, connection_pool: AsyncConnectionPool): ...

    async def create_collection(self, name: str, dimension: int = 1536) -> None
    async def ingest(
        self,
        documents: list[Document],
        collection: str = "default"
    ) -> int
    async def search(
        self,
        query: str,
        collection: str = "default",
        k: int = 10,
        filters: dict | None = None
    ) -> list[SearchResult]
    async def hybrid_search(
        self,
        query: str,
        collection: str = "default",
        k: int = 10,
        alpha: float = 0.5  # Balance semantic vs keyword
    ) -> list[SearchResult]

class Document(TypedDict):
    id: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None  # Generated if None
```

---

### WP-1C: Bias Detection
**Depends on**: WP-0D
**Priority**: P1 | **Complexity**: Medium
**Files to Create**:
- `src/tools/bias_detector.py`
**Files to Modify**:
- `src/agents/critic.py`

**Specification**:
```python
class BiasDetector:
    """Detect bias and assess source diversity."""

    def analyze_diversity(
        self,
        citations: list[CitationRecord]
    ) -> DiversityReport

    def detect_indicators(
        self,
        content: str
    ) -> list[BiasIndicator]

    def assess_viewpoints(
        self,
        findings: list[Finding]
    ) -> ViewpointAssessment

class DiversityReport(TypedDict):
    domain_diversity: float  # 0-1, entropy-based
    source_type_diversity: float
    geographic_diversity: float
    temporal_diversity: float
    recommendations: list[str]

class BiasIndicator(TypedDict):
    indicator_type: Literal["political", "commercial", "sensational", "selective"]
    description: str
    severity: Literal["low", "medium", "high"]
    evidence: str
```

---

## LEVEL 2: Advanced Features (Depends on Level 1)

### WP-2A: DAG Parallel Execution
**Depends on**: WP-1A
**Priority**: P0 | **Complexity**: High
**Files to Create**:
- `src/orchestrator/parallel.py`
**Files to Modify**:
- `src/orchestrator/workflow.py`
- `src/orchestrator/nodes.py`

**Specification**:
```python
class ParallelExecutor:
    """Execute DAG tasks in parallel where possible."""

    async def execute_ready_tasks(
        self,
        plan: ResearchPlan,
        state: AgentState,
        agents: dict[str, BaseAgent]
    ) -> list[TaskResult]

    def get_ready_tasks(self, plan: ResearchPlan) -> list[SubTask]
    def resolve_dependencies(self, plan: ResearchPlan) -> DependencyGraph

    async def fan_out(
        self,
        tasks: list[SubTask],
        state: AgentState
    ) -> list[Coroutine]

    async def fan_in(
        self,
        results: list[TaskResult]
    ) -> AggregatedResult

# Workflow integration - replace sequential with parallel
async def execute_plan_parallel(state: AgentState) -> AgentState:
    """Execute plan with parallel task execution."""
    executor = ParallelExecutor()
    while not is_plan_complete(state["plan"]):
        ready = executor.get_ready_tasks(state["plan"])
        results = await executor.execute_ready_tasks(...)
        state = merge_results(state, results)
    return state
```

---

### WP-2B: HTML/PDF Report Export
**Depends on**: WP-0A
**Priority**: P1 | **Complexity**: Medium
**Files to Create**:
- `src/templates/report_base.html.j2`
- `src/templates/report_default.html.j2`
- `src/services/report_exporter.py`
**Files to Modify**:
- `src/agents/reporter.py`
- `pyproject.toml` (add jinja2, weasyprint)

**Specification**:
```python
class ReportExporter:
    """Export reports to multiple formats."""

    def __init__(self, template_dir: Path = None): ...

    def to_html(
        self,
        state: AgentState,
        template: str = "default",
        include_graph: bool = True
    ) -> str

    def to_pdf(
        self,
        state: AgentState,
        template: str = "default"
    ) -> bytes

    def to_docx(self, state: AgentState) -> bytes

    def render_argument_graph_svg(
        self,
        knowledge_graph: KnowledgeGraph
    ) -> str

# Template context
class ReportContext(TypedDict):
    query: str
    synthesis: str
    findings: list[Finding]
    citations: list[CitationRecord]
    knowledge_graph: dict  # Cytoscape format
    quality_metrics: QualityMetrics
    generated_at: str
```

---

### WP-2C: Grafana Dashboard Templates
**Depends on**: WP-0C
**Priority**: P2 | **Complexity**: Low
**Files to Create**:
- `deployment/grafana/provisioning/datasources/prometheus.yaml`
- `deployment/grafana/provisioning/dashboards/dashboard.yaml`
- `deployment/grafana/dashboards/drx-overview.json`
- `deployment/grafana/dashboards/drx-agents.json`
**Files to Modify**:
- `deployment/docker-compose.prod.yaml` (add grafana service)

---

## LEVEL 3: Integration & UI (Depends on Level 2)

### WP-3A: Dataset Flywheel
**Depends on**: WP-1C
**Files to Create**:
- `ci/evaluation/dataset_collector.py`
- `ci/evaluation/feedback_store.py`
**Files to Modify**:
- `src/api/routes.py` (add feedback endpoint)

---

### WP-3B: Frontend DAG Visualization
**Depends on**: WP-2A
**Files to Create**:
- `frontend/src/components/workflow/DAGVisualization.tsx`
- `frontend/src/components/workflow/NodeDetail.tsx`
- `frontend/src/hooks/useWorkflowSSE.ts`

---

### WP-3C: Frontend Argument Graph
**Depends on**: WP-0A, WP-2B
**Files to Create**:
- `frontend/src/components/graph/ArgumentGraph.tsx`
- `frontend/src/components/graph/CitationHover.tsx`
- `frontend/src/components/graph/ClaimNode.tsx`

---

## Execution Matrix

| Level | Work Packet | Priority | Complexity | Parallel Group |
|-------|-------------|----------|------------|----------------|
| 0 | WP-0A: Knowledge Graph | P0 | Medium | Group A |
| 0 | WP-0B: PDF Extraction | P1 | Low | Group A |
| 0 | WP-0C: Prometheus Metrics | P1 | Medium | Group A |
| 0 | WP-0D: Citation Verifier | P1 | Medium | Group A |
| 0 | WP-0E: Budget Tracking | P0 | Medium | Group A |
| 1 | WP-1A: Worker Impl | P0 | High | Group B |
| 1 | WP-1B: RAG Search | P1 | High | Group B |
| 1 | WP-1C: Bias Detection | P1 | Medium | Group B |
| 2 | WP-2A: DAG Parallel | P0 | High | Group C |
| 2 | WP-2B: Report Export | P1 | Medium | Group C |
| 2 | WP-2C: Grafana | P2 | Low | Group C |
| 3 | WP-3A: Dataset Flywheel | P2 | Medium | Group D |
| 3 | WP-3B: DAG UI | P2 | Medium | Group D |
| 3 | WP-3C: Argument Graph UI | P2 | Medium | Group D |

---

## Git Tracking Strategy

Each work packet creates a feature branch:
```
main
├── feat/wp-0a-knowledge-graph
├── feat/wp-0b-pdf-extraction
├── feat/wp-0c-prometheus-metrics
├── feat/wp-0d-citation-verifier
├── feat/wp-0e-budget-tracking
└── ... (subsequent levels after merge)
```

Commit message format:
```
feat(wp-0a): implement Entity and Relation TypedDicts

- Add src/models/knowledge.py with core types
- Add KnowledgeGraph class with CRUD operations
- Add export_cytoscape() for frontend
- Add tests for graph operations

Part of: DRX Full Coverage Implementation
```
