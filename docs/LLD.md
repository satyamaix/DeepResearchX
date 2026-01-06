# DRX Low-Level Design (LLD) Document

## Table of Contents

1. [Class Diagrams](#class-diagrams)
2. [Knowledge Graph Classes](#knowledge-graph-classes)
3. [Tool Classes](#tool-classes)
4. [Service Classes](#service-classes)
5. [Database Design](#database-design)
6. [Sequence Diagrams](#sequence-diagrams)
7. [API Specifications](#api-specifications)
8. [Error Handling](#error-handling)
9. [Configuration Schema](#configuration-schema)

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
        -_map_dependencies() DependencyGraph
    }

    class SearcherAgent {
        +_process() AgentResponse
        -_expand_query(query: str) list~str~
        -_search(queries: list) list~SearchResult~
        -_dedupe(results: list) list
        -_rag_retrieve(query: str) list~Document~
    }

    class ReaderAgent {
        +_process() AgentResponse
        -_fetch_content(url: str) str
        -_extract(content: str) Finding
        -_parse_html(html: str) Document
        -_extract_pdf(source: str) ExtractedDocument
        -_extract_entities() list~Entity~
    }

    class SynthesizerAgent {
        +_process() AgentResponse
        -_aggregate(findings: list) Synthesis
        -_resolve_conflicts(findings: list) list
        -_build_argument(synthesis: Synthesis) ArgumentGraph
        -_build_knowledge_graph() KnowledgeGraph
    }

    class CriticAgent {
        +_process() AgentResponse
        -_evaluate(synthesis: str) QualityScore
        -_find_gaps(synthesis: str) list~str~
        -_score(report: str) float
        -_verify_citations(citations: list) list~VerificationResult~
        -_detect_bias(findings: list) BiasReport
    }

    class ReporterAgent {
        +_process() AgentResponse
        -_generate(synthesis: str) Report
        -_format(report: Report, fmt: str) str
        -_cite(report: Report, citations: list) Report
        -_export_html() str
        -_export_pdf() bytes
    }

    BaseAgent <|-- PlannerAgent
    BaseAgent <|-- SearcherAgent
    BaseAgent <|-- ReaderAgent
    BaseAgent <|-- SynthesizerAgent
    BaseAgent <|-- CriticAgent
    BaseAgent <|-- ReporterAgent
```

### State Classes (TypedDict for LangGraph)

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
        +knowledge_graph: dict
        +gaps: list~str~
        +policy_violations: list~PolicyViolation~
        +final_report: str | None
        +iteration_count: int
        +token_budget: int
        +tokens_used: int
        +cost_usd: float
        +metrics: AgentMetrics
    }

    class ResearchPlan {
        <<TypedDict>>
        +dag_nodes: list~SubTask~
        +current_iteration: int
        +max_iterations: int
        +coverage_score: float
        +execution_order: list~str~
        +dependency_graph: dict~str, list~str~~
    }

    class SubTask {
        <<TypedDict>>
        +id: str
        +description: str
        +agent_type: str
        +dependencies: list~str~
        +status: Literal~pending,running,completed,failed~
        +inputs: dict~str, Any~
        +outputs: dict~str, Any~ | None
        +quality_score: float | None
        +retry_count: int
    }

    class Finding {
        <<TypedDict>>
        +id: str
        +content: str
        +source_url: str
        +relevance_score: float
        +extracted_entities: list~str~
        +claims: list~str~
        +metadata: dict~str, Any~
    }

    class CitationRecord {
        <<TypedDict>>
        +url: str
        +title: str
        +snippet: str
        +relevance_score: float
        +retrieved_at: str
        +verification_status: str | None
    }

    class SteerabilityConfig {
        <<TypedDict>>
        +tone: str
        +format: str
        +max_sources: int
        +focus_areas: list~str~
        +exclude_topics: list~str~
        +preferred_domains: list~str~
        +language: str
    }

    AgentState *-- ResearchPlan
    AgentState *-- Finding
    AgentState *-- CitationRecord
    AgentState *-- SteerabilityConfig
    ResearchPlan *-- SubTask
```

---

## Knowledge Graph Classes

### Entity-Relation-Claim Model

```mermaid
classDiagram
    class Entity {
        <<TypedDict>>
        +id: str
        +name: str
        +entity_type: EntityType
        +properties: dict~str, Any~
        +embedding: list~float~ | None
        +source_ids: list~str~
        +created_at: str
    }

    class Relation {
        <<TypedDict>>
        +id: str
        +source_entity_id: str
        +target_entity_id: str
        +relation_type: str
        +confidence: float
        +evidence: str
        +created_at: str
    }

    class Claim {
        <<TypedDict>>
        +id: str
        +statement: str
        +supporting_entity_ids: list~str~
        +evidence_ids: list~str~
        +confidence: float
        +status: ClaimStatus
        +created_at: str
    }

    class EntityType {
        <<enumeration>>
        PERSON
        ORGANIZATION
        CONCEPT
        EVENT
        LOCATION
        DOCUMENT
    }

    class ClaimStatus {
        <<enumeration>>
        SUPPORTED
        CONTESTED
        REFUTED
        UNVERIFIED
    }

    class KnowledgeGraph {
        -entities: dict~str, Entity~
        -relations: dict~str, Relation~
        -claims: dict~str, Claim~
        -entity_index: dict~str, list~str~~
        +add_entity(entity: Entity) str
        +add_relation(relation: Relation) str
        +add_claim(claim: Claim) str
        +get_entity(entity_id: str) Entity | None
        +get_relation(relation_id: str) Relation | None
        +get_claim(claim_id: str) Claim | None
        +get_related_entities(entity_id: str, relation_type: str, depth: int) list~Entity~
        +get_claims_for_entity(entity_id: str) list~Claim~
        +update_claim_status(claim_id: str, status: ClaimStatus) bool
        +query_similar(query: str, k: int) list~Entity~
        +get_subgraph(entity_id: str, depth: int) dict
        +export_cytoscape() dict
        +export_jsonld() dict
        +export_mermaid() str
        +to_dict() dict
        +from_dict(data: dict)$ KnowledgeGraph
    }

    Entity --> EntityType
    Claim --> ClaimStatus
    KnowledgeGraph --> Entity
    KnowledgeGraph --> Relation
    KnowledgeGraph --> Claim
```

### Cytoscape Export Schema

```mermaid
classDiagram
    class CytoscapeExport {
        <<TypedDict>>
        +elements: CytoscapeElements
        +style: list~CytoscapeStyle~
        +layout: dict~str, Any~
    }

    class CytoscapeElements {
        <<TypedDict>>
        +nodes: list~CytoscapeNode~
        +edges: list~CytoscapeEdge~
    }

    class CytoscapeNode {
        <<TypedDict>>
        +data: NodeData
        +position: Position | None
    }

    class NodeData {
        <<TypedDict>>
        +id: str
        +label: str
        +type: str
        +nodeType: str
        +weight: int
        +confidence: float | None
        +status: str | None
        +sources: list~dict~ | None
    }

    class CytoscapeEdge {
        <<TypedDict>>
        +data: EdgeData
    }

    class EdgeData {
        <<TypedDict>>
        +id: str
        +source: str
        +target: str
        +label: str
        +confidence: float
    }

    CytoscapeExport --> CytoscapeElements
    CytoscapeElements --> CytoscapeNode
    CytoscapeElements --> CytoscapeEdge
    CytoscapeNode --> NodeData
    CytoscapeEdge --> EdgeData
```

---

## Tool Classes

### Base Tool Interface

```mermaid
classDiagram
    class BaseTool {
        <<abstract>>
        +name: str*
        +description: str*
        +execute(input: Any)* ToolResult
        +get_schema() dict
    }

    class ToolResult {
        <<TypedDict>>
        +status: ToolStatus
        +data: Any
        +error: str | None
        +metadata: dict~str, Any~
    }

    class ToolStatus {
        <<enumeration>>
        SUCCESS
        ERROR
        RATE_LIMITED
        TIMEOUT
    }

    BaseTool --> ToolResult
    ToolResult --> ToolStatus
```

### PDF Extractor

```mermaid
classDiagram
    class PDFExtractor {
        -timeout: int
        -max_pages: int
        +extract(source: str | bytes) ExtractedDocument
        +extract_tables(pdf_bytes: bytes) list~Table~
        +get_metadata(pdf_bytes: bytes) PDFMetadata
        -_fetch_pdf(url: str) bytes
        -_extract_text_pypdf(pdf_bytes: bytes) list~PageContent~
    }

    class ExtractedDocument {
        <<TypedDict>>
        +text: str
        +pages: list~PageContent~
        +tables: list~Table~
        +metadata: PDFMetadata
        +extraction_method: str
    }

    class PageContent {
        <<TypedDict>>
        +page_number: int
        +text: str
        +tables: list~Table~
    }

    class Table {
        <<TypedDict>>
        +page_number: int
        +headers: list~str~
        +rows: list~list~str~~
        +raw_text: str
    }

    class PDFMetadata {
        <<TypedDict>>
        +title: str | None
        +author: str | None
        +creation_date: str | None
        +page_count: int
        +file_size: int | None
    }

    PDFExtractor --> ExtractedDocument
    ExtractedDocument --> PageContent
    ExtractedDocument --> Table
    ExtractedDocument --> PDFMetadata
```

### Citation Verifier

```mermaid
classDiagram
    class CitationVerifier {
        -http_client: AsyncClient
        -timeout: int
        -similarity_threshold: float
        +verify_url_accessible(url: str) URLStatus
        +verify_quote(quote: str, source_content: str, threshold: float) QuoteVerification
        +batch_verify(citations: list~CitationRecord~) list~VerificationResult~
        -_fetch_content(url: str) str | None
        -_fuzzy_match(quote: str, content: str) tuple~float, str~
    }

    class URLStatus {
        <<TypedDict>>
        +url: str
        +accessible: bool
        +status_code: int | None
        +redirect_url: str | None
        +error: str | None
        +checked_at: str
    }

    class QuoteVerification {
        <<TypedDict>>
        +found: bool
        +similarity: float
        +best_match: str | None
        +position: int | None
    }

    class VerificationResult {
        <<TypedDict>>
        +citation_id: str
        +url: str
        +url_accessible: bool
        +url_status_code: int | None
        +quote_found: bool
        +quote_similarity: float
        +issues: list~str~
        +verified_at: str
    }

    CitationVerifier --> URLStatus
    CitationVerifier --> QuoteVerification
    CitationVerifier --> VerificationResult
```

### Bias Detector

```mermaid
classDiagram
    class BiasDetector {
        -political_keywords: dict~str, list~str~~
        -commercial_patterns: list~str~
        -sensational_patterns: list~str~
        +analyze_diversity(citations: list~CitationRecord~) DiversityReport
        +detect_indicators(content: str) list~BiasIndicator~
        +assess_viewpoints(findings: list~Finding~) ViewpointAssessment
        +generate_report(citations: list, findings: list) BiasReport
        -_calculate_entropy(distribution: dict) float
        -_extract_domain(url: str) str
        -_classify_source_type(url: str) str
    }

    class DiversityReport {
        <<TypedDict>>
        +domain_diversity: float
        +source_type_diversity: float
        +geographic_diversity: float
        +temporal_diversity: float
        +domain_distribution: dict~str, int~
        +source_type_distribution: dict~str, int~
        +recommendations: list~str~
    }

    class BiasIndicator {
        <<TypedDict>>
        +indicator_type: BiasType
        +description: str
        +severity: SeverityLevel
        +evidence: str
        +confidence: float
    }

    class BiasType {
        <<enumeration>>
        POLITICAL
        COMMERCIAL
        SENSATIONAL
        SELECTIVE
        CONFIRMATION
    }

    class SeverityLevel {
        <<enumeration>>
        LOW
        MEDIUM
        HIGH
    }

    class ViewpointAssessment {
        <<TypedDict>>
        +viewpoints_found: list~str~
        +balance_score: float
        +dominant_viewpoint: str | None
        +missing_perspectives: list~str~
    }

    class BiasReport {
        <<TypedDict>>
        +diversity: DiversityReport
        +indicators: list~BiasIndicator~
        +viewpoint_assessment: ViewpointAssessment
        +overall_bias_score: float
        +recommendations: list~str~
        +generated_at: str
    }

    BiasDetector --> DiversityReport
    BiasDetector --> BiasIndicator
    BiasDetector --> ViewpointAssessment
    BiasDetector --> BiasReport
    BiasIndicator --> BiasType
    BiasIndicator --> SeverityLevel
```

---

## Service Classes

### Parallel Executor

```mermaid
classDiagram
    class ParallelExecutor {
        -max_concurrent: int
        -timeout: float
        +execute_ready_tasks(plan: ResearchPlan, state: AgentState, agents: dict) AggregatedResult
        +get_ready_tasks(plan: ResearchPlan) list~SubTask~
        +resolve_dependencies(plan: ResearchPlan) DependencyGraph
        +fan_out(tasks: list~SubTask~, state: AgentState) list~Coroutine~
        +fan_in(results: list~TaskResult~) AggregatedResult
        -_topological_sort(graph: DependencyGraph) list~list~str~~
    }

    class DependencyGraph {
        <<TypedDict>>
        +nodes: dict~str, SubTask~
        +edges: dict~str, list~str~~
        +in_degree: dict~str, int~
    }

    class TaskResult {
        <<TypedDict>>
        +task_id: str
        +status: str
        +output: dict~str, Any~ | None
        +error: str | None
        +duration_ms: int
        +tokens_used: int
    }

    class AggregatedResult {
        <<TypedDict>>
        +completed_tasks: list~str~
        +failed_tasks: list~str~
        +merged_findings: list~Finding~
        +merged_citations: list~CitationRecord~
        +total_tokens: int
        +total_duration_ms: int
    }

    ParallelExecutor --> DependencyGraph
    ParallelExecutor --> TaskResult
    ParallelExecutor --> AggregatedResult
```

### Budget Tracker

```mermaid
classDiagram
    class BudgetTracker {
        -token_budget: int
        -cost_budget: float | None
        -tokens_used: int
        -cost_used: float
        -model_costs: dict~str, ModelCost~
        +track(model: str, input_tokens: int, output_tokens: int) None
        +estimate(model: str, estimated_tokens: int) float
        +can_afford(model: str, estimated_tokens: int) bool
        +enforce() None
        +status: BudgetStatus
        +remaining_tokens: int
        +remaining_cost: float | None
    }

    class ModelCost {
        <<TypedDict>>
        +input: float
        +output: float
    }

    class BudgetStatus {
        <<TypedDict>>
        +tokens_used: int
        +tokens_remaining: int
        +cost_used: float
        +cost_remaining: float | None
        +percentage_used: float
        +is_exceeded: bool
    }

    class BudgetExceededError {
        +budget_type: str
        +used: float
        +limit: float
        +message: str
    }

    BudgetTracker --> ModelCost
    BudgetTracker --> BudgetStatus
    BudgetTracker --> BudgetExceededError
```

### Vector Store

```mermaid
classDiagram
    class VectorStore {
        -pool: AsyncConnectionPool
        -embedding_provider: EmbeddingProvider
        -default_collection: str
        +create_collection(name: str, dimension: int) None
        +ingest(documents: list~Document~, collection: str) int
        +search(query: str, collection: str, k: int, filters: dict) list~SearchResult~
        +hybrid_search(query: str, collection: str, k: int, alpha: float) list~SearchResult~
        +delete_collection(name: str) bool
        -_generate_embedding(text: str) list~float~
        -_semantic_search(embedding: list, k: int) list~tuple~
        -_keyword_search(query: str, k: int) list~tuple~
        -_rrf_fusion(semantic_results: list, keyword_results: list, k: int) list~SearchResult~
    }

    class Document {
        <<TypedDict>>
        +id: str
        +content: str
        +metadata: dict~str, Any~
        +embedding: list~float~ | None
    }

    class SearchResult {
        <<TypedDict>>
        +id: str
        +content: str
        +score: float
        +metadata: dict~str, Any~
        +source: str
    }

    class EmbeddingProvider {
        <<abstract>>
        +embed(text: str)* list~float~
        +embed_batch(texts: list~str~)* list~list~float~~
        +dimension: int*
    }

    class OpenRouterEmbeddingProvider {
        -client: OpenRouterClient
        -model: str
        +embed(text: str) list~float~
        +embed_batch(texts: list) list
    }

    VectorStore --> Document
    VectorStore --> SearchResult
    VectorStore --> EmbeddingProvider
    EmbeddingProvider <|-- OpenRouterEmbeddingProvider
```

### Report Exporter

```mermaid
classDiagram
    class ReportExporter {
        -template_dir: Path
        -jinja_env: Environment
        +to_html(state: AgentState, template: str, include_graph: bool) str
        +to_pdf(state: AgentState, template: str) bytes
        +to_json(state: AgentState) str
        +render_knowledge_graph_svg(kg: KnowledgeGraph) str
        -_build_context(state: AgentState) ReportContext
        -_render_template(template: str, context: ReportContext) str
    }

    class ReportContext {
        <<TypedDict>>
        +query: str
        +synthesis: str
        +findings: list~Finding~
        +citations: list~CitationRecord~
        +knowledge_graph: dict
        +quality_metrics: QualityMetrics
        +generated_at: str
        +iteration_count: int
        +tokens_used: int
    }

    class QualityMetrics {
        <<TypedDict>>
        +coverage_score: float
        +citation_count: int
        +source_diversity: float
        +bias_score: float | None
    }

    ReportExporter --> ReportContext
    ReportContext --> QualityMetrics
```

### Progress Publisher

```mermaid
classDiagram
    class ProgressPublisher {
        -redis: Redis
        -session_id: str
        -channel: str
        +publish(event: StreamEvent) None
        +publish_progress(agent: str, message: str, progress: float) None
        +publish_dag_state(dag: dict) None
        +publish_complete(result: dict) None
        +publish_error(error: str) None
        +check_cancellation() bool
        +get_channel() str
    }

    class StreamEvent {
        <<TypedDict>>
        +type: StreamEventType
        +data: dict~str, Any~
        +timestamp: str
        +sequence: int
    }

    class StreamEventType {
        <<enumeration>>
        INTERACTION_START
        THOUGHT_SUMMARY
        CONTENT_DELTA
        TOOL_USE
        TOOL_RESULT
        DAG_STATE
        CHECKPOINT
        ERROR
        INTERACTION_COMPLETE
    }

    ProgressPublisher --> StreamEvent
    StreamEvent --> StreamEventType
```

### Dataset Collector

```mermaid
classDiagram
    class DatasetCollector {
        -storage_path: Path
        +record_session(session_id: str, query: str, report: str, citations: list, metrics: dict, ...) str
        +add_feedback(session_id: str, rating: int, feedback: str, labels: list) bool
        +export_training_set(min_rating: int, format: str, output_path: Path) Path
        +get_statistics() DatasetStatistics
        -_classify_tier(rating: int) QualityTier
        -_format_for_jsonl(record: SessionRecord) dict
        -_format_for_huggingface(record: SessionRecord) dict
        -_format_for_openai(record: SessionRecord) dict
    }

    class SessionRecord {
        <<TypedDict>>
        +session_id: str
        +query: str
        +final_report: str
        +citations: list~dict~
        +user_rating: int | None
        +user_feedback: str | None
        +quality_metrics: dict~str, float~
        +bias_report: dict | None
        +duration_seconds: float
        +token_count: int
        +model_used: str
        +created_at: str
        +feedback_labels: list~str~
    }

    class QualityTier {
        <<enumeration>>
        GOLD
        SILVER
        BRONZE
        UNRATED
    }

    class DatasetStatistics {
        <<TypedDict>>
        +total_sessions: int
        +rated_sessions: int
        +average_rating: float
        +rating_distribution: dict~int, int~
        +tier_distribution: dict~str, int~
        +label_counts: dict~str, int~
    }

    DatasetCollector --> SessionRecord
    DatasetCollector --> QualityTier
    DatasetCollector --> DatasetStatistics
```

### Feedback Store

```mermaid
classDiagram
    class FeedbackStore {
        -redis: Redis
        -db_pool: AsyncConnectionPool | None
        +submit_feedback(session_id: str, rating: int, comment: str, labels: list, user_id: str) str
        +get_feedback(session_id: str) list~FeedbackRecord~
        +get_aggregate_metrics() AggregateMetrics
        +get_feedback_by_rating(min_rating: int, max_rating: int) list~FeedbackRecord~
        +delete_feedback(feedback_id: str) bool
        -_cache_feedback(feedback: FeedbackRecord) None
        -_update_aggregate_metrics(feedback: FeedbackRecord) None
    }

    class FeedbackRecord {
        <<TypedDict>>
        +feedback_id: str
        +session_id: str
        +user_id: str | None
        +rating: int
        +comment: str | None
        +labels: list~str~
        +created_at: str
        +metadata: dict~str, Any~
    }

    class AggregateMetrics {
        <<TypedDict>>
        +total_feedback: int
        +average_rating: float
        +rating_distribution: dict~int, int~
        +common_labels: list~tuple~str, int~~
        +feedback_rate: float
    }

    FeedbackStore --> FeedbackRecord
    FeedbackStore --> AggregateMetrics
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
    RESEARCH_SESSIONS ||--o{ FEEDBACK : receives
    RESEARCH_SESSIONS ||--o{ ENTITIES : contains
    RESEARCH_SESSIONS ||--o{ CLAIMS : contains

    RESEARCH_STEPS ||--o{ TOOL_INVOCATIONS : performs
    RESEARCH_STEPS ||--o{ AGENT_INVOCATIONS : executes
    RESEARCH_STEPS ||--o| RESEARCH_STEPS : parent

    ENTITIES ||--o{ RELATIONS : source
    ENTITIES ||--o{ RELATIONS : target

    RESEARCH_SESSIONS {
        uuid id PK
        uuid user_id
        text query
        jsonb steerability
        varchar status
        jsonb config
        int tokens_used
        decimal cost_usd
        jsonb knowledge_graph
        jsonb final_report
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
        int tokens_used
        int latency_ms
        timestamp created_at
        timestamp completed_at
    }

    ENTITIES {
        uuid id PK
        uuid session_id FK
        varchar name
        enum entity_type
        jsonb properties
        vector embedding
        text[] source_ids
        timestamp created_at
    }

    RELATIONS {
        uuid id PK
        uuid session_id FK
        uuid source_entity_id FK
        uuid target_entity_id FK
        varchar relation_type
        float confidence
        text evidence
        timestamp created_at
    }

    CLAIMS {
        uuid id PK
        uuid session_id FK
        text statement
        uuid[] supporting_entity_ids
        uuid[] evidence_ids
        float confidence
        enum status
        timestamp created_at
    }

    FEEDBACK {
        uuid id PK
        uuid session_id FK
        uuid user_id
        int rating
        text comment
        text[] labels
        jsonb metadata
        timestamp created_at
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

---

## Sequence Diagrams

### Parallel Research Execution

```mermaid
sequenceDiagram
    autonumber
    participant W as Worker
    participant O as Orchestrator
    participant PE as ParallelExecutor
    participant A1 as Searcher 1
    participant A2 as Searcher 2
    participant A3 as Searcher 3
    participant DB as PostgreSQL
    participant R as Redis

    W->>O: Start orchestration
    O->>PE: execute_plan_parallel(state)

    PE->>PE: resolve_dependencies()
    PE->>PE: get_ready_tasks() → [s1, s2, s3]

    par Fan-Out Search
        PE->>A1: execute(task_1)
        PE->>A2: execute(task_2)
        PE->>A3: execute(task_3)
    end

    A1-->>PE: SearchResults 1
    A2-->>PE: SearchResults 2
    A3-->>PE: SearchResults 3

    PE->>PE: fan_in(results)
    PE->>PE: deduplicate_sources()
    PE-->>O: merged_state

    O->>R: PUBLISH dag_state
    O->>DB: Save checkpoint
```

### Citation Verification Flow

```mermaid
sequenceDiagram
    participant Critic as Critic Agent
    participant CV as CitationVerifier
    participant HTTP as HTTP Client
    participant Fuzzy as Fuzzy Matcher

    Critic->>CV: batch_verify(citations)

    loop For each citation
        CV->>HTTP: HEAD request (url)
        HTTP-->>CV: status_code
        CV->>CV: url_accessible = (status < 400)

        alt URL accessible
            CV->>HTTP: GET content
            HTTP-->>CV: page_content
            CV->>Fuzzy: fuzzy_match(quote, content, 0.85)
            Fuzzy-->>CV: (similarity, best_match)

            alt similarity >= threshold
                CV->>CV: quote_found = true
            else similarity < threshold
                CV->>CV: quote_found = false
                CV->>CV: add_issue("Quote not found")
            end
        else URL not accessible
            CV->>CV: add_issue("URL inaccessible")
        end
    end

    CV-->>Critic: list[VerificationResult]
```

### Feedback Collection Flow

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant FS as FeedbackStore
    participant DC as DatasetCollector
    participant R as Redis
    participant DB as PostgreSQL

    U->>API: POST /sessions/{id}/feedback
    API->>API: Validate request
    API->>DB: Verify session exists

    API->>FS: submit_feedback(session_id, rating, comment, labels)

    FS->>DB: INSERT INTO feedback
    DB-->>FS: feedback_id

    FS->>R: SET feedback:{id} (cache)
    FS->>R: HINCRBY aggregate:rating:{rating} 1

    FS->>DC: record_session_feedback()
    DC->>DC: Classify quality tier
    DC->>DC: Move to appropriate tier file

    FS-->>API: feedback_id
    API-->>U: 201 Created {feedback_id}
```

### Knowledge Graph Building

```mermaid
sequenceDiagram
    participant Synth as Synthesizer
    participant KG as KnowledgeGraph
    participant Embed as EmbeddingProvider

    Synth->>KG: Initialize graph

    loop For each finding
        Synth->>Synth: Extract entities
        Synth->>Synth: Extract relations

        loop For each entity
            Synth->>Embed: embed(entity.name)
            Embed-->>Synth: embedding
            Synth->>KG: add_entity(entity)
        end

        loop For each relation
            Synth->>KG: add_relation(relation)
        end
    end

    Synth->>KG: Build claims from synthesis
    loop For each claim
        Synth->>KG: add_claim(claim)
    end

    Synth->>KG: export_cytoscape()
    KG-->>Synth: cytoscape_json
```

---

## API Specifications

### Request/Response Models

```mermaid
classDiagram
    class CreateInteractionRequest {
        +input: str
        +steerability: SteerabilityConfig | None
        +config: InteractionConfig | None
    }

    class InteractionConfig {
        +max_iterations: int
        +token_budget: int
        +timeout_seconds: int
        +enable_citations: bool
        +enable_quality_checks: bool
    }

    class InteractionResponse {
        +id: str
        +status: str
        +query: str
        +result: InteractionResult | None
        +created_at: str
        +completed_at: str | None
    }

    class InteractionResult {
        +final_report: str
        +findings: list~Finding~
        +citations: list~CitationRecord~
        +knowledge_graph: dict
        +tokens_used: int
        +cost_usd: float
        +iteration_count: int
        +quality_metrics: QualityMetrics
    }

    class FeedbackRequest {
        +rating: int
        +comment: str | None
        +labels: list~str~
    }

    class FeedbackResponse {
        +feedback_id: str
        +session_id: str
        +status: str
    }

    CreateInteractionRequest --> InteractionConfig
    InteractionResponse --> InteractionResult
```

### SSE Event Schema

```mermaid
classDiagram
    class SSEEvent {
        +event: str
        +id: str
        +data: str
    }

    class InteractionStartEvent {
        +interaction_id: str
        +status: str
        +query: str
    }

    class ThoughtSummaryEvent {
        +agent: str
        +thought: str
        +timestamp: str
    }

    class DAGStateEvent {
        +nodes: list~DAGNode~
        +edges: list~DAGEdge~
        +current_node: str | None
    }

    class ContentDeltaEvent {
        +content: str
        +agent: str
    }

    class CompleteEvent {
        +interaction_id: str
        +status: str
        +result: InteractionResult
    }

    SSEEvent <|-- InteractionStartEvent
    SSEEvent <|-- ThoughtSummaryEvent
    SSEEvent <|-- DAGStateEvent
    SSEEvent <|-- ContentDeltaEvent
    SSEEvent <|-- CompleteEvent
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
        +details: dict
    }

    class ValidationError {
        +code: VAL_*
    }

    class AgentError {
        +code: AGT_*
    }

    class PolicyError {
        +code: POL_*
    }

    class ServiceError {
        +code: SVC_*
    }

    class BudgetExceededError {
        +budget_type: str
        +used: float
        +limit: float
    }

    class CircuitOpenError {
        +agent_id: str
        +opened_at: str
    }

    class CitationVerificationError {
        +citation_id: str
        +issues: list~str~
    }

    DRXError <|-- ValidationError
    DRXError <|-- AgentError
    DRXError <|-- PolicyError
    DRXError <|-- ServiceError
    PolicyError <|-- BudgetExceededError
    AgentError <|-- CircuitOpenError
    ServiceError <|-- CitationVerificationError
```

### Error Codes

| Category | Code | Description | Recoverable |
|----------|------|-------------|-------------|
| **Validation** | VAL_001 | Invalid input schema | No |
| | VAL_002 | Missing required field | No |
| | VAL_003 | Invalid rating value | No |
| **Agent** | AGT_001 | Agent timeout | Yes |
| | AGT_002 | Circuit breaker open | Yes |
| | AGT_003 | Max iterations exceeded | No |
| | AGT_004 | Parallel execution failed | Yes |
| **Policy** | POL_001 | Domain blocked | Yes |
| | POL_002 | Token budget exceeded | Yes |
| | POL_003 | Cost budget exceeded | Yes |
| | POL_004 | Rate limit exceeded | Yes |
| **Service** | SVC_001 | LLM rate limited | Yes |
| | SVC_002 | Citation verification failed | Yes |
| | SVC_003 | PDF extraction failed | Yes |
| | SVC_004 | Embedding generation failed | Yes |

---

## Configuration Schema

### Environment Variables

```mermaid
flowchart TB
    subgraph Core["Core"]
        APP_ENV["APP_ENV"]
        DEBUG["DEBUG"]
        LOG_LEVEL["LOG_LEVEL"]
    end

    subgraph Database["Database"]
        DATABASE_URL["DATABASE_URL"]
        DB_POOL_SIZE["DB_POOL_SIZE"]
    end

    subgraph Redis["Redis"]
        REDIS_URL["REDIS_URL"]
        CELERY_BROKER_URL["CELERY_BROKER_URL"]
    end

    subgraph LLM["LLM"]
        OPENROUTER_API_KEY["OPENROUTER_API_KEY"]
        DEFAULT_MODEL["DEFAULT_MODEL"]
        REASONING_MODEL["REASONING_MODEL"]
        SEARCH_MODEL["SEARCH_MODEL"]
    end

    subgraph Budget["Budget"]
        TOKEN_BUDGET_PER_SESSION["TOKEN_BUDGET_PER_SESSION"]
        COST_BUDGET_PER_SESSION["COST_BUDGET_PER_SESSION"]
    end

    subgraph Research["Research"]
        MAX_RESEARCH_ITERATIONS["MAX_RESEARCH_ITERATIONS"]
        MAX_SOURCES_PER_QUERY["MAX_SOURCES_PER_QUERY"]
        MIN_COVERAGE_SCORE["MIN_COVERAGE_SCORE"]
    end

    subgraph Observability["Observability"]
        PHOENIX_ENABLED["PHOENIX_ENABLED"]
        PHOENIX_COLLECTOR_ENDPOINT["PHOENIX_COLLECTOR_ENDPOINT"]
        PROMETHEUS_ENABLED["PROMETHEUS_ENABLED"]
    end
```

### Model Cost Configuration

```python
MODEL_COSTS = {
    "google/gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
    "google/gemini-3-flash-preview:online": {"input": 0.075, "output": 0.30},
    "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
    "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "openai/gpt-4o": {"input": 2.5, "output": 10.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
}
```

---

## Redis Key Patterns

```mermaid
flowchart TB
    subgraph Session["Session State"]
        SessionHash["drx:session:{id}<br/>HASH: status, node, iteration"]
    end

    subgraph Events["Event Streaming"]
        EventStream["drx:events:{id}<br/>STREAM: event_type, data"]
        CancelKey["drx:cancel:{id}<br/>STRING: '1' if cancelled"]
    end

    subgraph Feedback["Feedback Cache"]
        FeedbackHash["drx:feedback:{session_id}<br/>HASH: rating, labels"]
        AggregateHash["drx:feedback:aggregate<br/>HASH: total, avg_rating"]
    end

    subgraph Metrics["Real-time Metrics"]
        AgentMetrics["drx:agent:{id}:metrics<br/>HASH: tokens, latency"]
        CircuitState["drx:agent:{id}:circuit<br/>HASH: state, failures"]
    end
```
