-- =============================================================================
-- DRX Deep Research Platform - PostgreSQL Schema
-- Version: 1.0.0
-- PostgreSQL 16+ with pgvector 0.8.1
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector 0.8.1+

-- =============================================================================
-- CUSTOM TYPES & ENUMS
-- =============================================================================

-- Session status lifecycle
CREATE TYPE session_status AS ENUM (
    'pending',      -- Created but not started
    'running',      -- Active research in progress
    'paused',       -- User-paused or system-paused
    'completed',    -- Successfully finished
    'failed',       -- Terminal failure
    'cancelled'     -- User-cancelled
);

-- Research step status
CREATE TYPE step_status AS ENUM (
    'pending',
    'queued',
    'running',
    'succeeded',
    'failed',
    'skipped',
    'retrying'
);

-- DAG node status
CREATE TYPE dag_node_status AS ENUM (
    'pending',
    'ready',        -- All dependencies satisfied
    'running',
    'succeeded',
    'failed',
    'cancelled'
);

-- Agent types in the system
CREATE TYPE agent_type AS ENUM (
    'orchestrator',
    'planner',
    'searcher',
    'reader',
    'reasoner',
    'writer',
    'critic',
    'synthesizer',
    'reporter'
);

-- Step types for research workflow
CREATE TYPE step_type AS ENUM (
    'query_analysis',
    'plan_generation',
    'search_execution',
    'content_extraction',
    'reasoning',
    'synthesis',
    'quality_check',
    'final_output'
);

-- Policy violation severity
CREATE TYPE violation_severity AS ENUM (
    'info',
    'warning',
    'error',
    'critical'
);

-- =============================================================================
-- CORE TABLES - Part 1: Research Sessions & Steps
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Table: research_sessions
-- Purpose: Root entity for each deep research request
-- -----------------------------------------------------------------------------
CREATE TABLE research_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- User context
    user_id VARCHAR(255) NOT NULL,
    organization_id VARCHAR(255),

    -- Research query and configuration
    query TEXT NOT NULL,
    query_hash VARCHAR(64) GENERATED ALWAYS AS (encode(sha256(query::bytea), 'hex')) STORED,

    -- Steerability parameters (user-adjustable research behavior)
    steerability JSONB NOT NULL DEFAULT '{
        "depth": "standard",
        "breadth": "balanced",
        "source_preference": "academic",
        "time_budget_seconds": 300,
        "max_sources": 50,
        "language": "en",
        "include_images": false,
        "citation_style": "apa"
    }'::jsonb,

    -- Execution state
    status session_status NOT NULL DEFAULT 'pending',
    progress_pct SMALLINT DEFAULT 0 CHECK (progress_pct >= 0 AND progress_pct <= 100),

    -- Resource tracking
    total_tokens_used INTEGER DEFAULT 0,
    total_cost_usd NUMERIC(10, 6) DEFAULT 0,
    total_latency_ms INTEGER DEFAULT 0,

    -- Results
    final_output JSONB,
    sources_cited JSONB DEFAULT '[]'::jsonb,

    -- Error handling
    error_message TEXT,
    error_code VARCHAR(50),
    retry_count SMALLINT DEFAULT 0,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    tags TEXT[] DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '30 days'),

    -- Constraints
    CONSTRAINT valid_progress CHECK (progress_pct >= 0 AND progress_pct <= 100),
    CONSTRAINT valid_tokens CHECK (total_tokens_used >= 0),
    CONSTRAINT valid_cost CHECK (total_cost_usd >= 0)
);

-- Indexes for research_sessions
CREATE INDEX idx_sessions_user_id ON research_sessions(user_id);
CREATE INDEX idx_sessions_org_id ON research_sessions(organization_id) WHERE organization_id IS NOT NULL;
CREATE INDEX idx_sessions_status ON research_sessions(status);
CREATE INDEX idx_sessions_created_at ON research_sessions(created_at DESC);
CREATE INDEX idx_sessions_query_hash ON research_sessions(query_hash);
CREATE INDEX idx_sessions_expires_at ON research_sessions(expires_at) WHERE status NOT IN ('completed', 'failed', 'cancelled');
CREATE INDEX idx_sessions_steerability ON research_sessions USING GIN (steerability jsonb_path_ops);
CREATE INDEX idx_sessions_metadata ON research_sessions USING GIN (metadata jsonb_path_ops);
CREATE INDEX idx_sessions_tags ON research_sessions USING GIN (tags);

-- Partial index for active sessions (hot path optimization)
CREATE INDEX idx_sessions_active ON research_sessions(user_id, created_at DESC)
    WHERE status IN ('pending', 'running', 'paused');

-- -----------------------------------------------------------------------------
-- Table: research_steps
-- Purpose: Individual steps within a research session (tree structure)
-- -----------------------------------------------------------------------------
CREATE TABLE research_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,

    -- Hierarchical structure (self-referential for step tree)
    parent_step_id UUID REFERENCES research_steps(id) ON DELETE SET NULL,
    step_order INTEGER NOT NULL DEFAULT 0,
    depth INTEGER NOT NULL DEFAULT 0,
    path LTREE,  -- Requires ltree extension for hierarchical queries

    -- Step classification
    step_type step_type NOT NULL,
    agent_type agent_type NOT NULL,
    step_name VARCHAR(255),

    -- Execution state
    status step_status NOT NULL DEFAULT 'pending',

    -- Input/Output data
    inputs JSONB NOT NULL DEFAULT '{}'::jsonb,
    outputs JSONB DEFAULT '{}'::jsonb,

    -- Intermediate state (for resumability)
    checkpoint_data JSONB,

    -- Quality metrics
    quality_score NUMERIC(4, 3) CHECK (quality_score >= 0 AND quality_score <= 1),
    confidence_score NUMERIC(4, 3) CHECK (confidence_score >= 0 AND confidence_score <= 1),

    -- Resource consumption
    tokens_used INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_usd NUMERIC(10, 6) DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,

    -- Retry handling
    attempt_number SMALLINT DEFAULT 1,
    max_attempts SMALLINT DEFAULT 3,
    last_error TEXT,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- OpenTelemetry trace context
    trace_id VARCHAR(32),
    span_id VARCHAR(16),
    parent_span_id VARCHAR(16),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT valid_quality CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)),
    CONSTRAINT valid_confidence CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)),
    CONSTRAINT valid_tokens CHECK (tokens_used >= 0),
    CONSTRAINT valid_latency CHECK (latency_ms >= 0)
);

-- Enable ltree extension for hierarchical path queries
CREATE EXTENSION IF NOT EXISTS ltree;

-- Indexes for research_steps
CREATE INDEX idx_steps_session_id ON research_steps(session_id);
CREATE INDEX idx_steps_parent_id ON research_steps(parent_step_id) WHERE parent_step_id IS NOT NULL;
CREATE INDEX idx_steps_status ON research_steps(status);
CREATE INDEX idx_steps_agent_type ON research_steps(agent_type);
CREATE INDEX idx_steps_step_type ON research_steps(step_type);
CREATE INDEX idx_steps_trace_id ON research_steps(trace_id) WHERE trace_id IS NOT NULL;
CREATE INDEX idx_steps_created_at ON research_steps(created_at DESC);
CREATE INDEX idx_steps_inputs ON research_steps USING GIN (inputs jsonb_path_ops);
CREATE INDEX idx_steps_outputs ON research_steps USING GIN (outputs jsonb_path_ops);
CREATE INDEX idx_steps_metadata ON research_steps USING GIN (metadata jsonb_path_ops);
CREATE INDEX idx_steps_path ON research_steps USING GIST (path);

-- Composite index for session step traversal
CREATE INDEX idx_steps_session_order ON research_steps(session_id, step_order, depth);

-- -----------------------------------------------------------------------------
-- Table: dag_nodes
-- Purpose: DAG representation for parallel execution planning
-- -----------------------------------------------------------------------------
CREATE TABLE dag_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,

    -- Node identification
    node_name VARCHAR(255) NOT NULL,
    node_type step_type NOT NULL,

    -- DAG structure
    dependencies UUID[] DEFAULT '{}',  -- Array of dag_node IDs this depends on
    dependents UUID[] DEFAULT '{}',    -- Nodes that depend on this (denormalized for perf)

    -- Execution assignment
    assigned_agent agent_type,
    assigned_worker VARCHAR(255),
    priority SMALLINT DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),

    -- State
    status dag_node_status NOT NULL DEFAULT 'pending',

    -- Data flow
    inputs JSONB NOT NULL DEFAULT '{}'::jsonb,
    outputs JSONB DEFAULT '{}'::jsonb,

    -- Execution metrics
    estimated_duration_ms INTEGER,
    actual_duration_ms INTEGER,
    tokens_used INTEGER DEFAULT 0,

    -- Retry configuration
    retry_policy JSONB DEFAULT '{
        "max_retries": 3,
        "backoff_multiplier": 2,
        "initial_delay_ms": 1000,
        "max_delay_ms": 30000
    }'::jsonb,
    current_retry INTEGER DEFAULT 0,

    -- Error state
    error_message TEXT,
    error_type VARCHAR(100),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    scheduled_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT valid_priority CHECK (priority >= 1 AND priority <= 10),
    CONSTRAINT no_self_dependency CHECK (NOT (id = ANY(dependencies)))
);

-- Indexes for dag_nodes
CREATE INDEX idx_dag_session_id ON dag_nodes(session_id);
CREATE INDEX idx_dag_status ON dag_nodes(status);
CREATE INDEX idx_dag_node_type ON dag_nodes(node_type);
CREATE INDEX idx_dag_assigned_agent ON dag_nodes(assigned_agent) WHERE assigned_agent IS NOT NULL;
CREATE INDEX idx_dag_priority ON dag_nodes(priority DESC, created_at ASC) WHERE status IN ('pending', 'ready');
CREATE INDEX idx_dag_dependencies ON dag_nodes USING GIN (dependencies);
CREATE INDEX idx_dag_dependents ON dag_nodes USING GIN (dependents);
CREATE INDEX idx_dag_inputs ON dag_nodes USING GIN (inputs jsonb_path_ops);
CREATE INDEX idx_dag_outputs ON dag_nodes USING GIN (outputs jsonb_path_ops);

-- Composite index for ready node selection (hot path)
CREATE INDEX idx_dag_ready_nodes ON dag_nodes(session_id, priority DESC, created_at ASC)
    WHERE status = 'ready';

-- =============================================================================
-- AGENT TABLES - Part 2: Registry, Tools, Policies
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Table: agent_registry
-- Purpose: Configuration and capabilities for each agent type
-- -----------------------------------------------------------------------------
CREATE TABLE agent_registry (
    id VARCHAR(100) PRIMARY KEY,  -- e.g., 'searcher_v1', 'reasoner_v2'

    -- Versioning
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    is_active BOOLEAN NOT NULL DEFAULT true,
    is_default BOOLEAN DEFAULT false,

    -- Agent classification
    agent_type agent_type NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Capabilities
    capabilities TEXT[] NOT NULL DEFAULT '{}',
    supported_step_types step_type[] DEFAULT '{}',

    -- Domain restrictions
    allowed_domains TEXT[] DEFAULT '{}',  -- Empty = all allowed
    blocked_domains TEXT[] DEFAULT '{}',

    -- Resource limits
    max_budget_usd NUMERIC(10, 4) DEFAULT 1.00,
    max_tokens_per_call INTEGER DEFAULT 4096,
    max_concurrent_calls INTEGER DEFAULT 5,
    timeout_seconds INTEGER DEFAULT 60,

    -- Model configuration
    model_config JSONB NOT NULL DEFAULT '{
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 0.95
    }'::jsonb,

    -- Tool permissions
    allowed_tools TEXT[] DEFAULT '{}',
    tool_config JSONB DEFAULT '{}'::jsonb,

    -- Rate limiting
    rate_limit_rpm INTEGER DEFAULT 60,  -- Requests per minute
    rate_limit_tpm INTEGER DEFAULT 100000,  -- Tokens per minute

    -- Health tracking
    health_status VARCHAR(20) DEFAULT 'healthy',
    last_health_check TIMESTAMPTZ,
    failure_count INTEGER DEFAULT 0,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deprecated_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT valid_budget CHECK (max_budget_usd > 0),
    CONSTRAINT valid_timeout CHECK (timeout_seconds > 0),
    CONSTRAINT unique_default_per_type UNIQUE (agent_type, is_default)
);

-- Indexes for agent_registry
CREATE INDEX idx_agent_type ON agent_registry(agent_type);
CREATE INDEX idx_agent_active ON agent_registry(is_active) WHERE is_active = true;
CREATE INDEX idx_agent_capabilities ON agent_registry USING GIN (capabilities);
CREATE INDEX idx_agent_allowed_tools ON agent_registry USING GIN (allowed_tools);
CREATE INDEX idx_agent_model_config ON agent_registry USING GIN (model_config jsonb_path_ops);

-- Partial index for default agents lookup
CREATE UNIQUE INDEX idx_agent_default ON agent_registry(agent_type)
    WHERE is_active = true AND is_default = true;

-- -----------------------------------------------------------------------------
-- Table: tool_invocations
-- Purpose: Audit log for all tool calls with timing and token usage
-- -----------------------------------------------------------------------------
CREATE TABLE tool_invocations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Context references
    session_id UUID REFERENCES research_sessions(id) ON DELETE SET NULL,
    step_id UUID REFERENCES research_steps(id) ON DELETE SET NULL,
    dag_node_id UUID REFERENCES dag_nodes(id) ON DELETE SET NULL,
    agent_id VARCHAR(100) REFERENCES agent_registry(id) ON DELETE SET NULL,

    -- Tool identification
    tool_name VARCHAR(255) NOT NULL,
    tool_version VARCHAR(20),

    -- Invocation details
    invocation_id VARCHAR(100) UNIQUE,  -- External correlation ID

    -- Input/Output (potentially large, consider separate table for production)
    input_params JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_result JSONB,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, running, succeeded, failed
    error_message TEXT,
    error_code VARCHAR(50),

    -- Resource metrics
    tokens_used INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_usd NUMERIC(10, 6) DEFAULT 0,

    -- Timing
    latency_ms INTEGER,
    queue_time_ms INTEGER,
    execution_time_ms INTEGER,

    -- Rate limiting context
    rate_limit_bucket VARCHAR(255),
    rate_limit_remaining INTEGER,

    -- OpenTelemetry context
    trace_id VARCHAR(32),
    span_id VARCHAR(16),
    parent_span_id VARCHAR(16),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT valid_invocation_tokens CHECK (tokens_used >= 0),
    CONSTRAINT valid_invocation_cost CHECK (cost_usd >= 0)
);

-- Indexes for tool_invocations
CREATE INDEX idx_tool_session_id ON tool_invocations(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_tool_step_id ON tool_invocations(step_id) WHERE step_id IS NOT NULL;
CREATE INDEX idx_tool_agent_id ON tool_invocations(agent_id) WHERE agent_id IS NOT NULL;
CREATE INDEX idx_tool_name ON tool_invocations(tool_name);
CREATE INDEX idx_tool_status ON tool_invocations(status);
CREATE INDEX idx_tool_created_at ON tool_invocations(created_at DESC);
CREATE INDEX idx_tool_trace_id ON tool_invocations(trace_id) WHERE trace_id IS NOT NULL;
CREATE INDEX idx_tool_invocation_id ON tool_invocations(invocation_id) WHERE invocation_id IS NOT NULL;
CREATE INDEX idx_tool_input_params ON tool_invocations USING GIN (input_params jsonb_path_ops);

-- Time-based partitioning hint (implement in production)
-- PARTITION BY RANGE (created_at);

-- -----------------------------------------------------------------------------
-- Table: policy_violations
-- Purpose: Track compliance violations per session/step
-- -----------------------------------------------------------------------------
CREATE TABLE policy_violations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Context references
    session_id UUID REFERENCES research_sessions(id) ON DELETE CASCADE,
    step_id UUID REFERENCES research_steps(id) ON DELETE SET NULL,
    agent_id VARCHAR(100) REFERENCES agent_registry(id) ON DELETE SET NULL,
    tool_invocation_id UUID REFERENCES tool_invocations(id) ON DELETE SET NULL,

    -- Violation classification
    violation_type VARCHAR(100) NOT NULL,
    violation_code VARCHAR(50) NOT NULL,
    severity violation_severity NOT NULL DEFAULT 'warning',

    -- Violation details
    description TEXT NOT NULL,
    policy_name VARCHAR(255),
    policy_version VARCHAR(20),

    -- Context data
    context JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- What triggered the violation
    trigger_action VARCHAR(255),
    trigger_input JSONB,

    -- Resolution
    resolution_status VARCHAR(20) DEFAULT 'unresolved',  -- unresolved, acknowledged, resolved, false_positive
    resolution_notes TEXT,
    resolved_by VARCHAR(255),
    resolved_at TIMESTAMPTZ,

    -- Impact assessment
    blocked_execution BOOLEAN DEFAULT false,
    required_human_review BOOLEAN DEFAULT false,

    -- OpenTelemetry context
    trace_id VARCHAR(32),
    span_id VARCHAR(16),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_resolution CHECK (
        resolution_status IN ('unresolved', 'acknowledged', 'resolved', 'false_positive')
    )
);

-- Indexes for policy_violations
CREATE INDEX idx_violations_session_id ON policy_violations(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_violations_step_id ON policy_violations(step_id) WHERE step_id IS NOT NULL;
CREATE INDEX idx_violations_agent_id ON policy_violations(agent_id) WHERE agent_id IS NOT NULL;
CREATE INDEX idx_violations_type ON policy_violations(violation_type);
CREATE INDEX idx_violations_severity ON policy_violations(severity);
CREATE INDEX idx_violations_created_at ON policy_violations(created_at DESC);
CREATE INDEX idx_violations_unresolved ON policy_violations(session_id, created_at DESC)
    WHERE resolution_status = 'unresolved';
CREATE INDEX idx_violations_context ON policy_violations USING GIN (context jsonb_path_ops);

-- =============================================================================
-- VECTOR TABLES - Part 3: Document Storage & Embeddings
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Table: document_chunks
-- Purpose: Vector storage with embeddings for semantic search
-- -----------------------------------------------------------------------------
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Session context (optional - chunks can be shared across sessions)
    session_id UUID REFERENCES research_sessions(id) ON DELETE SET NULL,
    step_id UUID REFERENCES research_steps(id) ON DELETE SET NULL,

    -- Source identification
    source_url TEXT,
    source_domain VARCHAR(255),
    source_type VARCHAR(50),  -- webpage, pdf, api, database
    source_hash VARCHAR(64),  -- SHA256 of source content for dedup

    -- Document structure
    document_id UUID,  -- Groups chunks from same document
    chunk_index INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER,

    -- Content
    content TEXT NOT NULL,
    content_hash VARCHAR(64) GENERATED ALWAYS AS (encode(sha256(content::bytea), 'hex')) STORED,

    -- Embedding vector (OpenAI ada-002 compatible: 1536 dimensions)
    embedding vector(1536),

    -- Alternative embedding for multi-model support
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-ada-002',
    embedding_version VARCHAR(20),

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Extracted entities and keywords
    entities JSONB DEFAULT '[]'::jsonb,
    keywords TEXT[] DEFAULT '{}',

    -- Quality signals
    relevance_score NUMERIC(4, 3),
    quality_score NUMERIC(4, 3),
    freshness_score NUMERIC(4, 3),

    -- Source credibility
    source_credibility NUMERIC(4, 3),
    citation_count INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fetched_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT valid_chunk_index CHECK (chunk_index >= 0),
    CONSTRAINT valid_relevance CHECK (relevance_score IS NULL OR (relevance_score >= 0 AND relevance_score <= 1)),
    CONSTRAINT valid_quality CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1))
);

-- B-tree indexes for document_chunks
CREATE INDEX idx_chunks_session_id ON document_chunks(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_chunks_document_id ON document_chunks(document_id) WHERE document_id IS NOT NULL;
CREATE INDEX idx_chunks_source_hash ON document_chunks(source_hash) WHERE source_hash IS NOT NULL;
CREATE INDEX idx_chunks_content_hash ON document_chunks(content_hash);
CREATE INDEX idx_chunks_source_domain ON document_chunks(source_domain) WHERE source_domain IS NOT NULL;
CREATE INDEX idx_chunks_created_at ON document_chunks(created_at DESC);
CREATE INDEX idx_chunks_expires_at ON document_chunks(expires_at) WHERE expires_at IS NOT NULL;

-- GIN indexes for JSONB and array columns
CREATE INDEX idx_chunks_metadata ON document_chunks USING GIN (metadata jsonb_path_ops);
CREATE INDEX idx_chunks_entities ON document_chunks USING GIN (entities jsonb_path_ops);
CREATE INDEX idx_chunks_keywords ON document_chunks USING GIN (keywords);

-- IVFFlat index for vector similarity search
-- Note: Create after loading data for optimal list count
-- lists = sqrt(row_count) is a good starting point
CREATE INDEX idx_chunks_embedding_ivfflat ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- HNSW index alternative (better recall, more memory)
-- Uncomment if you prefer HNSW over IVFFlat
-- CREATE INDEX idx_chunks_embedding_hnsw ON document_chunks
--     USING hnsw (embedding vector_cosine_ops)
--     WITH (m = 16, ef_construction = 64);

-- -----------------------------------------------------------------------------
-- Table: document_sources
-- Purpose: Track original document sources for citation
-- -----------------------------------------------------------------------------
CREATE TABLE document_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Source identification
    url TEXT UNIQUE,
    url_hash VARCHAR(64) GENERATED ALWAYS AS (encode(sha256(url::bytea), 'hex')) STORED,
    domain VARCHAR(255),

    -- Content metadata
    title TEXT,
    author TEXT,
    publisher TEXT,
    published_date DATE,

    -- Source classification
    source_type VARCHAR(50),  -- academic, news, blog, official, social
    content_type VARCHAR(50),  -- article, paper, report, documentation
    language VARCHAR(10) DEFAULT 'en',

    -- Credibility metrics
    credibility_score NUMERIC(4, 3),
    domain_authority NUMERIC(4, 3),
    citation_count INTEGER DEFAULT 0,

    -- Access information
    is_paywalled BOOLEAN DEFAULT false,
    requires_auth BOOLEAN DEFAULT false,

    -- Raw content storage
    raw_content TEXT,
    raw_content_hash VARCHAR(64),
    content_length INTEGER,

    -- Fetch metadata
    last_fetched_at TIMESTAMPTZ,
    fetch_status VARCHAR(20),
    http_status INTEGER,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for document_sources
CREATE INDEX idx_sources_url_hash ON document_sources(url_hash);
CREATE INDEX idx_sources_domain ON document_sources(domain);
CREATE INDEX idx_sources_source_type ON document_sources(source_type) WHERE source_type IS NOT NULL;
CREATE INDEX idx_sources_credibility ON document_sources(credibility_score DESC) WHERE credibility_score IS NOT NULL;
CREATE INDEX idx_sources_metadata ON document_sources USING GIN (metadata jsonb_path_ops);

-- =============================================================================
-- FUNCTIONS & TRIGGERS
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Function: update_updated_at()
-- Purpose: Auto-update updated_at timestamp on row modification
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
CREATE TRIGGER trg_sessions_updated_at
    BEFORE UPDATE ON research_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_agent_registry_updated_at
    BEFORE UPDATE ON agent_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_document_sources_updated_at
    BEFORE UPDATE ON document_sources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- -----------------------------------------------------------------------------
-- Function: cleanup_old_sessions()
-- Purpose: Remove expired sessions and related data for data retention
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION cleanup_old_sessions(
    retention_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    sessions_deleted BIGINT,
    steps_deleted BIGINT,
    chunks_deleted BIGINT,
    invocations_deleted BIGINT
) AS $$
DECLARE
    cutoff_date TIMESTAMPTZ;
    v_sessions_deleted BIGINT := 0;
    v_steps_deleted BIGINT := 0;
    v_chunks_deleted BIGINT := 0;
    v_invocations_deleted BIGINT := 0;
    batch_size INTEGER := 1000;
    deleted_count INTEGER;
BEGIN
    cutoff_date := NOW() - (retention_days || ' days')::INTERVAL;

    -- Delete in batches to avoid long locks
    LOOP
        -- Delete tool invocations for expired sessions
        WITH deleted AS (
            DELETE FROM tool_invocations
            WHERE session_id IN (
                SELECT id FROM research_sessions
                WHERE expires_at < NOW()
                   OR (created_at < cutoff_date AND status IN ('completed', 'failed', 'cancelled'))
                LIMIT batch_size
            )
            RETURNING 1
        )
        SELECT COUNT(*) INTO deleted_count FROM deleted;
        v_invocations_deleted := v_invocations_deleted + deleted_count;

        EXIT WHEN deleted_count = 0;
    END LOOP;

    -- Delete document chunks
    LOOP
        WITH deleted AS (
            DELETE FROM document_chunks
            WHERE session_id IN (
                SELECT id FROM research_sessions
                WHERE expires_at < NOW()
                   OR (created_at < cutoff_date AND status IN ('completed', 'failed', 'cancelled'))
                LIMIT batch_size
            )
            RETURNING 1
        )
        SELECT COUNT(*) INTO deleted_count FROM deleted;
        v_chunks_deleted := v_chunks_deleted + deleted_count;

        EXIT WHEN deleted_count = 0;
    END LOOP;

    -- Delete research steps (cascades from sessions, but explicit for counting)
    LOOP
        WITH deleted AS (
            DELETE FROM research_steps
            WHERE session_id IN (
                SELECT id FROM research_sessions
                WHERE expires_at < NOW()
                   OR (created_at < cutoff_date AND status IN ('completed', 'failed', 'cancelled'))
                LIMIT batch_size
            )
            RETURNING 1
        )
        SELECT COUNT(*) INTO deleted_count FROM deleted;
        v_steps_deleted := v_steps_deleted + deleted_count;

        EXIT WHEN deleted_count = 0;
    END LOOP;

    -- Finally delete sessions
    LOOP
        WITH deleted AS (
            DELETE FROM research_sessions
            WHERE expires_at < NOW()
               OR (created_at < cutoff_date AND status IN ('completed', 'failed', 'cancelled'))
            LIMIT batch_size
            RETURNING 1
        )
        SELECT COUNT(*) INTO deleted_count FROM deleted;
        v_sessions_deleted := v_sessions_deleted + deleted_count;

        EXIT WHEN deleted_count = 0;
    END LOOP;

    RETURN QUERY SELECT v_sessions_deleted, v_steps_deleted, v_chunks_deleted, v_invocations_deleted;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- Function: get_ready_dag_nodes()
-- Purpose: Find DAG nodes ready for execution (all dependencies satisfied)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_ready_dag_nodes(
    p_session_id UUID,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    node_id UUID,
    node_name VARCHAR(255),
    node_type step_type,
    assigned_agent agent_type,
    priority SMALLINT,
    inputs JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH dependency_status AS (
        SELECT
            dn.id,
            dn.dependencies,
            COALESCE(
                bool_and(dep.status = 'succeeded'),
                true  -- No dependencies means ready
            ) as deps_satisfied
        FROM dag_nodes dn
        LEFT JOIN LATERAL unnest(dn.dependencies) AS dep_id ON true
        LEFT JOIN dag_nodes dep ON dep.id = dep_id
        WHERE dn.session_id = p_session_id
          AND dn.status = 'pending'
        GROUP BY dn.id, dn.dependencies
    )
    SELECT
        dn.id,
        dn.node_name,
        dn.node_type,
        dn.assigned_agent,
        dn.priority,
        dn.inputs
    FROM dag_nodes dn
    JOIN dependency_status ds ON ds.id = dn.id
    WHERE ds.deps_satisfied = true
    ORDER BY dn.priority DESC, dn.created_at ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- Function: vector_search()
-- Purpose: Semantic similarity search with filtering
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION vector_search(
    query_embedding vector(1536),
    p_session_id UUID DEFAULT NULL,
    p_limit INTEGER DEFAULT 10,
    p_min_score NUMERIC DEFAULT 0.7,
    p_source_types TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    chunk_id UUID,
    content TEXT,
    source_url TEXT,
    similarity_score NUMERIC,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.content,
        dc.source_url,
        (1 - (dc.embedding <=> query_embedding))::NUMERIC as sim_score,
        dc.metadata
    FROM document_chunks dc
    WHERE
        (p_session_id IS NULL OR dc.session_id = p_session_id)
        AND dc.embedding IS NOT NULL
        AND (p_source_types IS NULL OR dc.source_type = ANY(p_source_types))
        AND (1 - (dc.embedding <=> query_embedding)) >= p_min_score
    ORDER BY dc.embedding <=> query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- Function: calculate_session_metrics()
-- Purpose: Aggregate metrics for a research session
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calculate_session_metrics(p_session_id UUID)
RETURNS TABLE (
    total_steps INTEGER,
    completed_steps INTEGER,
    failed_steps INTEGER,
    total_tokens INTEGER,
    total_cost NUMERIC,
    total_latency INTEGER,
    avg_quality_score NUMERIC,
    progress_pct INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as total_steps,
        COUNT(*) FILTER (WHERE rs.status = 'succeeded')::INTEGER as completed_steps,
        COUNT(*) FILTER (WHERE rs.status = 'failed')::INTEGER as failed_steps,
        COALESCE(SUM(rs.tokens_used), 0)::INTEGER as total_tokens,
        COALESCE(SUM(rs.cost_usd), 0)::NUMERIC as total_cost,
        COALESCE(SUM(rs.latency_ms), 0)::INTEGER as total_latency,
        COALESCE(AVG(rs.quality_score), 0)::NUMERIC as avg_quality_score,
        CASE
            WHEN COUNT(*) = 0 THEN 0
            ELSE ((COUNT(*) FILTER (WHERE rs.status IN ('succeeded', 'skipped'))::NUMERIC / COUNT(*)::NUMERIC) * 100)::INTEGER
        END as progress_pct
    FROM research_steps rs
    WHERE rs.session_id = p_session_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SEED DATA - Default Agent Registry
-- =============================================================================

INSERT INTO agent_registry (id, agent_type, version, display_name, description, capabilities, allowed_tools, model_config, is_default)
VALUES
    ('orchestrator_v1', 'orchestrator', '1.0.0', 'Research Orchestrator',
     'Coordinates research workflow and manages agent interactions',
     ARRAY['workflow_management', 'task_delegation', 'state_management'],
     ARRAY['dag_execute', 'agent_invoke', 'session_manage'],
     '{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 4096}'::jsonb,
     true),

    ('planner_v1', 'planner', '1.0.0', 'Research Planner',
     'Decomposes research queries into executable plans',
     ARRAY['query_analysis', 'plan_generation', 'strategy_selection'],
     ARRAY['query_parse', 'plan_create'],
     '{"model": "claude-sonnet-4-20250514", "temperature": 0.5, "max_tokens": 8192}'::jsonb,
     true),

    ('searcher_v1', 'searcher', '1.0.0', 'Web Searcher',
     'Executes web searches and retrieves relevant sources',
     ARRAY['web_search', 'source_discovery', 'query_expansion'],
     ARRAY['web_search', 'url_fetch', 'search_expand'],
     '{"model": "claude-sonnet-4-20250514", "temperature": 0.2, "max_tokens": 2048}'::jsonb,
     true),

    ('reader_v1', 'reader', '1.0.0', 'Content Reader',
     'Extracts and processes content from web sources',
     ARRAY['content_extraction', 'text_processing', 'entity_recognition'],
     ARRAY['url_fetch', 'html_parse', 'pdf_extract'],
     '{"model": "claude-sonnet-4-20250514", "temperature": 0.1, "max_tokens": 16384}'::jsonb,
     true),

    ('reasoner_v1', 'reasoner', '1.0.0', 'Research Reasoner',
     'Performs logical reasoning and analysis on gathered information',
     ARRAY['logical_reasoning', 'fact_checking', 'inference'],
     ARRAY['knowledge_query', 'fact_verify'],
     '{"model": "claude-sonnet-4-20250514", "temperature": 0.4, "max_tokens": 8192}'::jsonb,
     true),

    ('writer_v1', 'writer', '1.0.0', 'Report Writer',
     'Generates structured research reports and summaries',
     ARRAY['report_generation', 'summarization', 'citation_formatting'],
     ARRAY['report_create', 'citation_format'],
     '{"model": "claude-sonnet-4-20250514", "temperature": 0.6, "max_tokens": 16384}'::jsonb,
     true),

    ('critic_v1', 'critic', '1.0.0', 'Quality Critic',
     'Reviews and critiques research output for quality',
     ARRAY['quality_assessment', 'bias_detection', 'completeness_check'],
     ARRAY['quality_score', 'bias_detect'],
     '{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 4096}'::jsonb,
     true),

    ('synthesizer_v1', 'synthesizer', '1.0.0', 'Knowledge Synthesizer',
     'Combines information from multiple sources into coherent output',
     ARRAY['information_synthesis', 'conflict_resolution', 'knowledge_integration'],
     ARRAY['synthesize', 'merge_findings'],
     '{"model": "claude-sonnet-4-20250514", "temperature": 0.5, "max_tokens": 8192}'::jsonb,
     true)
ON CONFLICT (id) DO NOTHING;

-- =============================================================================
-- ROW-LEVEL SECURITY (Optional - Enable for multi-tenant deployments)
-- =============================================================================

-- Enable RLS on research_sessions
-- ALTER TABLE research_sessions ENABLE ROW LEVEL SECURITY;

-- Policy for user access
-- CREATE POLICY user_sessions_policy ON research_sessions
--     FOR ALL
--     USING (user_id = current_setting('app.current_user_id', true));

-- =============================================================================
-- MAINTENANCE COMMANDS (Run periodically)
-- =============================================================================

-- Analyze tables for query optimizer
-- ANALYZE research_sessions;
-- ANALYZE research_steps;
-- ANALYZE dag_nodes;
-- ANALYZE document_chunks;

-- Reindex vector indexes (after significant data changes)
-- REINDEX INDEX idx_chunks_embedding_ivfflat;

-- Vacuum to reclaim space
-- VACUUM (VERBOSE, ANALYZE) research_sessions;
-- VACUUM (VERBOSE, ANALYZE) document_chunks;

COMMENT ON SCHEMA public IS 'DRX Deep Research Platform - Database Schema v1.0.0';
