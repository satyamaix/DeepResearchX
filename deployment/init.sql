-- =============================================================================
-- DRX Deep Research - PostgreSQL Docker Initialization Script
-- This file runs on first container startup to initialize the database
-- =============================================================================

-- Connection settings for initialization
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";       -- Cryptographic functions
CREATE EXTENSION IF NOT EXISTS "vector";          -- pgvector for embeddings

-- =============================================================================
-- PHOENIX SCHEMA (for Arize Phoenix observability)
-- =============================================================================

-- Create Phoenix schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS phoenix;

-- Grant permissions to the main user
GRANT ALL PRIVILEGES ON SCHEMA phoenix TO drx;
ALTER DEFAULT PRIVILEGES IN SCHEMA phoenix GRANT ALL PRIVILEGES ON TABLES TO drx;
ALTER DEFAULT PRIVILEGES IN SCHEMA phoenix GRANT ALL PRIVILEGES ON SEQUENCES TO drx;

-- =============================================================================
-- CUSTOM TYPES & ENUMS
-- =============================================================================

-- Session status lifecycle
DO $$ BEGIN
    CREATE TYPE session_status AS ENUM (
        'pending',
        'running',
        'paused',
        'completed',
        'failed',
        'cancelled'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Research step status
DO $$ BEGIN
    CREATE TYPE step_status AS ENUM (
        'pending',
        'queued',
        'running',
        'succeeded',
        'failed',
        'skipped',
        'retrying'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- DAG node status
DO $$ BEGIN
    CREATE TYPE dag_node_status AS ENUM (
        'pending',
        'ready',
        'running',
        'succeeded',
        'failed',
        'cancelled'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Agent types in the system
DO $$ BEGIN
    CREATE TYPE agent_type AS ENUM (
        'orchestrator',
        'planner',
        'searcher',
        'reader',
        'reasoner',
        'writer',
        'critic',
        'synthesizer'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Step types for research workflow
DO $$ BEGIN
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
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Policy violation severity
DO $$ BEGIN
    CREATE TYPE violation_severity AS ENUM (
        'info',
        'warning',
        'error',
        'critical'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Table: research_sessions
-- Purpose: Root entity for each deep research request
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    organization_id VARCHAR(255),
    query TEXT NOT NULL,
    query_hash VARCHAR(64) GENERATED ALWAYS AS (encode(sha256(query::bytea), 'hex')) STORED,
    steerability JSONB NOT NULL DEFAULT '{
        "depth": "standard",
        "breadth": "balanced",
        "source_priority": ["academic", "authoritative", "recent"],
        "max_iterations": 5
    }',
    config JSONB NOT NULL DEFAULT '{}',
    status session_status NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    token_budget INTEGER NOT NULL DEFAULT 500000,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    final_report_id UUID,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_research_sessions_user_id ON research_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_research_sessions_status ON research_sessions(status);
CREATE INDEX IF NOT EXISTS idx_research_sessions_created_at ON research_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_sessions_query_hash ON research_sessions(query_hash);

-- -----------------------------------------------------------------------------
-- Table: research_steps
-- Purpose: Individual steps within a research session
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    parent_step_id UUID REFERENCES research_steps(id),
    step_type step_type NOT NULL,
    step_order INTEGER NOT NULL,
    status step_status NOT NULL DEFAULT 'pending',
    agent_type agent_type NOT NULL,
    input_data JSONB NOT NULL DEFAULT '{}',
    output_data JSONB,
    tokens_used INTEGER DEFAULT 0,
    duration_ms INTEGER,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_steps_session_id ON research_steps(session_id);
CREATE INDEX IF NOT EXISTS idx_research_steps_status ON research_steps(status);
CREATE INDEX IF NOT EXISTS idx_research_steps_step_type ON research_steps(step_type);

-- -----------------------------------------------------------------------------
-- Table: sources
-- Purpose: Discovered sources during research
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    url_hash VARCHAR(64) GENERATED ALWAYS AS (encode(sha256(url::bytea), 'hex')) STORED,
    title TEXT,
    domain VARCHAR(255),
    content_type VARCHAR(50),
    content TEXT,
    content_length INTEGER,
    relevance_score FLOAT CHECK (relevance_score BETWEEN 0 AND 1),
    credibility_score FLOAT CHECK (credibility_score BETWEEN 0 AND 1),
    freshness_score FLOAT CHECK (freshness_score BETWEEN 0 AND 1),
    overall_score FLOAT GENERATED ALWAYS AS (
        COALESCE(relevance_score, 0) * 0.5 +
        COALESCE(credibility_score, 0) * 0.3 +
        COALESCE(freshness_score, 0) * 0.2
    ) STORED,
    embedding vector(1536),
    fetched_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sources_session_id ON sources(session_id);
CREATE INDEX IF NOT EXISTS idx_sources_url_hash ON sources(url_hash);
CREATE INDEX IF NOT EXISTS idx_sources_overall_score ON sources(overall_score DESC);

-- Vector similarity index for embeddings
CREATE INDEX IF NOT EXISTS idx_sources_embedding ON sources
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- -----------------------------------------------------------------------------
-- Table: reports
-- Purpose: Generated research reports
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    version INTEGER NOT NULL DEFAULT 1,
    title TEXT NOT NULL,
    executive_summary TEXT,
    content TEXT NOT NULL,
    format VARCHAR(50) NOT NULL DEFAULT 'markdown',
    word_count INTEGER,
    quality_score FLOAT CHECK (quality_score BETWEEN 0 AND 1),
    is_final BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_reports_session_id ON reports(session_id);
CREATE INDEX IF NOT EXISTS idx_reports_is_final ON reports(is_final);

-- -----------------------------------------------------------------------------
-- Table: checkpoints
-- Purpose: LangGraph checkpoint storage for state persistence
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255) NOT NULL,
    parent_checkpoint_id VARCHAR(255),
    checkpoint_data JSONB NOT NULL,
    channel_values JSONB,
    channel_versions JSONB,
    versions_seen JSONB,
    pending_sends JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(thread_id, checkpoint_id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at DESC);

-- -----------------------------------------------------------------------------
-- Table: checkpoint_writes
-- Purpose: Pending checkpoint writes for LangGraph
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255) NOT NULL,
    channel VARCHAR(255) NOT NULL,
    idx INTEGER NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(thread_id, checkpoint_id, task_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread_id ON checkpoint_writes(thread_id);

-- =============================================================================
-- UPDATE TRIGGERS
-- =============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
DO $$
BEGIN
    -- research_sessions
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_research_sessions_updated_at') THEN
        CREATE TRIGGER update_research_sessions_updated_at
            BEFORE UPDATE ON research_sessions
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;

    -- research_steps
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_research_steps_updated_at') THEN
        CREATE TRIGGER update_research_steps_updated_at
            BEFORE UPDATE ON research_steps
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- =============================================================================
-- PERFORMANCE SETTINGS
-- =============================================================================

-- Analyze tables for query optimization
ANALYZE research_sessions;
ANALYZE research_steps;
ANALYZE sources;
ANALYZE reports;
ANALYZE checkpoints;
ANALYZE checkpoint_writes;

-- =============================================================================
-- GRANT PERMISSIONS
-- =============================================================================

-- Grant full access to the application user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO drx;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO drx;
GRANT USAGE ON SCHEMA public TO drx;

-- =============================================================================
-- INITIALIZATION COMPLETE
-- =============================================================================

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'DRX Database initialization completed successfully at %', NOW();
END $$;
