"""
DRX Orchestrator Module.

Provides LangGraph-based workflow orchestration for deep research.

Core Components:
- AgentState: TypedDict state flowing through the workflow
- ResearchOrchestrator: High-level orchestrator for running research
- create_research_workflow: StateGraph builder function
- StreamEvent: Events emitted during workflow execution

Example:
    ```python
    from src.orchestrator import ResearchOrchestrator, StreamEventType

    orchestrator = ResearchOrchestrator()
    await orchestrator.initialize()

    async for event in orchestrator.run("What is quantum computing?"):
        if event.event_type == StreamEventType.REPORT_GENERATED:
            print(event.data)
    ```
"""

from src.orchestrator.state import (
    # Type aliases
    AgentType,
    TaskStatus,
    ToneType,
    FormatType,
    # TypedDicts
    SubTask,
    ResearchPlan,
    CitationRecord,
    Finding,
    SteerabilityParams,
    QualityMetrics,
    AgentState,
    # Factory functions
    create_initial_state,
    create_empty_plan,
    create_subtask,
    create_finding,
    create_citation,
    # Helpers
    update_tokens_used,
    get_pending_tasks,
    get_task_by_id,
    is_plan_complete,
    has_failed_tasks,
)

from src.orchestrator.checkpointer import (
    get_checkpointer,
    create_checkpointer,
    CheckpointerManager,
    checkpointer_context,
    close_checkpointer,
    get_thread_state,
    list_thread_checkpoints,
    delete_thread_checkpoints,
    CheckpointerError,
)

from src.orchestrator.nodes import (
    # Node functions
    plan_research,
    search_sources,
    read_documents,
    synthesize_findings,
    critique_synthesis,
    check_policy,
    generate_report,
    # Conditional edge functions
    should_continue,
    has_policy_violation,
    # Parallel execution helpers
    parallel_node_execution,
    # Agent management
    get_agent,
    register_agent,
    clear_agents,
)

from src.orchestrator.workflow import (
    # Core classes
    ResearchOrchestrator,
    # Event types
    StreamEvent,
    StreamEventType,
    # Graph builders
    create_research_workflow,
    compile_workflow,
    # Convenience functions
    run_research,
    stream_research,
)

from src.orchestrator.budget import (
    # Core classes
    BudgetTracker,
    BudgetExceededError,
    # Type definitions
    BudgetStatus,
    UsageRecord,
    # Cost configuration
    MODEL_COSTS,
    get_model_cost,
    # Factory function
    create_budget_tracker,
)

from src.orchestrator.parallel import (
    # Core classes
    ParallelExecutor,
    TaskResult,
    AggregatedResult,
    DependencyGraph,
    # Utility functions
    merge_task_results,
    is_plan_complete as parallel_is_plan_complete,
    has_critical_failures,
    # Factory function
    create_parallel_executor,
)


__all__ = [
    # State types
    "AgentType",
    "TaskStatus",
    "ToneType",
    "FormatType",
    "SubTask",
    "ResearchPlan",
    "CitationRecord",
    "Finding",
    "SteerabilityParams",
    "QualityMetrics",
    "AgentState",
    # State factory functions
    "create_initial_state",
    "create_empty_plan",
    "create_subtask",
    "create_finding",
    "create_citation",
    # State helpers
    "update_tokens_used",
    "get_pending_tasks",
    "get_task_by_id",
    "is_plan_complete",
    "has_failed_tasks",
    # Checkpointer
    "get_checkpointer",
    "create_checkpointer",
    "CheckpointerManager",
    "checkpointer_context",
    "close_checkpointer",
    "get_thread_state",
    "list_thread_checkpoints",
    "delete_thread_checkpoints",
    "CheckpointerError",
    # Node functions
    "plan_research",
    "search_sources",
    "read_documents",
    "synthesize_findings",
    "critique_synthesis",
    "check_policy",
    "generate_report",
    # Conditional edge functions
    "should_continue",
    "has_policy_violation",
    # Parallel execution
    "parallel_node_execution",
    # Agent management
    "get_agent",
    "register_agent",
    "clear_agents",
    # Orchestrator
    "ResearchOrchestrator",
    # Events
    "StreamEvent",
    "StreamEventType",
    # Graph builders
    "create_research_workflow",
    "compile_workflow",
    # Convenience functions
    "run_research",
    "stream_research",
    # Budget tracking
    "BudgetTracker",
    "BudgetExceededError",
    "BudgetStatus",
    "UsageRecord",
    "MODEL_COSTS",
    "get_model_cost",
    "create_budget_tracker",
    # Parallel execution
    "ParallelExecutor",
    "TaskResult",
    "AggregatedResult",
    "DependencyGraph",
    "merge_task_results",
    "parallel_is_plan_complete",
    "has_critical_failures",
    "create_parallel_executor",
]
