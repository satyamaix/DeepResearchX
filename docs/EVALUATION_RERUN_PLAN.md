# DRX Evaluation Pipeline Re-run Implementation Plan

**Created:** 2026-01-06
**Objective:** Re-run evaluation tests using Ragas/DeepEval against updated codebase with Gemini provider routing

## Executive Summary

This plan defines a comprehensive evaluation re-run strategy organized in topological order with loose coupling. Each work packet (WP) can be delegated to specialized agents with clear interfaces.

## Dependency Graph (Topological Order)

```
WP-1: Environment Validation
    │
    ├──► WP-2A: Docker Stack Restart (with new provider config)
    │         │
    │         └──► WP-3: API Health Verification
    │                   │
    │                   └──► WP-4A: Smoke Test (2 scenarios)
    │                             │
    │                             └──► WP-4B: Full Evaluation (10 scenarios)
    │
    └──► WP-2B: Install DeepEval/Ragas Dependencies
              │
              └──► WP-5: DeepEval Metrics Evaluation
                        │
                        └──► WP-6: Ragas RAG Metrics Evaluation
                                  │
                                  └──► WP-7: Report Generation & Analysis
                                            │
                                            └──► WP-8: Git Tracking & Commit
```

## Work Packets

---

### WP-1: Environment Validation
**Owner:** Infrastructure Agent
**Dependencies:** None
**Priority:** P0 (Critical Path)

**Tasks:**
1. Verify OpenRouter API key is configured
2. Verify GEMINI_PROVIDER=google-ai-studio is set
3. Verify PROVIDER_ALLOW_FALLBACKS=false is set
4. Check OpenRouter credit balance if possible
5. Validate Python venv has required dependencies

**Outputs:**
- Environment validation report (JSON)
- List of missing dependencies
- API key validity status

**Interface Contract:**
```python
class EnvironmentValidation(TypedDict):
    openrouter_configured: bool
    provider_routing_configured: bool
    credit_available: bool | None
    missing_dependencies: list[str]
    validation_passed: bool
```

---

### WP-2A: Docker Stack Restart
**Owner:** Infrastructure Agent
**Dependencies:** WP-1 (must pass)
**Priority:** P0 (Critical Path)

**Tasks:**
1. Stop existing Docker stack
2. Rebuild API and Worker containers to pick up config changes
3. Start Docker stack with updated environment
4. Wait for all services to be healthy
5. Verify provider routing is active in logs

**Outputs:**
- Docker stack status report
- Container health status
- Provider routing confirmation from logs

**Interface Contract:**
```python
class DockerStackStatus(TypedDict):
    services: dict[str, str]  # service_name -> status
    all_healthy: bool
    provider_routing_active: bool
    restart_time_seconds: float
```

---

### WP-2B: Install DeepEval/Ragas Dependencies
**Owner:** Dependencies Agent
**Dependencies:** WP-1 (parallel with WP-2A)
**Priority:** P0 (Critical Path)

**Tasks:**
1. Install DeepEval: `pip install deepeval`
2. Install Ragas: `pip install ragas`
3. Install datasets: `pip install datasets`
4. Verify imports work correctly
5. Configure EVAL_MODEL environment variable

**Outputs:**
- Dependency installation status
- Import verification results
- Configured evaluation model

**Interface Contract:**
```python
class DependencyStatus(TypedDict):
    deepeval_installed: bool
    ragas_installed: bool
    datasets_installed: bool
    all_imports_working: bool
    eval_model: str
```

---

### WP-3: API Health Verification
**Owner:** API Testing Agent
**Dependencies:** WP-2A (must pass)
**Priority:** P0 (Critical Path)

**Tasks:**
1. Check API root endpoint returns 200
2. Check /api/v1/health endpoint
3. Verify interactions endpoint is accessible
4. Submit a minimal test request to verify end-to-end flow
5. Verify response includes provider information

**Outputs:**
- API health status
- Endpoint availability matrix
- Sample interaction ID from test request

**Interface Contract:**
```python
class APIHealthStatus(TypedDict):
    root_healthy: bool
    health_endpoint: bool
    interactions_endpoint: bool
    sample_request_success: bool
    sample_interaction_id: str | None
    latency_ms: float
```

---

### WP-4A: Smoke Test Execution
**Owner:** Evaluation Runner Agent
**Dependencies:** WP-3 (must pass)
**Priority:** P1 (High)

**Tasks:**
1. Load smoke_test group from curated_test_cases.yaml
2. Execute 2 scenarios: competitor_analysis, quick_fact_check
3. Collect outputs and timing metrics
4. Verify outputs are non-empty
5. Save results to ci/evaluation/smoke_test_results.json

**Outputs:**
- Smoke test results JSON
- Pass/fail status for each scenario
- Token usage if available
- Duration metrics

**Interface Contract:**
```python
class SmokeTestResults(TypedDict):
    scenarios_run: int
    scenarios_passed: int
    results: list[EvaluationResult]
    total_duration_seconds: float
    all_passed: bool
```

---

### WP-4B: Full Evaluation Execution
**Owner:** Evaluation Runner Agent
**Dependencies:** WP-4A (must pass)
**Priority:** P1 (High)

**Tasks:**
1. Load full_evaluation group (10 scenarios)
2. Execute all scenarios sequentially (avoid credit exhaustion)
3. Implement graceful handling of credit limits
4. Collect comprehensive outputs including:
   - Final reports
   - Citations and retrieval context
   - Policy block status for negative tests
5. Save results to ci/evaluation/eval_results.json

**Outputs:**
- Full evaluation results JSON
- Per-scenario metrics
- Policy compliance results for negative tests
- Total token usage

**Interface Contract:**
```python
class FullEvaluationResults(TypedDict):
    total_scenarios: int
    successful: int
    failed: int
    policy_blocked: int
    results: list[EvaluationResult]
    total_duration_seconds: float
    total_tokens_used: int | None
```

---

### WP-5: DeepEval Metrics Evaluation
**Owner:** Metrics Agent (DeepEval Specialist)
**Dependencies:** WP-2B, WP-4B (both must pass)
**Priority:** P2 (Medium)

**Tasks:**
1. Configure OpenRouter as DeepEval judge
2. Create LLMTestCases from evaluation results
3. Run FaithfulnessMetric on all scenarios
4. Run HallucinationMetric on all scenarios
5. Run AnswerRelevancyMetric on all scenarios
6. Aggregate scores and determine pass/fail
7. Save DeepEval results to ci/evaluation/deepeval_results.json

**Outputs:**
- DeepEval metrics report
- Per-scenario metric scores
- Aggregate statistics
- Pass/fail determination

**Thresholds:**
- Faithfulness: >= 0.8
- Hallucination: <= 0.2
- Answer Relevancy: >= 0.7

**Interface Contract:**
```python
class DeepEvalResults(TypedDict):
    scenarios: list[BatchEvaluationResult]
    summary: dict[str, Any]
    avg_faithfulness: float
    avg_hallucination: float
    avg_answer_relevancy: float
    overall_passed: bool
```

---

### WP-6: Ragas RAG Metrics Evaluation
**Owner:** Metrics Agent (Ragas Specialist)
**Dependencies:** WP-2B, WP-4B (both must pass)
**Priority:** P2 (Medium)

**Tasks:**
1. Convert evaluation results to Ragas Dataset format
2. Run context_precision metric
3. Run context_recall metric
4. Run Ragas faithfulness metric
5. Run Ragas answer_relevancy metric
6. Aggregate and compare with DeepEval results
7. Save Ragas results to ci/evaluation/ragas_results.json

**Outputs:**
- Ragas metrics report
- Context precision/recall scores
- Comparison with DeepEval metrics
- Combined quality score

**Thresholds:**
- Context Precision: >= 0.6
- Context Recall: >= 0.6
- Faithfulness: >= 0.8
- Answer Relevancy: >= 0.7

**Interface Contract:**
```python
class RagasResults(TypedDict):
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    overall_passed: bool
```

---

### WP-7: Report Generation & Analysis
**Owner:** Report Generator Agent
**Dependencies:** WP-5, WP-6 (both must pass)
**Priority:** P3 (Normal)

**Tasks:**
1. Load all evaluation results (eval_results, deepeval_results, ragas_results)
2. Generate comprehensive markdown report
3. Include executive summary with key metrics
4. Create detailed per-scenario analysis
5. Generate comparison charts (text-based)
6. Provide recommendations based on results
7. Save to ci/evaluation/EVALUATION_REPORT.md
8. Generate JSON summary for CI integration

**Outputs:**
- EVALUATION_REPORT.md (comprehensive)
- EVALUATION_ANALYSIS.md (detailed analysis)
- eval_report.json (structured data)

**Report Sections:**
1. Executive Summary
2. Hard Gates (must-pass metrics)
3. Soft Gates (warning metrics)
4. Per-Scenario Results
5. Negative Test Compliance
6. Token Usage Analysis
7. Recommendations

---

### WP-8: Git Tracking & Commit
**Owner:** Git Agent
**Dependencies:** WP-7 (must pass)
**Priority:** P3 (Normal)

**Tasks:**
1. Stage all new/modified evaluation files
2. Create descriptive commit message
3. Include metrics summary in commit
4. Commit to current branch
5. Optionally create tag for evaluation run

**Files to Track:**
- ci/evaluation/eval_results.json
- ci/evaluation/smoke_test_results.json
- ci/evaluation/deepeval_results.json
- ci/evaluation/ragas_results.json
- ci/evaluation/EVALUATION_REPORT.md
- ci/evaluation/EVALUATION_ANALYSIS.md
- ci/evaluation/eval_report.json

---

## Agent Delegation Strategy

### Agent Profiles (IC9 Level)

| Agent | Specialization | Tools | Model |
|-------|---------------|-------|-------|
| Infrastructure Agent | Docker, Environment, DevOps | Bash, Read, Edit | Opus 4.5 |
| Dependencies Agent | Python packaging, pip, venv | Bash | Opus 4.5 |
| API Testing Agent | HTTP, REST, async Python | Bash, Read | Opus 4.5 |
| Evaluation Runner Agent | Async Python, httpx, YAML | Bash, Read, Write | Opus 4.5 |
| DeepEval Metrics Agent | DeepEval, LLM evaluation | Bash, Read, Write | Opus 4.5 |
| Ragas Metrics Agent | Ragas, RAG metrics, datasets | Bash, Read, Write | Opus 4.5 |
| Report Generator Agent | Markdown, JSON, analysis | Read, Write | Opus 4.5 |
| Git Agent | Git operations, commits | Bash | Opus 4.5 |

### Parallelization Opportunities

```
Phase 1 (Sequential): WP-1
Phase 2 (Parallel):   WP-2A || WP-2B
Phase 3 (Sequential): WP-3
Phase 4 (Sequential): WP-4A -> WP-4B
Phase 5 (Parallel):   WP-5 || WP-6
Phase 6 (Sequential): WP-7 -> WP-8
```

### Error Handling

Each agent should:
1. Validate preconditions before starting
2. Report failures immediately
3. Provide actionable error messages
4. Suggest remediation steps
5. Not proceed if dependencies failed

### Output Token Management

To avoid 32k token output limits:
1. Each agent outputs in multiple steps
2. Write intermediate results to files
3. Use structured JSON for data exchange
4. Keep markdown reports concise with links to details

---

## Execution Commands

```bash
# WP-1: Environment Validation
python -c "from src.config import get_settings; s = get_settings(); print(f'Provider: {s.GEMINI_PROVIDER}')"

# WP-2A: Docker Restart
cd deployment && docker compose down && docker compose up -d --build

# WP-2B: Install Dependencies
pip install deepeval ragas datasets

# WP-3: API Health Check
curl -s http://localhost:8000/api/v1/health

# WP-4A: Smoke Test
python ci/evaluation/run_evaluation.py --scenarios ci/evaluation/curated_test_cases.yaml --group smoke_test --output ci/evaluation/smoke_test_results.json -v

# WP-4B: Full Evaluation
python ci/evaluation/run_evaluation.py --scenarios ci/evaluation/curated_test_cases.yaml --group full_evaluation --output ci/evaluation/eval_results.json -v

# WP-5-7: Metrics and Reports
python ci/evaluation/report_generator.py --input ci/evaluation/eval_results.json --output ci/evaluation/EVALUATION_REPORT.md

# WP-8: Git Commit
git add ci/evaluation/ && git commit -m "feat(eval): Re-run evaluation with Gemini provider routing"
```

---

## Success Criteria

| Metric | Threshold | Gate Type |
|--------|-----------|-----------|
| Faithfulness | >= 0.8 | Hard |
| Hallucination | <= 0.2 | Hard |
| Task Completion | >= 0.7 | Hard |
| Policy Violations | = 0 | Hard |
| Answer Relevancy | >= 0.7 | Soft |
| Context Precision | >= 0.6 | Soft |
| Context Recall | >= 0.6 | Soft |

---

*Plan generated for DRX Evaluation Pipeline Re-run*
