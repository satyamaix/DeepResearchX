"""Report Generator for DRX Evaluation Pipeline.

Generates comprehensive evaluation reports in markdown and JSON formats,
aggregating results from DeepEval and Ragas metrics.

WP-1A: Report Generator Implementation

Key Features:
- Aggregates all metric results from evaluation runs
- Computes pass/fail for each threshold (hard gates vs soft gates)
- Generates markdown report with tables
- Creates JSON output for CI consumption
- Per-scenario breakdown with detailed metrics
- Summary statistics and recommendations

Usage:
    python ci/evaluation/report_generator.py --input eval_results.json --output report.md
    python ci/evaluation/report_generator.py --input eval_results.json --output results.json --format json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class ThresholdConfig:
    """Configuration for evaluation thresholds."""

    # Hard gates (must pass for CI)
    faithfulness: float = 0.8
    hallucination_max: float = 0.2
    task_completion: float = 0.7
    policy_violations_max: int = 0

    # Soft gates (warnings)
    answer_relevancy: float = 0.7
    context_precision: float = 0.6
    context_recall: float = 0.6


@dataclass
class GateResult:
    """Result for a single gate check."""

    name: str
    value: float | int
    threshold: float | int
    passed: bool
    is_hard_gate: bool
    reason: str | None = None


@dataclass
class ScenarioReport:
    """Report for a single scenario."""

    scenario_id: str
    name: str
    input_query: str
    success: bool
    duration_seconds: float
    policy_blocked: bool
    error: str | None
    metrics: dict[str, float]
    gate_results: list[GateResult]


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    timestamp: str
    version: str
    total_cases: int
    passed: int
    failed: int
    warnings: int
    hard_gates: dict[str, Any]
    soft_gates: dict[str, Any]
    scenarios: list[ScenarioReport]
    recommendations: list[str] = field(default_factory=list)
    overall_passed: bool = True


# =============================================================================
# Report Generator
# =============================================================================


class ReportGenerator:
    """Generate evaluation reports from metric results.

    This class aggregates evaluation results and generates comprehensive
    reports in both markdown and JSON formats.

    Attributes:
        thresholds: Configured thresholds for gates
        results: Raw evaluation results
        metric_results: DeepEval/Ragas metric results
    """

    def __init__(
        self,
        thresholds: ThresholdConfig | None = None,
    ) -> None:
        """Initialize the report generator.

        Args:
            thresholds: Optional custom threshold configuration
        """
        self.thresholds = thresholds or ThresholdConfig()
        self.results: list[dict[str, Any]] = []
        self.metric_results: dict[str, Any] = {}
        self.test_case_lookup: dict[str, dict[str, Any]] = {}

    def load_results(
        self,
        eval_results_path: Path,
        metric_results_path: Path | None = None,
        test_cases_path: Path | None = None,
    ) -> None:
        """Load evaluation results from files.

        Args:
            eval_results_path: Path to evaluation results JSON
            metric_results_path: Optional path to metric results JSON
            test_cases_path: Optional path to test cases YAML for metadata
        """
        # Load evaluation results
        with open(eval_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.results = data.get("results", [])
            if "metric_results" in data:
                self.metric_results = data["metric_results"]

        # Load metric results if separate file
        if metric_results_path and metric_results_path.exists():
            with open(metric_results_path, "r", encoding="utf-8") as f:
                self.metric_results = json.load(f)

        # Load test case metadata if available
        if test_cases_path and test_cases_path.exists():
            import yaml

            with open(test_cases_path, "r", encoding="utf-8") as f:
                test_data = yaml.safe_load(f)
                for tc in test_data.get("test_cases", []):
                    self.test_case_lookup[tc["id"]] = tc

    def generate_report(self) -> EvaluationReport:
        """Generate comprehensive evaluation report.

        Returns:
            EvaluationReport with all metrics and gate results
        """
        scenarios: list[ScenarioReport] = []
        total_passed = 0
        total_failed = 0
        total_warnings = 0

        # Track aggregate metrics
        all_faithfulness: list[float] = []
        all_hallucination: list[float] = []
        all_answer_relevancy: list[float] = []
        all_context_precision: list[float] = []
        all_context_recall: list[float] = []
        all_task_completion: list[float] = []
        policy_violations = 0

        for result in self.results:
            scenario_id = result.get("scenario_id", "unknown")
            test_case = self.test_case_lookup.get(scenario_id, {})

            # Build scenario metrics from result and metric_results
            metrics: dict[str, float] = {}
            gate_results: list[GateResult] = []

            # Get DeepEval metrics for this scenario
            deepeval_scenario = self._find_metric_result(scenario_id)
            if deepeval_scenario:
                for metric in deepeval_scenario.get("metrics", []):
                    metric_name = metric.get("metric_name", "")
                    score = metric.get("score", 0.0)
                    metrics[metric_name] = score

            # Check success/failure
            success = result.get("success", False)
            policy_blocked = result.get("policy_blocked", False)

            # For negative tests, policy_blocked is expected success
            is_negative = test_case.get("is_negative_test", False)
            if is_negative:
                success = policy_blocked

            # Track policy violations
            if policy_blocked and not is_negative:
                policy_violations += 1

            # Calculate task completion based on expected outputs
            task_completion = self._calculate_task_completion(result, test_case)
            metrics["task_completion"] = task_completion

            # Build gate results
            if "faithfulness" in metrics:
                all_faithfulness.append(metrics["faithfulness"])
                gate_results.append(
                    GateResult(
                        name="faithfulness",
                        value=metrics["faithfulness"],
                        threshold=self.thresholds.faithfulness,
                        passed=metrics["faithfulness"] >= self.thresholds.faithfulness,
                        is_hard_gate=True,
                    )
                )

            if "hallucination" in metrics:
                all_hallucination.append(metrics["hallucination"])
                gate_results.append(
                    GateResult(
                        name="hallucination",
                        value=metrics["hallucination"],
                        threshold=self.thresholds.hallucination_max,
                        passed=metrics["hallucination"] <= self.thresholds.hallucination_max,
                        is_hard_gate=True,
                    )
                )

            all_task_completion.append(task_completion)
            gate_results.append(
                GateResult(
                    name="task_completion",
                    value=task_completion,
                    threshold=self.thresholds.task_completion,
                    passed=task_completion >= self.thresholds.task_completion,
                    is_hard_gate=True,
                )
            )

            if "answer_relevancy" in metrics:
                all_answer_relevancy.append(metrics["answer_relevancy"])
                gate_results.append(
                    GateResult(
                        name="answer_relevancy",
                        value=metrics["answer_relevancy"],
                        threshold=self.thresholds.answer_relevancy,
                        passed=metrics["answer_relevancy"] >= self.thresholds.answer_relevancy,
                        is_hard_gate=False,
                    )
                )

            if "context_precision" in metrics:
                all_context_precision.append(metrics["context_precision"])
                gate_results.append(
                    GateResult(
                        name="context_precision",
                        value=metrics["context_precision"],
                        threshold=self.thresholds.context_precision,
                        passed=metrics["context_precision"] >= self.thresholds.context_precision,
                        is_hard_gate=False,
                    )
                )

            if "context_recall" in metrics:
                all_context_recall.append(metrics["context_recall"])
                gate_results.append(
                    GateResult(
                        name="context_recall",
                        value=metrics["context_recall"],
                        threshold=self.thresholds.context_recall,
                        passed=metrics["context_recall"] >= self.thresholds.context_recall,
                        is_hard_gate=False,
                    )
                )

            # Determine scenario pass/fail
            hard_gates_passed = all(g.passed for g in gate_results if g.is_hard_gate)
            soft_gates_warnings = sum(1 for g in gate_results if not g.is_hard_gate and not g.passed)

            if success and hard_gates_passed:
                total_passed += 1
            else:
                total_failed += 1

            total_warnings += soft_gates_warnings

            scenarios.append(
                ScenarioReport(
                    scenario_id=scenario_id,
                    name=test_case.get("name", scenario_id),
                    input_query=result.get("input", ""),
                    success=success and hard_gates_passed,
                    duration_seconds=result.get("duration_seconds", 0.0),
                    policy_blocked=policy_blocked,
                    error=result.get("error"),
                    metrics=metrics,
                    gate_results=gate_results,
                )
            )

        # Calculate aggregate hard gate results
        hard_gates = {
            "faithfulness": {
                "avg": sum(all_faithfulness) / len(all_faithfulness) if all_faithfulness else 0.0,
                "threshold": self.thresholds.faithfulness,
                "passed": (sum(all_faithfulness) / len(all_faithfulness) >= self.thresholds.faithfulness)
                if all_faithfulness
                else False,
            },
            "hallucination": {
                "avg": sum(all_hallucination) / len(all_hallucination) if all_hallucination else 0.0,
                "threshold": self.thresholds.hallucination_max,
                "passed": (sum(all_hallucination) / len(all_hallucination) <= self.thresholds.hallucination_max)
                if all_hallucination
                else True,
            },
            "task_completion": {
                "avg": sum(all_task_completion) / len(all_task_completion) if all_task_completion else 0.0,
                "threshold": self.thresholds.task_completion,
                "passed": (sum(all_task_completion) / len(all_task_completion) >= self.thresholds.task_completion)
                if all_task_completion
                else False,
            },
            "policy_violations": {
                "count": policy_violations,
                "threshold": self.thresholds.policy_violations_max,
                "passed": policy_violations <= self.thresholds.policy_violations_max,
            },
        }

        # Calculate aggregate soft gate results
        soft_gates = {
            "answer_relevancy": {
                "avg": sum(all_answer_relevancy) / len(all_answer_relevancy) if all_answer_relevancy else 0.0,
                "threshold": self.thresholds.answer_relevancy,
                "passed": (sum(all_answer_relevancy) / len(all_answer_relevancy) >= self.thresholds.answer_relevancy)
                if all_answer_relevancy
                else True,
            },
            "context_precision": {
                "avg": sum(all_context_precision) / len(all_context_precision) if all_context_precision else 0.0,
                "threshold": self.thresholds.context_precision,
                "passed": (sum(all_context_precision) / len(all_context_precision) >= self.thresholds.context_precision)
                if all_context_precision
                else True,
            },
            "context_recall": {
                "avg": sum(all_context_recall) / len(all_context_recall) if all_context_recall else 0.0,
                "threshold": self.thresholds.context_recall,
                "passed": (sum(all_context_recall) / len(all_context_recall) >= self.thresholds.context_recall)
                if all_context_recall
                else True,
            },
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(hard_gates, soft_gates, scenarios)

        # Determine overall pass/fail
        overall_passed = all(gate["passed"] for gate in hard_gates.values())

        return EvaluationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="1.0.0",
            total_cases=len(self.results),
            passed=total_passed,
            failed=total_failed,
            warnings=total_warnings,
            hard_gates=hard_gates,
            soft_gates=soft_gates,
            scenarios=scenarios,
            recommendations=recommendations,
            overall_passed=overall_passed,
        )

    def _find_metric_result(self, scenario_id: str) -> dict[str, Any] | None:
        """Find metric results for a specific scenario.

        Args:
            scenario_id: The scenario ID to look up

        Returns:
            Metric result dictionary or None
        """
        if not self.metric_results:
            return None

        # Check deepeval results
        deepeval = self.metric_results.get("deepeval", {})
        scenarios = deepeval.get("scenarios", [])

        for scenario in scenarios:
            if scenario.get("scenario_id") == scenario_id:
                return scenario

        return None

    def _calculate_task_completion(
        self,
        result: dict[str, Any],
        test_case: dict[str, Any],
    ) -> float:
        """Calculate task completion score.

        Checks if expected outputs appear in actual output.

        Args:
            result: Evaluation result
            test_case: Test case definition

        Returns:
            Task completion score (0.0 to 1.0)
        """
        expected_outputs = test_case.get("expected_outputs", [])
        if not expected_outputs:
            # If no expected outputs defined, use success flag
            return 1.0 if result.get("success", False) else 0.0

        actual_output = result.get("actual_output", "").lower()
        if not actual_output:
            return 0.0

        # Count how many expected items appear in output
        matches = 0
        for expected in expected_outputs:
            if isinstance(expected, str):
                if expected.lower() in actual_output:
                    matches += 1
            elif isinstance(expected, dict):
                # Handle structured expected output
                key = expected.get("text", expected.get("keyword", ""))
                if key and key.lower() in actual_output:
                    matches += 1

        return matches / len(expected_outputs) if expected_outputs else 1.0

    def _generate_recommendations(
        self,
        hard_gates: dict[str, Any],
        soft_gates: dict[str, Any],
        scenarios: list[ScenarioReport],
    ) -> list[str]:
        """Generate recommendations based on evaluation results.

        Args:
            hard_gates: Hard gate results
            soft_gates: Soft gate results
            scenarios: List of scenario reports

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check hard gates
        if not hard_gates["faithfulness"]["passed"]:
            avg = hard_gates["faithfulness"]["avg"]
            threshold = hard_gates["faithfulness"]["threshold"]
            recommendations.append(
                f"Faithfulness score ({avg:.2f}) below threshold ({threshold}). "
                "Consider improving context retrieval and citation accuracy."
            )

        if not hard_gates["hallucination"]["passed"]:
            avg = hard_gates["hallucination"]["avg"]
            threshold = hard_gates["hallucination"]["threshold"]
            recommendations.append(
                f"Hallucination score ({avg:.2f}) above threshold ({threshold}). "
                "Review output validation and fact-checking mechanisms."
            )

        if not hard_gates["task_completion"]["passed"]:
            avg = hard_gates["task_completion"]["avg"]
            threshold = hard_gates["task_completion"]["threshold"]
            recommendations.append(
                f"Task completion rate ({avg:.2f}) below threshold ({threshold}). "
                "Analyze failed scenarios to identify common patterns."
            )

        if not hard_gates["policy_violations"]["passed"]:
            count = hard_gates["policy_violations"]["count"]
            recommendations.append(
                f"Found {count} policy violation(s). "
                "Review policy enforcement in the research pipeline."
            )

        # Check soft gates
        if not soft_gates["answer_relevancy"]["passed"]:
            avg = soft_gates["answer_relevancy"]["avg"]
            recommendations.append(
                f"Answer relevancy ({avg:.2f}) could be improved. "
                "Consider refining query interpretation and response focus."
            )

        if not soft_gates["context_precision"]["passed"]:
            avg = soft_gates["context_precision"]["avg"]
            recommendations.append(
                f"Context precision ({avg:.2f}) is low. "
                "Improve source ranking and filtering algorithms."
            )

        if not soft_gates["context_recall"]["passed"]:
            avg = soft_gates["context_recall"]["avg"]
            recommendations.append(
                f"Context recall ({avg:.2f}) is low. "
                "Expand search coverage and diversify source retrieval."
            )

        # Check for error patterns
        error_scenarios = [s for s in scenarios if s.error]
        if error_scenarios:
            recommendations.append(
                f"Found {len(error_scenarios)} scenario(s) with errors. "
                "Review error handling and timeout configurations."
            )

        # Check for slow scenarios
        slow_scenarios = [s for s in scenarios if s.duration_seconds > 300]
        if slow_scenarios:
            recommendations.append(
                f"Found {len(slow_scenarios)} scenario(s) exceeding 5 minutes. "
                "Consider optimizing research iteration depth or parallelization."
            )

        if not recommendations:
            recommendations.append(
                "All evaluation gates passed. System is performing within expected parameters."
            )

        return recommendations

    def to_json(self, report: EvaluationReport) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary.

        Args:
            report: The evaluation report

        Returns:
            JSON-serializable dictionary
        """
        return {
            "timestamp": report.timestamp,
            "version": report.version,
            "summary": {
                "total_cases": report.total_cases,
                "passed": report.passed,
                "failed": report.failed,
                "warnings": report.warnings,
                "overall_passed": report.overall_passed,
            },
            "hard_gates": report.hard_gates,
            "soft_gates": report.soft_gates,
            "scenarios": [
                {
                    "scenario_id": s.scenario_id,
                    "name": s.name,
                    "input_query": s.input_query,
                    "success": s.success,
                    "duration_seconds": s.duration_seconds,
                    "policy_blocked": s.policy_blocked,
                    "error": s.error,
                    "metrics": s.metrics,
                    "gate_results": [
                        {
                            "name": g.name,
                            "value": g.value,
                            "threshold": g.threshold,
                            "passed": g.passed,
                            "is_hard_gate": g.is_hard_gate,
                        }
                        for g in s.gate_results
                    ],
                }
                for s in report.scenarios
            ],
            "recommendations": report.recommendations,
        }

    def to_markdown(self, report: EvaluationReport) -> str:
        """Convert report to markdown format.

        Args:
            report: The evaluation report

        Returns:
            Markdown-formatted string
        """
        lines = []

        # Header
        lines.append("# DRX Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {report.timestamp}")
        lines.append(f"**Version:** {report.version}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        status = "PASSED" if report.overall_passed else "FAILED"
        status_icon = "✅" if report.overall_passed else "❌"
        lines.append(f"**Overall Status:** {status_icon} {status}")
        lines.append("")
        lines.append(f"- **Total Cases:** {report.total_cases}")
        lines.append(f"- **Passed:** {report.passed}")
        lines.append(f"- **Failed:** {report.failed}")
        lines.append(f"- **Warnings:** {report.warnings}")
        lines.append("")

        # Hard Gates Table
        lines.append("## Hard Gates (Must Pass)")
        lines.append("")
        lines.append("| Metric | Score | Threshold | Status |")
        lines.append("|--------|-------|-----------|--------|")

        for name, data in report.hard_gates.items():
            if name == "policy_violations":
                score = str(data["count"])
                threshold = f"≤ {data['threshold']}"
            else:
                score = f"{data['avg']:.2f}"
                if name == "hallucination":
                    threshold = f"≤ {data['threshold']}"
                else:
                    threshold = f"≥ {data['threshold']}"

            status = "✅ Pass" if data["passed"] else "❌ Fail"
            lines.append(f"| {name} | {score} | {threshold} | {status} |")

        lines.append("")

        # Soft Gates Table
        lines.append("## Soft Gates (Warnings)")
        lines.append("")
        lines.append("| Metric | Score | Threshold | Status |")
        lines.append("|--------|-------|-----------|--------|")

        for name, data in report.soft_gates.items():
            score = f"{data['avg']:.2f}"
            threshold = f"≥ {data['threshold']}"
            status = "✅ Pass" if data["passed"] else "⚠️ Warning"
            lines.append(f"| {name} | {score} | {threshold} | {status} |")

        lines.append("")

        # Scenario Details
        lines.append("## Scenario Results")
        lines.append("")
        lines.append("| ID | Name | Status | Duration | Task Completion |")
        lines.append("|----|------|--------|----------|-----------------|")

        for scenario in report.scenarios:
            status = "✅" if scenario.success else "❌"
            duration = f"{scenario.duration_seconds:.1f}s"
            task_comp = f"{scenario.metrics.get('task_completion', 0):.0%}"

            if scenario.policy_blocked:
                status = "🚫 Blocked"
            elif scenario.error:
                status = f"❌ Error"

            lines.append(
                f"| {scenario.scenario_id} | {scenario.name} | {status} | {duration} | {task_comp} |"
            )

        lines.append("")

        # Failed Scenarios Details
        failed_scenarios = [s for s in report.scenarios if not s.success]
        if failed_scenarios:
            lines.append("### Failed Scenario Details")
            lines.append("")

            for scenario in failed_scenarios:
                lines.append(f"#### {scenario.scenario_id}: {scenario.name}")
                lines.append("")

                if scenario.error:
                    lines.append(f"**Error:** `{scenario.error}`")
                    lines.append("")

                if scenario.policy_blocked:
                    lines.append("**Policy:** Request was blocked by policy enforcement.")
                    lines.append("")

                lines.append("**Gate Results:**")
                lines.append("")
                for gate in scenario.gate_results:
                    gate_type = "Hard" if gate.is_hard_gate else "Soft"
                    status = "✅" if gate.passed else "❌"
                    lines.append(f"- {gate.name} ({gate_type}): {gate.value:.2f} vs {gate.threshold} {status}")

                lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by DRX Evaluation Pipeline*")

        return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation reports from DRX evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate markdown report
  python report_generator.py --input eval_results.json --output report.md

  # Generate JSON report
  python report_generator.py --input eval_results.json --output results.json --format json

  # Include test case metadata
  python report_generator.py --input eval_results.json --test-cases curated_test_cases.yaml --output report.md
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to evaluation results JSON file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the generated report",
    )

    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    parser.add_argument(
        "--metric-results",
        type=Path,
        default=None,
        help="Optional path to separate metric results JSON",
    )

    parser.add_argument(
        "--test-cases",
        type=Path,
        default=None,
        help="Optional path to test cases YAML for metadata",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for report generator."""
    args = parse_args()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate input file
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Create report generator
    generator = ReportGenerator()

    # Load results
    try:
        generator.load_results(
            eval_results_path=args.input,
            metric_results_path=args.metric_results,
            test_cases_path=args.test_cases,
        )
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1

    # Generate report
    report = generator.generate_report()

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        output_data = generator.to_json(report)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
    else:
        output_text = generator.to_markdown(report)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)

    logger.info(f"Report saved to {args.output}")

    # Return exit code based on overall status
    return 0 if report.overall_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "ThresholdConfig",
    "GateResult",
    "ScenarioReport",
    "EvaluationReport",
    "ReportGenerator",
]
