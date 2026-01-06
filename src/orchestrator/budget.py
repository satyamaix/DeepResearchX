"""
Budget Tracking for DRX Deep Research System.

Provides token and cost budget management for research sessions,
including usage tracking, estimation, and enforcement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


# =============================================================================
# Model Cost Configuration
# =============================================================================

# Cost per 1 million tokens (USD)
MODEL_COSTS: dict[str, dict[str, float]] = {
    # Google models
    "google/gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
    "google/gemini-3-flash-preview:online": {"input": 0.075, "output": 0.30},
    "google/gemini-2.5-pro-preview": {"input": 1.25, "output": 5.00},
    # DeepSeek models
    "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
    # Anthropic models
    "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    # OpenAI models
    "openai/gpt-4o": {"input": 2.50, "output": 10.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Default fallback
    "_default": {"input": 1.0, "output": 4.0},
}


def get_model_cost(model: str) -> dict[str, float]:
    """Get cost configuration for a model."""
    # Try exact match
    if model in MODEL_COSTS:
        return MODEL_COSTS[model]

    # Try prefix match for versioned models
    for key in MODEL_COSTS:
        if model.startswith(key.split(":")[0]):
            return MODEL_COSTS[key]

    logger.warning(f"Unknown model cost for {model}, using default")
    return MODEL_COSTS["_default"]


# =============================================================================
# Type Definitions
# =============================================================================


class BudgetStatus(TypedDict):
    """Current budget status."""
    tokens_used: int
    tokens_remaining: int
    tokens_budget: int
    cost_used: float
    cost_remaining: float | None
    cost_budget: float | None
    utilization_percent: float
    is_exceeded: bool
    exceeded_type: str | None  # "tokens" or "cost" or None


class UsageRecord(TypedDict):
    """Record of a single usage event."""
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    agent: str
    timestamp: str


# =============================================================================
# Budget Exceeded Exception
# =============================================================================


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded."""

    def __init__(
        self,
        budget_type: str,
        used: float,
        limit: float,
        model: str | None = None,
    ):
        self.budget_type = budget_type
        self.used = used
        self.limit = limit
        self.model = model

        if budget_type == "tokens":
            msg = f"Token budget exceeded: {used:,.0f} / {limit:,.0f} tokens used"
        elif budget_type == "cost":
            msg = f"Cost budget exceeded: ${used:.4f} / ${limit:.4f} spent"
        else:
            msg = f"Budget exceeded: {used} / {limit}"

        super().__init__(msg)


# =============================================================================
# Budget Tracker
# =============================================================================


@dataclass
class BudgetTracker:
    """
    Track and enforce token and cost budgets for a research session.

    Provides:
    - Usage tracking per model and agent
    - Cost estimation before LLM calls
    - Budget enforcement with configurable limits
    - Detailed usage history
    """

    token_budget: int
    cost_budget: float | None = None

    # Internal tracking
    _tokens_used: int = field(default=0, init=False)
    _cost_used: float = field(default=0.0, init=False)
    _usage_history: list[UsageRecord] = field(default_factory=list, init=False)
    _created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        init=False,
    )

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent: str = "unknown",
    ) -> UsageRecord:
        """
        Track token usage for an LLM call.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            agent: Agent that made the call

        Returns:
            UsageRecord for this call
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens

        self._tokens_used += total_tokens
        self._cost_used += cost

        record = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            agent=agent,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        self._usage_history.append(record)

        logger.debug(
            f"Budget tracked: {total_tokens} tokens, ${cost:.4f} "
            f"({self._tokens_used}/{self.token_budget} total)"
        )

        return record

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate the cost for a specific token usage."""
        costs = get_model_cost(model)
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        return round(input_cost + output_cost, 6)

    def estimate(
        self,
        model: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int | None = None,
    ) -> dict[str, float]:
        """
        Estimate cost for a planned LLM call.

        Args:
            model: Model to use
            estimated_input_tokens: Expected input tokens
            estimated_output_tokens: Expected output tokens (defaults to input/4)

        Returns:
            Dict with estimated cost and tokens
        """
        if estimated_output_tokens is None:
            # Assume output is roughly 1/4 of input as default
            estimated_output_tokens = max(100, estimated_input_tokens // 4)

        cost = self.calculate_cost(model, estimated_input_tokens, estimated_output_tokens)
        total_tokens = estimated_input_tokens + estimated_output_tokens

        return {
            "estimated_tokens": total_tokens,
            "estimated_cost": cost,
            "would_exceed_tokens": (self._tokens_used + total_tokens) > self.token_budget,
            "would_exceed_cost": (
                self.cost_budget is not None and
                (self._cost_used + cost) > self.cost_budget
            ),
        }

    def can_afford(
        self,
        model: str,
        estimated_tokens: int,
    ) -> bool:
        """
        Check if we can afford an estimated LLM call.

        Args:
            model: Model to use
            estimated_tokens: Total estimated tokens (input + output)

        Returns:
            True if within budget
        """
        # Check token budget
        if (self._tokens_used + estimated_tokens) > self.token_budget:
            return False

        # Check cost budget if set
        if self.cost_budget is not None:
            estimated_cost = self.calculate_cost(model, estimated_tokens // 2, estimated_tokens // 2)
            if (self._cost_used + estimated_cost) > self.cost_budget:
                return False

        return True

    def enforce(self) -> None:
        """
        Enforce budget limits.

        Raises:
            BudgetExceededError: If any budget limit is exceeded
        """
        if self._tokens_used > self.token_budget:
            raise BudgetExceededError(
                "tokens",
                self._tokens_used,
                self.token_budget,
            )

        if self.cost_budget is not None and self._cost_used > self.cost_budget:
            raise BudgetExceededError(
                "cost",
                self._cost_used,
                self.cost_budget,
            )

    def enforce_before_call(
        self,
        model: str,
        estimated_tokens: int,
    ) -> None:
        """
        Enforce budget before making an LLM call.

        Args:
            model: Model to use
            estimated_tokens: Estimated total tokens

        Raises:
            BudgetExceededError: If call would exceed budget
        """
        if not self.can_afford(model, estimated_tokens):
            raise BudgetExceededError(
                "tokens",
                self._tokens_used + estimated_tokens,
                self.token_budget,
                model=model,
            )

    @property
    def status(self) -> BudgetStatus:
        """Get current budget status."""
        tokens_remaining = max(0, self.token_budget - self._tokens_used)
        utilization = (self._tokens_used / self.token_budget * 100) if self.token_budget > 0 else 0

        is_token_exceeded = self._tokens_used > self.token_budget
        is_cost_exceeded = (
            self.cost_budget is not None and
            self._cost_used > self.cost_budget
        )

        exceeded_type = None
        if is_token_exceeded:
            exceeded_type = "tokens"
        elif is_cost_exceeded:
            exceeded_type = "cost"

        return BudgetStatus(
            tokens_used=self._tokens_used,
            tokens_remaining=tokens_remaining,
            tokens_budget=self.token_budget,
            cost_used=round(self._cost_used, 4),
            cost_remaining=(
                round(self.cost_budget - self._cost_used, 4)
                if self.cost_budget is not None else None
            ),
            cost_budget=self.cost_budget,
            utilization_percent=round(utilization, 1),
            is_exceeded=is_token_exceeded or is_cost_exceeded,
            exceeded_type=exceeded_type,
        )

    @property
    def tokens_used(self) -> int:
        """Get total tokens used."""
        return self._tokens_used

    @property
    def tokens_remaining(self) -> int:
        """Get remaining token budget."""
        return max(0, self.token_budget - self._tokens_used)

    @property
    def cost_used(self) -> float:
        """Get total cost used."""
        return self._cost_used

    @property
    def usage_history(self) -> list[UsageRecord]:
        """Get usage history."""
        return self._usage_history.copy()

    def get_usage_by_agent(self) -> dict[str, dict[str, Any]]:
        """Get usage breakdown by agent."""
        by_agent: dict[str, dict[str, Any]] = {}

        for record in self._usage_history:
            agent = record["agent"]
            if agent not in by_agent:
                by_agent[agent] = {
                    "tokens": 0,
                    "cost": 0.0,
                    "calls": 0,
                }
            by_agent[agent]["tokens"] += record["input_tokens"] + record["output_tokens"]
            by_agent[agent]["cost"] += record["cost"]
            by_agent[agent]["calls"] += 1

        return by_agent

    def get_usage_by_model(self) -> dict[str, dict[str, Any]]:
        """Get usage breakdown by model."""
        by_model: dict[str, dict[str, Any]] = {}

        for record in self._usage_history:
            model = record["model"]
            if model not in by_model:
                by_model[model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                    "calls": 0,
                }
            by_model[model]["input_tokens"] += record["input_tokens"]
            by_model[model]["output_tokens"] += record["output_tokens"]
            by_model[model]["cost"] += record["cost"]
            by_model[model]["calls"] += 1

        return by_model

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state to dict."""
        return {
            "token_budget": self.token_budget,
            "cost_budget": self.cost_budget,
            "tokens_used": self._tokens_used,
            "cost_used": self._cost_used,
            "usage_history": self._usage_history,
            "created_at": self._created_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetTracker:
        """Deserialize tracker from dict."""
        tracker = cls(
            token_budget=data["token_budget"],
            cost_budget=data.get("cost_budget"),
        )
        tracker._tokens_used = data.get("tokens_used", 0)
        tracker._cost_used = data.get("cost_used", 0.0)
        tracker._usage_history = data.get("usage_history", [])
        tracker._created_at = data.get("created_at", tracker._created_at)
        return tracker


# =============================================================================
# Factory Function
# =============================================================================


def create_budget_tracker(
    token_budget: int = 500000,
    cost_budget: float | None = None,
) -> BudgetTracker:
    """
    Create a configured BudgetTracker instance.

    Args:
        token_budget: Maximum tokens to consume
        cost_budget: Optional maximum cost in dollars

    Returns:
        Configured BudgetTracker
    """
    return BudgetTracker(
        token_budget=token_budget,
        cost_budget=cost_budget,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BudgetTracker",
    "BudgetStatus",
    "UsageRecord",
    "BudgetExceededError",
    "MODEL_COSTS",
    "get_model_cost",
    "create_budget_tracker",
]
