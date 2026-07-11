"""Ringer: a router-judge multi-agent harness.

Fable/Opus plan and judge; Sonnet/Haiku -- or, per unit, an OpenRouter
open-weight model (GLM 5.2, Kimi K2) -- do the work; a mechanical checker
gates everything before it's trusted. See docs/design.html for the design
rationale behind each module.
"""

from .agent_test import AgentTestInputs, AgentTestResult, score
from .checker import Checker, CheckResult, compose, schema_checker, source_attachment_checker
from .models import Provider, Role, Tier, resolve_model
from .orchestrator import AgentTestGateError, OrchestratorConfig, RunReport, run
from .spec import TaskSpec, UnitResult, UnitStatus

__all__ = [
    "AgentTestInputs",
    "AgentTestResult",
    "score",
    "Provider",
    "Role",
    "Tier",
    "resolve_model",
    "Checker",
    "CheckResult",
    "compose",
    "schema_checker",
    "source_attachment_checker",
    "AgentTestGateError",
    "OrchestratorConfig",
    "RunReport",
    "run",
    "TaskSpec",
    "UnitResult",
    "UnitStatus",
]
