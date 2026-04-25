"""
Phase 2 — ground-truth impact tracing.

Given an agent's predicted list of affected consumers, compare against
the cascade scenario's ground truth and emit a precision/recall-style
``ImpactTraceResult`` so the reward layer can score each consumer
decision independently (composable rubric).
"""

from dataclasses import dataclass, field
from typing import List

from .service_graph import CascadeScenario


@dataclass
class ImpactTraceResult:
    """Per-consumer outcome of one ``trace_impact`` action."""

    correct_hits: List[str] = field(default_factory=list)
    missed: List[str] = field(default_factory=list)
    false_flags: List[str] = field(default_factory=list)
    unknown_services: List[str] = field(default_factory=list)
    total_consumers: int = 0

    @property
    def precision(self) -> float:
        flagged = len(self.correct_hits) + len(self.false_flags)
        if flagged == 0:
            return 0.0
        return len(self.correct_hits) / flagged

    @property
    def recall(self) -> float:
        truly_affected = len(self.correct_hits) + len(self.missed)
        if truly_affected == 0:
            return 1.0  # nothing to find
        return len(self.correct_hits) / truly_affected

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


def _normalise(name: str) -> str:
    return name.strip().lower()


def trace_impact(
    scenario: CascadeScenario,
    predicted_affected: List[str],
) -> ImpactTraceResult:
    """Compare agent's predicted consumer list against ground truth.

    ``predicted_affected`` is matched case-insensitively. Names that match
    no consumer in the scenario are reported in ``unknown_services`` and
    treated as false flags for reward purposes.
    """
    truth_lookup = {_normalise(n): n for n in scenario.ground_truth_affected}
    known_consumers = {_normalise(c.name): c.name for c in scenario.consumers}

    seen: set = set()
    correct_hits: List[str] = []
    false_flags: List[str] = []
    unknown_services: List[str] = []

    for raw in predicted_affected:
        key = _normalise(raw)
        if key in seen:
            continue
        seen.add(key)

        if key in truth_lookup:
            correct_hits.append(truth_lookup[key])
        elif key in known_consumers:
            false_flags.append(known_consumers[key])
        else:
            unknown_services.append(raw)

    missed = [
        name
        for name in scenario.ground_truth_affected
        if _normalise(name) not in {_normalise(c) for c in correct_hits}
    ]

    return ImpactTraceResult(
        correct_hits=correct_hits,
        missed=missed,
        false_flags=false_flags,
        unknown_services=unknown_services,
        total_consumers=len(scenario.consumers),
    )
