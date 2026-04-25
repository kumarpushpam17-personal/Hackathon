"""
Enterprise service graph for Phase 2 / Phase 3 tasks.

Defines a small fictional microservice ecosystem inspired by real
e-commerce platforms: a producer service (UserService) whose API change
ripples through several consumers (OrdersService, BillingService,
NotificationsService, AnalyticsETL).

Each scenario contains:
  * ``producer_spec_v1``    — original OpenAPI spec
  * ``producer_spec_v2``    — spec after the breaking change
  * ``violation``           — the breaking-change record being analysed
  * ``consumers``           — declarations of which fields each consumer
                              depends on plus their own contract specs
  * ``ground_truth_affected`` — names of consumers whose contract is broken

The graph is intentionally compact: judges can read the whole graph in
under a minute, but the dependency structure is rich enough to surface
multi-hop impact (e.g. AnalyticsETL only breaks because BillingService
forwards a renamed field).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class ConsumerDeclaration:
    """How one consumer depends on the producer."""

    name: str
    description: str
    fields_consumed: List[str]
    spec_excerpt: Dict[str, Any]


@dataclass
class CascadeScenario:
    """A complete Phase 2/Phase 3 scenario."""

    scenario_id: str
    producer_name: str
    producer_spec_v1: Dict[str, Any]
    producer_spec_v2: Dict[str, Any]
    violation: Dict[str, Any]
    consumers: List[ConsumerDeclaration]
    ground_truth_affected: List[str]
    description: str
    acceptable_fix_strategies: List[str] = field(
        default_factory=lambda: [
            "field_alias",
            "version_bump",
            "deprecation_window",
            "dual_write",
        ]
    )


# ── Scenario A — UserService renames `email` to `email_address` ──────────


def _scenario_user_email_rename() -> CascadeScenario:
    """A producer renames a field that three consumers read directly."""

    producer_v1 = {
        "openapi": "3.0.0",
        "info": {"title": "UserService", "version": "1.0.0"},
        "paths": {
            "/users/{id}": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["id", "email", "created_at"],
                                        "properties": {
                                            "id": {"type": "string"},
                                            "email": {
                                                "type": "string",
                                                "format": "email",
                                            },
                                            "name": {"type": "string"},
                                            "created_at": {
                                                "type": "string",
                                                "format": "date-time",
                                            },
                                        },
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
    }

    producer_v2 = {
        "openapi": "3.0.0",
        "info": {"title": "UserService", "version": "2.0.0"},
        "paths": {
            "/users/{id}": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": [
                                            "id",
                                            "email_address",
                                            "created_at",
                                        ],
                                        "properties": {
                                            "id": {"type": "string"},
                                            "email_address": {
                                                "type": "string",
                                                "format": "email",
                                            },
                                            "name": {"type": "string"},
                                            "created_at": {
                                                "type": "string",
                                                "format": "date-time",
                                            },
                                        },
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
    }

    violation = {
        "field_path": "GET /users/{id}.email",
        "violation_type": "breaking_change",
        "description": (
            "Field 'email' was renamed to 'email_address' and the original "
            "'email' was removed from required fields and properties."
        ),
        "from": "email",
        "to": "email_address",
    }

    consumers = [
        ConsumerDeclaration(
            name="OrdersService",
            description="Reads user.email to attach customer email to orders.",
            fields_consumed=["id", "email"],
            spec_excerpt={
                "expects": {
                    "id": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                }
            },
        ),
        ConsumerDeclaration(
            name="BillingService",
            description=(
                "Reads user.email to send invoices; forwards email to "
                "AnalyticsETL through its own response."
            ),
            fields_consumed=["id", "email"],
            spec_excerpt={
                "expects": {
                    "id": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                }
            },
        ),
        ConsumerDeclaration(
            name="NotificationsService",
            description="Sends transactional emails to user.email.",
            fields_consumed=["email"],
            spec_excerpt={
                "expects": {
                    "email": {"type": "string", "format": "email"},
                }
            },
        ),
        ConsumerDeclaration(
            name="AnalyticsETL",
            description=(
                "Reads only id and created_at from UserService directly. "
                "(Tempting false-flag — does NOT consume 'email'.)"
            ),
            fields_consumed=["id", "created_at"],
            spec_excerpt={
                "expects": {
                    "id": {"type": "string"},
                    "created_at": {"type": "string", "format": "date-time"},
                }
            },
        ),
    ]

    return CascadeScenario(
        scenario_id="user_email_rename",
        producer_name="UserService",
        producer_spec_v1=producer_v1,
        producer_spec_v2=producer_v2,
        violation=violation,
        consumers=consumers,
        ground_truth_affected=[
            "OrdersService",
            "BillingService",
            "NotificationsService",
        ],
        description=(
            "UserService renamed 'email' to 'email_address'. Identify which "
            "downstream services break, and propose a fix that keeps all "
            "consumers working without forcing them to redeploy."
        ),
        acceptable_fix_strategies=[
            "field_alias",
            "version_bump",
            "deprecation_window",
            "dual_write",
        ],
    )


# ── Scenario B — OrdersService narrows `status` enum ─────────────────────


def _scenario_orders_status_narrowed() -> CascadeScenario:
    """A producer removes enum values that two consumers still emit."""

    producer_v1 = {
        "openapi": "3.0.0",
        "info": {"title": "OrdersService", "version": "1.0.0"},
        "paths": {
            "/orders/{id}/status": {
                "put": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["status"],
                                    "properties": {
                                        "status": {
                                            "type": "string",
                                            "enum": [
                                                "pending",
                                                "confirmed",
                                                "shipped",
                                                "delivered",
                                                "cancelled",
                                                "refunded",
                                            ],
                                        }
                                    },
                                }
                            }
                        }
                    }
                }
            }
        },
    }

    producer_v2 = {
        "openapi": "3.0.0",
        "info": {"title": "OrdersService", "version": "2.0.0"},
        "paths": {
            "/orders/{id}/status": {
                "put": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["status"],
                                    "properties": {
                                        "status": {
                                            "type": "string",
                                            "enum": [
                                                "pending",
                                                "confirmed",
                                                "shipped",
                                                "delivered",
                                            ],
                                        }
                                    },
                                }
                            }
                        }
                    }
                }
            }
        },
    }

    violation = {
        "field_path": "PUT /orders/{id}/status.status",
        "violation_type": "breaking_change",
        "description": (
            "Enum narrowed: 'cancelled' and 'refunded' were removed from "
            "the allowed status values."
        ),
        "removed_values": ["cancelled", "refunded"],
    }

    consumers = [
        ConsumerDeclaration(
            name="ReturnsService",
            description=(
                "Sets status='refunded' when processing a return. Will "
                "fail on v2 because 'refunded' is no longer accepted."
            ),
            fields_consumed=["status"],
            spec_excerpt={
                "emits": {"status": {"enum": ["refunded"]}}
            },
        ),
        ConsumerDeclaration(
            name="SupportPortal",
            description=(
                "Allows agents to mark orders as 'cancelled'. Will fail on "
                "v2 because 'cancelled' is no longer accepted."
            ),
            fields_consumed=["status"],
            spec_excerpt={
                "emits": {"status": {"enum": ["cancelled"]}}
            },
        ),
        ConsumerDeclaration(
            name="ShippingService",
            description=(
                "Only emits 'shipped' and 'delivered' — both still valid. "
                "(False-flag candidate.)"
            ),
            fields_consumed=["status"],
            spec_excerpt={
                "emits": {"status": {"enum": ["shipped", "delivered"]}}
            },
        ),
    ]

    return CascadeScenario(
        scenario_id="orders_status_narrowed",
        producer_name="OrdersService",
        producer_spec_v1=producer_v1,
        producer_spec_v2=producer_v2,
        violation=violation,
        consumers=consumers,
        ground_truth_affected=["ReturnsService", "SupportPortal"],
        description=(
            "OrdersService narrowed the order status enum, removing "
            "'cancelled' and 'refunded'. Identify which consumers can no "
            "longer call this endpoint."
        ),
        acceptable_fix_strategies=[
            "version_bump",
            "deprecation_window",
            "consumer_patch",
        ],
    )


# ── Public registry ───────────────────────────────────────────────────────


_SCENARIOS: Dict[str, CascadeScenario] = {
    "user_email_rename": _scenario_user_email_rename(),
    "orders_status_narrowed": _scenario_orders_status_narrowed(),
}


def get_cascade_scenario(
    scenario_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> CascadeScenario:
    """Return a cascade scenario by id, or pick one deterministically by seed.

    Parameters
    ----------
    scenario_id:
        Explicit scenario name. Takes precedence over ``seed``.
    seed:
        If provided (and ``scenario_id`` is not), selects ``user_email_rename``
        for even seeds and ``orders_status_narrowed`` for odd seeds.

    Returns
    -------
    CascadeScenario
        The (immutable) selected scenario.
    """
    if scenario_id is not None:
        if scenario_id not in _SCENARIOS:
            raise ValueError(
                f"Unknown cascade scenario '{scenario_id}'. "
                f"Available: {list(_SCENARIOS)}"
            )
        return _SCENARIOS[scenario_id]

    keys = sorted(_SCENARIOS.keys())
    if seed is None:
        return _SCENARIOS[keys[0]]
    return _SCENARIOS[keys[seed % len(keys)]]


def public_observation(scenario: CascadeScenario) -> Dict[str, Any]:
    """Return the portion of the scenario the agent is allowed to see.

    The ground-truth ``ground_truth_affected`` list is held back so the
    agent must reason about impact from the consumer declarations rather
    than read the answer.
    """
    return {
        "producer": scenario.producer_name,
        "producer_spec_v1": scenario.producer_spec_v1,
        "producer_spec_v2": scenario.producer_spec_v2,
        "violation": scenario.violation,
        "consumers": [
            {
                "name": c.name,
                "description": c.description,
                "fields_consumed": c.fields_consumed,
                "spec_excerpt": c.spec_excerpt,
            }
            for c in scenario.consumers
        ],
    }


def consumer_specs_for_fix(scenario: CascadeScenario) -> Dict[str, Any]:
    """Return only the consumer specs needed for Phase 3 fix validation."""
    return {
        c.name: {
            "spec_excerpt": c.spec_excerpt,
            "fields_consumed": c.fields_consumed,
        }
        for c in scenario.consumers
    }


CASCADE_SCENARIO_IDS: List[str] = sorted(_SCENARIOS.keys())
