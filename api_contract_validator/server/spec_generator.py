"""
Spec generator for the API Contract Validator Environment.

Generates OpenAPI specifications, payloads with planted violations,
and ground-truth violation records for six difficulty levels:

    find_type_mismatches        Easy   — 4 violations from pool of 12 (495 combos)
    validate_nested_objects     Medium — 7 violations, 2 scenario variants
    detect_breaking_changes     Hard   — 9 breaking changes between v1 and v2
    validate_response_schema    Expert — 10 format/constraint violations, 2 variants
    validate_cross_field_constraints Expert — 7 arithmetic + date cross-field errors
    validate_auth_request       Expert — 6 auth/security violations, 2 variants

Each generator accepts an optional *seed* for deterministic randomisation:
    seed=None  → fixed canonical scenario (backward-compatible defaults)
    seed=int   → reproducible randomised variant (use for RL training)

This makes the environment suitable for both evaluation (fixed seed) and
training (varied seeds) — the key distinction between a one-shot evaluator
and a genuine RL training environment.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Sentinel used in violation pool entries to signal "remove key from payload"
_REMOVE = object()


@dataclass
class PlantedViolation:
    """A single ground-truth violation planted in a payload."""

    field_path: str
    violation_type: str
    description: str
    expected_value: str
    actual_value: str


@dataclass
class TaskScenario:
    """Everything the environment needs for one episode."""

    task_name: str
    task_description: str
    api_spec: Dict[str, Any]
    payload: Dict[str, Any]
    violations: List[PlantedViolation]
    max_steps: int


# ── Easy Task — pool-based randomisation ──────────────────────────────────
#
# The spec exposes 8 fields.  Each episode seeds a random draw of 4 of the
# 8 possible violations, giving 70 unique episode combinations — enough to
# train an agent rather than just evaluate it once.

_EASY_SPEC: Dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "User Service API", "version": "1.0.0"},
    "paths": {
        "/users": {
            "post": {
                "summary": "Create a new user",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["username", "email", "age", "is_active"],
                                "properties": {
                                    "username": {
                                        "type": "string",
                                        "minLength": 3,
                                        "maxLength": 32,
                                    },
                                    "email": {
                                        "type": "string",
                                        "format": "email",
                                    },
                                    "age": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 120,
                                    },
                                    "is_active": {"type": "boolean"},
                                    "role": {
                                        "type": "string",
                                        "enum": ["admin", "editor", "viewer"],
                                    },
                                    "phone": {
                                        "type": "string",
                                        "pattern": r"^\+[1-9][0-9]{7,14}$",
                                    },
                                    "account_balance": {
                                        "type": "number",
                                        "minimum": 0.0,
                                    },
                                    "terms_accepted": {"type": "boolean"},
                                },
                            }
                        }
                    },
                },
            }
        }
    },
}

# Base payload — all fields valid.  Pool entries mutate specific keys.
_EASY_VALID_PAYLOAD: Dict[str, Any] = {
    "username": "alice_smith",
    "email": "alice@example.com",
    "age": 28,
    "is_active": True,
    "role": "editor",
    "phone": "+14155551234",
    "account_balance": 250.00,
    "terms_accepted": True,
}

# Pool: (top-level key, bad value or _REMOVE, PlantedViolation)
# The first 4 entries reproduce the original fixed scenario (seed=None).
_EASY_POOL: List[Tuple[str, Any, PlantedViolation]] = [
    (
        "email",
        _REMOVE,
        PlantedViolation(
            field_path="email",
            violation_type="missing_required",
            description="Required field 'email' is missing from the payload.",
            expected_value="(present, type: string, format: email)",
            actual_value="(missing)",
        ),
    ),
    (
        "age",
        "twenty-five",
        PlantedViolation(
            field_path="age",
            violation_type="type_mismatch",
            description="Field 'age' should be integer but received string.",
            expected_value="integer",
            actual_value="string ('twenty-five')",
        ),
    ),
    (
        "is_active",
        "yes",
        PlantedViolation(
            field_path="is_active",
            violation_type="type_mismatch",
            description="Field 'is_active' should be boolean but received string.",
            expected_value="boolean",
            actual_value="string ('yes')",
        ),
    ),
    (
        "role",
        "superadmin",
        PlantedViolation(
            field_path="role",
            violation_type="invalid_enum",
            description="Value 'superadmin' is not in allowed enum [admin, editor, viewer].",
            expected_value="one of: admin, editor, viewer",
            actual_value="superadmin",
        ),
    ),
    (
        "phone",
        "555-CALL-US",
        PlantedViolation(
            field_path="phone",
            violation_type="format_error",
            description="Field 'phone' contains letters; must match international format +E.164.",
            expected_value=r"pattern: ^\+[1-9][0-9]{7,14}$",
            actual_value="555-CALL-US",
        ),
    ),
    (
        "account_balance",
        -50.00,
        PlantedViolation(
            field_path="account_balance",
            violation_type="format_error",
            description="Field 'account_balance' is -50.0, below the minimum of 0.",
            expected_value="number >= 0",
            actual_value="-50.0",
        ),
    ),
    (
        "terms_accepted",
        "agreed",
        PlantedViolation(
            field_path="terms_accepted",
            violation_type="type_mismatch",
            description="Field 'terms_accepted' should be boolean but received string.",
            expected_value="boolean",
            actual_value="string ('agreed')",
        ),
    ),
    (
        "username",
        "ab",
        PlantedViolation(
            field_path="username",
            violation_type="format_error",
            description="Field 'username' is 'ab' (length 2), below minLength of 3.",
            expected_value="string, minLength: 3",
            actual_value="'ab' (length 2)",
        ),
    ),
    (
        "account_balance",
        "two-fifty",
        PlantedViolation(
            field_path="account_balance",
            violation_type="type_mismatch",
            description="Field 'account_balance' should be number but received string.",
            expected_value="number",
            actual_value="string ('two-fifty')",
        ),
    ),
    (
        "phone",
        "0014155551234",
        PlantedViolation(
            field_path="phone",
            violation_type="format_error",
            description="Field 'phone' starts with 00 not +; must match E.164 format.",
            expected_value=r"pattern: ^\+[1-9][0-9]{7,14}$",
            actual_value="'0014155551234'",
        ),
    ),
    (
        "terms_accepted",
        1,
        PlantedViolation(
            field_path="terms_accepted",
            violation_type="type_mismatch",
            description="Field 'terms_accepted' should be boolean but received integer.",
            expected_value="boolean",
            actual_value="integer (1)",
        ),
    ),
    (
        "username",
        "a" * 33,
        PlantedViolation(
            field_path="username",
            violation_type="format_error",
            description="Field 'username' has length 33, exceeds maxLength of 32.",
            expected_value="string, maxLength: 32",
            actual_value=f"'{('a' * 33)}' (length 33)",
        ),
    ),
]


def generate_easy_scenario(seed: Optional[int] = None) -> TaskScenario:
    """Easy scenario: 4 violations sampled from a pool of 8.

    seed=None always returns the original canonical 4 violations.
    Any integer seed reproducibly draws a different subset.
    """
    if seed is None:
        selected = _EASY_POOL[:4]
    else:
        rng = random.Random(seed)
        selected = rng.sample(_EASY_POOL, 4)

    payload = dict(_EASY_VALID_PAYLOAD)
    violations: List[PlantedViolation] = []
    for key, bad_val, planted in selected:
        if bad_val is _REMOVE:
            payload.pop(key, None)
        else:
            payload[key] = bad_val
        violations.append(planted)

    return TaskScenario(
        task_name="find_type_mismatches",
        task_description=(
            "You are given an OpenAPI specification and an API request payload. "
            "Find all violations in the payload: type mismatches, missing required "
            "fields, invalid enum values, and format errors. "
            "Report one violation per step using the field's dot-notation path. "
            "Submit field_path='DONE' when finished."
        ),
        api_spec=_EASY_SPEC,
        payload=payload,
        violations=violations,
        max_steps=10,
    )


# ── Medium Task — two complete scenario variants ──────────────────────────

def _medium_scenario_a() -> TaskScenario:
    """Original Order Service scenario with 7 nested violations."""
    api_spec: Dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {"title": "Order Service API", "version": "2.1.0"},
        "paths": {
            "/orders": {
                "post": {
                    "summary": "Place a new order",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": [
                                        "customer",
                                        "items",
                                        "shipping_address",
                                        "payment_method",
                                    ],
                                    "properties": {
                                        "customer": {
                                            "type": "object",
                                            "required": ["id", "email"],
                                            "properties": {
                                                "id": {"type": "integer"},
                                                "email": {
                                                    "type": "string",
                                                    "format": "email",
                                                },
                                                "loyalty_tier": {
                                                    "type": "string",
                                                    "enum": [
                                                        "bronze",
                                                        "silver",
                                                        "gold",
                                                        "platinum",
                                                    ],
                                                },
                                            },
                                        },
                                        "items": {
                                            "type": "array",
                                            "minItems": 1,
                                            "items": {
                                                "type": "object",
                                                "required": [
                                                    "product_id",
                                                    "quantity",
                                                    "unit_price",
                                                ],
                                                "properties": {
                                                    "product_id": {"type": "string"},
                                                    "quantity": {
                                                        "type": "integer",
                                                        "minimum": 1,
                                                    },
                                                    "unit_price": {
                                                        "type": "number",
                                                        "minimum": 0,
                                                    },
                                                },
                                            },
                                        },
                                        "shipping_address": {
                                            "type": "object",
                                            "required": [
                                                "street",
                                                "city",
                                                "zip_code",
                                                "country",
                                            ],
                                            "properties": {
                                                "street": {"type": "string"},
                                                "city": {"type": "string"},
                                                "zip_code": {"type": "string"},
                                                "country": {
                                                    "type": "string",
                                                    "pattern": "^[A-Z]{2}$",
                                                },
                                            },
                                        },
                                        "payment_method": {
                                            "type": "string",
                                            "enum": [
                                                "credit_card",
                                                "debit_card",
                                                "paypal",
                                                "bank_transfer",
                                            ],
                                        },
                                    },
                                }
                            }
                        },
                    },
                }
            }
        },
    }

    payload: Dict[str, Any] = {
        "customer": {
            "id": "C-1001",
            "loyalty_tier": "diamond",
        },
        "items": [
            {"product_id": "SKU-001", "quantity": 2, "unit_price": 29.99},
            {"product_id": 12345, "quantity": "three", "unit_price": 15.50},
        ],
        "shipping_address": {
            "street": "123 Main St",
            "city": "Springfield",
            "zip_code": 62704,
            "country": "usa",
        },
        "payment_method": "crypto",
    }

    violations = [
        PlantedViolation(
            field_path="customer.email",
            violation_type="missing_required",
            description="Required field 'email' missing from 'customer' object.",
            expected_value="(present, type: string, format: email)",
            actual_value="(missing)",
        ),
        PlantedViolation(
            field_path="customer.id",
            violation_type="type_mismatch",
            description="Field 'customer.id' should be integer, got string.",
            expected_value="integer",
            actual_value="string ('C-1001')",
        ),
        PlantedViolation(
            field_path="customer.loyalty_tier",
            violation_type="invalid_enum",
            description="Value 'diamond' not in enum [bronze, silver, gold, platinum].",
            expected_value="one of: bronze, silver, gold, platinum",
            actual_value="diamond",
        ),
        PlantedViolation(
            field_path="items[1].product_id",
            violation_type="type_mismatch",
            description="Array item 'items[1].product_id' should be string, got integer.",
            expected_value="string",
            actual_value="integer (12345)",
        ),
        PlantedViolation(
            field_path="items[1].quantity",
            violation_type="type_mismatch",
            description="Array item 'items[1].quantity' should be integer, got string.",
            expected_value="integer",
            actual_value="string ('three')",
        ),
        PlantedViolation(
            field_path="shipping_address.zip_code",
            violation_type="type_mismatch",
            description="Field 'shipping_address.zip_code' should be string, got integer.",
            expected_value="string",
            actual_value="integer (62704)",
        ),
        PlantedViolation(
            field_path="payment_method",
            violation_type="invalid_enum",
            description="Value 'crypto' not in enum [credit_card, debit_card, paypal, bank_transfer].",
            expected_value="one of: credit_card, debit_card, paypal, bank_transfer",
            actual_value="crypto",
        ),
    ]

    return TaskScenario(
        task_name="validate_nested_objects",
        task_description=(
            "You are given an OpenAPI specification and an API request payload "
            "with nested objects and arrays. Find all violations including type "
            "mismatches in nested fields, missing required fields inside objects, "
            "invalid enum values, and type errors in array items. Use dot-notation "
            "for nested paths (e.g. 'customer.email') and bracket notation for "
            "arrays (e.g. 'items[1].quantity'). Submit field_path='DONE' when finished."
        ),
        api_spec=api_spec,
        payload=payload,
        violations=violations,
        max_steps=15,
    )


def _medium_scenario_b() -> TaskScenario:
    """Alternate Event Booking scenario — different domain, 7 nested violations."""
    api_spec: Dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {"title": "Event Booking API", "version": "1.5.0"},
        "paths": {
            "/bookings": {
                "post": {
                    "summary": "Book an event",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["event", "attendees", "organizer", "payment"],
                                    "properties": {
                                        "event": {
                                            "type": "object",
                                            "required": ["id", "type", "capacity"],
                                            "properties": {
                                                "id": {"type": "integer"},
                                                "type": {
                                                    "type": "string",
                                                    "enum": [
                                                        "concert",
                                                        "conference",
                                                        "sports",
                                                        "theater",
                                                    ],
                                                },
                                                "capacity": {
                                                    "type": "integer",
                                                    "minimum": 1,
                                                },
                                                "date": {
                                                    "type": "string",
                                                    "format": "date",
                                                },
                                            },
                                        },
                                        "attendees": {
                                            "type": "array",
                                            "minItems": 1,
                                            "items": {
                                                "type": "object",
                                                "required": ["name", "email", "ticket_type"],
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "email": {
                                                        "type": "string",
                                                        "format": "email",
                                                    },
                                                    "age": {
                                                        "type": "integer",
                                                        "minimum": 0,
                                                        "maximum": 120,
                                                    },
                                                    "ticket_type": {
                                                        "type": "string",
                                                        "enum": [
                                                            "vip",
                                                            "standard",
                                                            "economy",
                                                        ],
                                                    },
                                                },
                                            },
                                        },
                                        "organizer": {
                                            "type": "object",
                                            "required": ["name", "contact_email"],
                                            "properties": {
                                                "name": {"type": "string"},
                                                "contact_email": {
                                                    "type": "string",
                                                    "format": "email",
                                                },
                                                "phone": {"type": "string"},
                                            },
                                        },
                                        "payment": {
                                            "type": "object",
                                            "required": ["method", "total_amount"],
                                            "properties": {
                                                "method": {
                                                    "type": "string",
                                                    "enum": [
                                                        "card",
                                                        "invoice",
                                                        "cash",
                                                    ],
                                                },
                                                "total_amount": {
                                                    "type": "number",
                                                    "minimum": 0,
                                                },
                                            },
                                        },
                                    },
                                }
                            }
                        },
                    },
                }
            }
        },
    }

    payload: Dict[str, Any] = {
        "event": {
            "id": "EVT-001",
            "type": "festival",
            "capacity": 500,
            "date": "2025-06-15",
        },
        "attendees": [
            {
                "name": "Sarah Chen",
                "ticket_type": "vip",
            },
            {
                "name": "Mark Rivera",
                "email": "mark@example.com",
                "age": 135,
                "ticket_type": "premium",
            },
        ],
        "organizer": {
            "name": "LiveNation Events",
            "contact_email": "organizer.example.com",
            "phone": "+442071234567",
        },
        "payment": {
            "method": "card",
            "total_amount": "free",
        },
    }

    violations = [
        PlantedViolation(
            field_path="event.id",
            violation_type="type_mismatch",
            description="Field 'event.id' should be integer, got string.",
            expected_value="integer",
            actual_value="string ('EVT-001')",
        ),
        PlantedViolation(
            field_path="event.type",
            violation_type="invalid_enum",
            description="Value 'festival' not in enum [concert, conference, sports, theater].",
            expected_value="one of: concert, conference, sports, theater",
            actual_value="festival",
        ),
        PlantedViolation(
            field_path="attendees[0].email",
            violation_type="missing_required",
            description="Required field 'email' missing from attendees[0].",
            expected_value="(present, type: string, format: email)",
            actual_value="(missing)",
        ),
        PlantedViolation(
            field_path="attendees[1].age",
            violation_type="format_error",
            description="Field 'attendees[1].age' is 135, exceeds maximum of 120.",
            expected_value="integer, maximum: 120",
            actual_value="135",
        ),
        PlantedViolation(
            field_path="attendees[1].ticket_type",
            violation_type="invalid_enum",
            description="Value 'premium' not in enum [vip, standard, economy].",
            expected_value="one of: vip, standard, economy",
            actual_value="premium",
        ),
        PlantedViolation(
            field_path="organizer.contact_email",
            violation_type="format_error",
            description="Field 'organizer.contact_email' is not a valid email (missing @).",
            expected_value="string, format: email",
            actual_value="'organizer.example.com'",
        ),
        PlantedViolation(
            field_path="payment.total_amount",
            violation_type="type_mismatch",
            description="Field 'payment.total_amount' should be number, got string.",
            expected_value="number",
            actual_value="string ('free')",
        ),
    ]

    return TaskScenario(
        task_name="validate_nested_objects",
        task_description=(
            "You are given an OpenAPI specification and an API request payload "
            "with nested objects and arrays. Find all violations including type "
            "mismatches in nested fields, missing required fields inside objects, "
            "invalid enum values, format errors, and type errors in array items. "
            "Use dot-notation for nested paths (e.g. 'organizer.contact_email') "
            "and bracket notation for arrays (e.g. 'attendees[1].age'). "
            "Submit field_path='DONE' when finished."
        ),
        api_spec=api_spec,
        payload=payload,
        violations=violations,
        max_steps=15,
    )


def generate_medium_scenario(seed: Optional[int] = None) -> TaskScenario:
    """Medium scenario: two complete variants selected by seed.

    seed=None or even seed → Order Service (variant A, original).
    Odd seed → Event Booking (variant B).
    """
    if seed is None or seed % 2 == 0:
        return _medium_scenario_a()
    return _medium_scenario_b()


# ── Hard Task — breaking changes (fixed, no variant needed) ───────────────


def generate_hard_scenario(seed: Optional[int] = None) -> TaskScenario:
    """Hard scenario: 9 breaking changes between two API spec versions.

    The task is inherently complex (spec-diffing); a single well-designed
    scenario is more valuable than noisy variants.  seed is accepted for
    API uniformity but not used.
    """
    api_spec: Dict[str, Any] = {
        "description": (
            "Compare v1 (old) and v2 (new) of the Product Catalog API. "
            "Identify all BREAKING changes that would cause existing v1 "
            "clients to fail when calling the v2 API."
        ),
        "v1": {
            "openapi": "3.0.3",
            "info": {"title": "Product Catalog API", "version": "1.0.0"},
            "paths": {
                "/products": {
                    "post": {
                        "summary": "Create a product",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["name", "price", "category"],
                                        "properties": {
                                            "name": {"type": "string"},
                                            "description": {"type": "string"},
                                            "price": {"type": "number"},
                                            "category": {
                                                "type": "string",
                                                "enum": [
                                                    "electronics",
                                                    "clothing",
                                                    "books",
                                                    "home",
                                                    "sports",
                                                ],
                                            },
                                            "tags": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "weight_kg": {"type": "number"},
                                            "supplier_code": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                    "get": {
                        "summary": "List products",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "integer"},
                                                    "name": {"type": "string"},
                                                    "price": {"type": "number"},
                                                    "category": {"type": "string"},
                                                    "discount_percent": {
                                                        "type": "number"
                                                    },
                                                },
                                            },
                                        }
                                    }
                                }
                            }
                        },
                    },
                }
            },
        },
        "v2": {
            "openapi": "3.0.3",
            "info": {"title": "Product Catalog API", "version": "2.0.0"},
            "paths": {
                "/products": {
                    "post": {
                        "summary": "Create a product",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": [
                                            "name",
                                            "price",
                                            "category",
                                            "sku",
                                        ],
                                        "properties": {
                                            "name": {"type": "string"},
                                            "description": {"type": "string"},
                                            "price": {"type": "string"},
                                            "category": {
                                                "type": "string",
                                                "enum": [
                                                    "electronics",
                                                    "clothing",
                                                    "books",
                                                ],
                                            },
                                            "tags": {"type": "integer"},
                                            "sku": {"type": "string"},
                                            "weight_grams": {"type": "integer"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                    "get": {
                        "summary": "List products",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "string"},
                                                    "name": {"type": "string"},
                                                    "price": {"type": "string"},
                                                    "category": {"type": "string"},
                                                },
                                            },
                                        }
                                    }
                                }
                            }
                        },
                    },
                }
            },
        },
    }

    payload: Dict[str, Any] = {
        "name": "Wireless Headphones",
        "description": "Noise-cancelling over-ear headphones",
        "price": 79.99,
        "category": "sports",
        "tags": ["audio", "wireless"],
        "weight_kg": 0.35,
        "supplier_code": "SUP-442",
    }

    violations = [
        PlantedViolation(
            field_path="POST /products.price",
            violation_type="breaking_change",
            description="Field 'price' type changed from 'number' to 'string' in v2.",
            expected_value="number (v1)",
            actual_value="string (v2)",
        ),
        PlantedViolation(
            field_path="POST /products.category",
            violation_type="breaking_change",
            description="Enum for 'category' narrowed: 'home' and 'sports' removed in v2.",
            expected_value="enum: electronics, clothing, books, home, sports (v1)",
            actual_value="enum: electronics, clothing, books (v2)",
        ),
        PlantedViolation(
            field_path="POST /products.tags",
            violation_type="breaking_change",
            description="Field 'tags' type changed from 'array of strings' to 'integer' in v2.",
            expected_value="array of strings (v1)",
            actual_value="integer (v2)",
        ),
        PlantedViolation(
            field_path="POST /products.sku",
            violation_type="breaking_change",
            description="New required field 'sku' added in v2 — breaks existing clients.",
            expected_value="(not required in v1)",
            actual_value="required string (v2)",
        ),
        PlantedViolation(
            field_path="POST /products.supplier_code",
            violation_type="breaking_change",
            description="Field 'supplier_code' removed in v2 — clients sending it get rejected.",
            expected_value="string (v1)",
            actual_value="(removed in v2)",
        ),
        PlantedViolation(
            field_path="POST /products.weight_kg",
            violation_type="breaking_change",
            description="Field 'weight_kg' removed; replaced by 'weight_grams' with different type.",
            expected_value="number 'weight_kg' (v1)",
            actual_value="integer 'weight_grams' (v2)",
        ),
        PlantedViolation(
            field_path="GET /products[].id",
            violation_type="breaking_change",
            description="Response field 'id' type changed from 'integer' to 'string' in v2.",
            expected_value="integer (v1)",
            actual_value="string (v2)",
        ),
        PlantedViolation(
            field_path="GET /products[].price",
            violation_type="breaking_change",
            description="Response field 'price' type changed from 'number' to 'string' in v2.",
            expected_value="number (v1)",
            actual_value="string (v2)",
        ),
        PlantedViolation(
            field_path="GET /products[].discount_percent",
            violation_type="breaking_change",
            description="Response field 'discount_percent' removed in v2 — dependent clients break.",
            expected_value="number (v1)",
            actual_value="(removed in v2)",
        ),
    ]

    return TaskScenario(
        task_name="detect_breaking_changes",
        task_description=(
            "You are given two versions (v1 and v2) of a Product Catalog API "
            "specification, plus a sample v1 client payload. Identify all "
            "BREAKING changes between v1 and v2 that would cause existing "
            "clients to fail. Breaking changes include: type changes, removed "
            "fields, narrowed enums, new required fields, and removed response "
            "fields. Use the format 'METHOD /path.field' for paths "
            "(e.g. 'POST /products.price'). Submit field_path='DONE' when finished."
        ),
        api_spec=api_spec,
        payload=payload,
        violations=violations,
        max_steps=20,
    )


# ── Expert Task — response schema format validation ───────────────────────
#
# Agents must validate an API *response* (not a request) against the spec.
# Violations are subtle: pattern mismatches, out-of-range numerics, wrong
# date formats, and invalid enum values scattered across nested objects and
# arrays.  Two complete variants are provided via seed selection.

_RESPONSE_SPEC: Dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "E-Commerce Order API", "version": "3.0.0"},
    "paths": {
        "/orders/{order_id}": {
            "get": {
                "summary": "Retrieve a single order",
                "responses": {
                    "200": {
                        "description": "Order details",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": [
                                        "order_id",
                                        "created_at",
                                        "status",
                                        "customer",
                                        "items",
                                        "billing",
                                    ],
                                    "properties": {
                                        "order_id": {
                                            "type": "string",
                                            "pattern": "^ORD-[0-9]{6}$",
                                            "description": "Must match ORD-NNNNNN exactly",
                                        },
                                        "created_at": {
                                            "type": "string",
                                            "format": "date-time",
                                            "description": "ISO 8601 date-time, e.g. 2024-01-15T10:30:00Z",
                                        },
                                        "promised_delivery_date": {
                                            "type": "string",
                                            "format": "date",
                                            "description": "ISO 8601 date, e.g. 2024-03-20",
                                        },
                                        "status": {
                                            "type": "string",
                                            "enum": [
                                                "pending",
                                                "processing",
                                                "shipped",
                                                "delivered",
                                                "cancelled",
                                            ],
                                        },
                                        "customer": {
                                            "type": "object",
                                            "required": ["email", "phone", "loyalty_points"],
                                            "properties": {
                                                "email": {
                                                    "type": "string",
                                                    "format": "email",
                                                },
                                                "phone": {
                                                    "type": "string",
                                                    "pattern": r"^\+[1-9][0-9]{7,14}$",
                                                    "description": "E.164 format, e.g. +14155552671",
                                                },
                                                "loyalty_points": {
                                                    "type": "integer",
                                                    "minimum": 0,
                                                },
                                            },
                                        },
                                        "items": {
                                            "type": "array",
                                            "minItems": 1,
                                            "items": {
                                                "type": "object",
                                                "required": [
                                                    "sku",
                                                    "unit_price",
                                                    "quantity",
                                                    "discount_rate",
                                                ],
                                                "properties": {
                                                    "sku": {
                                                        "type": "string",
                                                        "pattern": "^SKU-[A-Z0-9]{6}$",
                                                        "description": "Must match SKU-XXXXXX",
                                                    },
                                                    "unit_price": {
                                                        "type": "number",
                                                        "minimum": 0.01,
                                                    },
                                                    "quantity": {
                                                        "type": "integer",
                                                        "minimum": 1,
                                                        "maximum": 100,
                                                    },
                                                    "discount_rate": {
                                                        "type": "number",
                                                        "minimum": 0,
                                                        "maximum": 1,
                                                    },
                                                },
                                            },
                                        },
                                        "billing": {
                                            "type": "object",
                                            "required": ["subtotal", "tax_rate", "total"],
                                            "properties": {
                                                "subtotal": {
                                                    "type": "number",
                                                    "minimum": 0,
                                                },
                                                "tax_rate": {
                                                    "type": "number",
                                                    "minimum": 0,
                                                    "maximum": 0.5,
                                                    "description": "Fraction 0.0–0.5 (max 50%)",
                                                },
                                                "total": {
                                                    "type": "number",
                                                    "minimum": 0,
                                                },
                                            },
                                        },
                                        "tracking_code": {
                                            "type": "string",
                                            "pattern": "^[A-Z]{2}[0-9]{9}[A-Z]{2}$",
                                            "description": "2 upper letters + 9 digits + 2 upper letters",
                                        },
                                        "estimated_days": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 30,
                                        },
                                    },
                                }
                            }
                        },
                    }
                },
            }
        }
    },
}

_RESPONSE_TASK_DESCRIPTION = (
    "You are given an OpenAPI response schema for GET /orders/{order_id} "
    "and an actual API response. Validate the response against the schema. "
    "Find all violations including: format errors (invalid date/time formats, "
    "pattern mismatches, out-of-range numeric values), type mismatches, and "
    "invalid enum values. These are subtle — read the spec constraints carefully "
    "(patterns, minimum/maximum, format strings). "
    "Use dot-notation for nested paths and bracket notation for arrays. "
    "Submit field_path='DONE' when finished."
)


def _response_scenario_a() -> TaskScenario:
    """Expert variant A: 10 format/constraint violations in an order response."""
    payload: Dict[str, Any] = {
        "order_id": "ORDER-123456",           # format_error: must be ORD-NNNNNN
        "created_at": "2024-01-15 10:30:00",  # format_error: missing T separator (not ISO 8601)
        "promised_delivery_date": "15/01/2024",  # format_error: DD/MM/YYYY not YYYY-MM-DD
        "status": "refunded",                 # invalid_enum
        "customer": {
            "email": "customer@@example.com", # format_error: double @
            "phone": "555-1234",              # format_error: not E.164
            "loyalty_points": 1500,           # valid
        },
        "items": [
            {
                "sku": "SKU-AB1234",           # valid
                "unit_price": 0.00,           # format_error: below minimum 0.01
                "quantity": 2,
                "discount_rate": 0.1,
            },
            {
                "sku": "SKU-CD5678",           # valid
                "unit_price": 49.99,
                "quantity": 150,              # format_error: exceeds maximum 100
                "discount_rate": 0.15,
            },
        ],
        "billing": {
            "subtotal": 149.97,
            "tax_rate": 0.65,                 # format_error: exceeds maximum 0.5
            "total": 248.95,
        },
        "tracking_code": "TRACK123456789",    # format_error: wrong pattern
        "estimated_days": 5,
    }

    violations = [
        PlantedViolation(
            field_path="order_id",
            violation_type="format_error",
            description="'ORDER-123456' does not match required pattern ^ORD-[0-9]{6}$.",
            expected_value="string matching ^ORD-[0-9]{6}$",
            actual_value="'ORDER-123456'",
        ),
        PlantedViolation(
            field_path="created_at",
            violation_type="format_error",
            description="'2024-01-15 10:30:00' is not valid ISO 8601 date-time (missing T separator).",
            expected_value="date-time, e.g. 2024-01-15T10:30:00Z",
            actual_value="'2024-01-15 10:30:00'",
        ),
        PlantedViolation(
            field_path="promised_delivery_date",
            violation_type="format_error",
            description="'15/01/2024' is not valid ISO 8601 date (expected YYYY-MM-DD).",
            expected_value="date, e.g. 2024-01-15",
            actual_value="'15/01/2024'",
        ),
        PlantedViolation(
            field_path="status",
            violation_type="invalid_enum",
            description="Value 'refunded' not in enum [pending, processing, shipped, delivered, cancelled].",
            expected_value="one of: pending, processing, shipped, delivered, cancelled",
            actual_value="'refunded'",
        ),
        PlantedViolation(
            field_path="customer.email",
            violation_type="format_error",
            description="'customer@@example.com' is not a valid email (double @ symbol).",
            expected_value="string, format: email",
            actual_value="'customer@@example.com'",
        ),
        PlantedViolation(
            field_path="customer.phone",
            violation_type="format_error",
            description="'555-1234' does not match E.164 pattern ^\\+[1-9][0-9]{7,14}$.",
            expected_value=r"string matching ^\+[1-9][0-9]{7,14}$",
            actual_value="'555-1234'",
        ),
        PlantedViolation(
            field_path="items[0].unit_price",
            violation_type="format_error",
            description="'items[0].unit_price' is 0.0, below minimum of 0.01.",
            expected_value="number >= 0.01",
            actual_value="0.0",
        ),
        PlantedViolation(
            field_path="items[1].quantity",
            violation_type="format_error",
            description="'items[1].quantity' is 150, exceeds maximum of 100.",
            expected_value="integer, maximum: 100",
            actual_value="150",
        ),
        PlantedViolation(
            field_path="billing.tax_rate",
            violation_type="format_error",
            description="'billing.tax_rate' is 0.65, exceeds maximum of 0.5 (50%).",
            expected_value="number, maximum: 0.5",
            actual_value="0.65",
        ),
        PlantedViolation(
            field_path="tracking_code",
            violation_type="format_error",
            description="'TRACK123456789' does not match pattern ^[A-Z]{2}[0-9]{9}[A-Z]{2}$.",
            expected_value="string matching ^[A-Z]{2}[0-9]{9}[A-Z]{2}$",
            actual_value="'TRACK123456789'",
        ),
    ]

    return TaskScenario(
        task_name="validate_response_schema",
        task_description=_RESPONSE_TASK_DESCRIPTION,
        api_spec=_RESPONSE_SPEC,
        payload=payload,
        violations=violations,
        max_steps=25,
    )


def _response_scenario_b() -> TaskScenario:
    """Expert variant B: 10 different format/constraint violations — harder to spot."""
    payload: Dict[str, Any] = {
        "order_id": "ORD-12AB56",             # format_error: non-digits in numeric portion
        "created_at": "2024-13-01T10:30:00Z", # format_error: month 13 is invalid
        "promised_delivery_date": "2024-00-15", # format_error: month 0 is invalid
        "status": "returned",                  # invalid_enum
        "customer": {
            "email": "user@",                  # format_error: incomplete email, no domain
            "phone": "+14155551234",           # valid
            "loyalty_points": "1500",          # type_mismatch: string instead of integer
        },
        "items": [
            {
                "sku": "PROD-AB1234",          # format_error: wrong prefix (PROD vs SKU)
                "unit_price": 29.99,
                "quantity": 2,
                "discount_rate": 1.5,          # format_error: exceeds maximum 1.0
            },
            {
                "sku": "SKU-CD5678",           # valid
                "unit_price": 49.99,
                "quantity": 3,
                "discount_rate": 0.0,
            },
        ],
        "billing": {
            "subtotal": "99.50",              # type_mismatch: string instead of number
            "tax_rate": 0.18,
            "total": 117.41,
        },
        "tracking_code": "AB123456789CD",     # valid — matches ^[A-Z]{2}[0-9]{9}[A-Z]{2}$
        "estimated_days": 0,                  # format_error: below minimum 1
    }

    violations = [
        PlantedViolation(
            field_path="order_id",
            violation_type="format_error",
            description="'ORD-12AB56' does not match ^ORD-[0-9]{6}$ (non-digit chars 'AB').",
            expected_value="string matching ^ORD-[0-9]{6}$",
            actual_value="'ORD-12AB56'",
        ),
        PlantedViolation(
            field_path="created_at",
            violation_type="format_error",
            description="'2024-13-01T10:30:00Z' has invalid month 13 — not a valid date-time.",
            expected_value="date-time with valid calendar date",
            actual_value="'2024-13-01T10:30:00Z'",
        ),
        PlantedViolation(
            field_path="promised_delivery_date",
            violation_type="format_error",
            description="'2024-00-15' has month 0, which is not a valid calendar month.",
            expected_value="date with valid month (01–12)",
            actual_value="'2024-00-15'",
        ),
        PlantedViolation(
            field_path="status",
            violation_type="invalid_enum",
            description="Value 'returned' not in enum [pending, processing, shipped, delivered, cancelled].",
            expected_value="one of: pending, processing, shipped, delivered, cancelled",
            actual_value="'returned'",
        ),
        PlantedViolation(
            field_path="customer.email",
            violation_type="format_error",
            description="'user@' is not a valid email address (missing domain after @).",
            expected_value="string, format: email",
            actual_value="'user@'",
        ),
        PlantedViolation(
            field_path="customer.loyalty_points",
            violation_type="type_mismatch",
            description="Field 'customer.loyalty_points' should be integer, got string.",
            expected_value="integer",
            actual_value="string ('1500')",
        ),
        PlantedViolation(
            field_path="items[0].sku",
            violation_type="format_error",
            description="'PROD-AB1234' does not match required pattern ^SKU-[A-Z0-9]{6}$.",
            expected_value="string matching ^SKU-[A-Z0-9]{6}$",
            actual_value="'PROD-AB1234'",
        ),
        PlantedViolation(
            field_path="items[0].discount_rate",
            violation_type="format_error",
            description="'items[0].discount_rate' is 1.5, exceeds maximum of 1.0.",
            expected_value="number, maximum: 1.0",
            actual_value="1.5",
        ),
        PlantedViolation(
            field_path="billing.subtotal",
            violation_type="type_mismatch",
            description="Field 'billing.subtotal' should be number, got string.",
            expected_value="number",
            actual_value="string ('99.50')",
        ),
        PlantedViolation(
            field_path="estimated_days",
            violation_type="format_error",
            description="'estimated_days' is 0, below minimum of 1.",
            expected_value="integer, minimum: 1",
            actual_value="0",
        ),
    ]

    return TaskScenario(
        task_name="validate_response_schema",
        task_description=_RESPONSE_TASK_DESCRIPTION,
        api_spec=_RESPONSE_SPEC,
        payload=payload,
        violations=violations,
        max_steps=25,
    )


def generate_format_validation_scenario(seed: Optional[int] = None) -> TaskScenario:
    """Expert scenario: validate an API response for format/constraint violations.

    seed=None or even seed → variant A.
    Odd seed → variant B (different set of violations, same spec).
    """
    if seed is None or seed % 2 == 0:
        return _response_scenario_a()
    return _response_scenario_b()


# ── Expert Task 2 — cross-field constraint validation ─────────────────────
#
# Violations require multi-field reasoning: arithmetic, date ordering, and
# conditional requirements.  Standard schema validators cannot catch these;
# an agent must actively compute and cross-reference values.


def generate_cross_field_scenario(seed: Optional[int] = None) -> TaskScenario:
    """Expert scenario: 7 cross-field constraint violations.

    Requires the agent to:
    - Verify arithmetic (line_total = quantity * unit_price)
    - Check date ordering (due_date after invoice_date)
    - Validate computed totals (tax_amount = subtotal * tax_rate)
    - Enforce conditional rules (trial accounts: discount_amount = 0)
    - Count array elements (item_count = len(line_items))

    seed is accepted for API uniformity but not used (single canonical scenario).
    """
    api_spec: Dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {"title": "Invoice Service API", "version": "1.0.0"},
        "paths": {
            "/invoices": {
                "post": {
                    "summary": "Create a new invoice",
                    "description": (
                        "Cross-field constraints (not expressible in JSON Schema): "
                        "(1) due_date must be strictly after invoice_date. "
                        "(2) Each line_items[N].line_total must equal quantity * unit_price. "
                        "(3) billing.subtotal must equal the sum of all line_items[N].line_total values. "
                        "(4) billing.tax_amount must equal billing.subtotal * billing.tax_rate. "
                        "(5) billing.total must equal billing.subtotal + billing.tax_amount - billing.discount_amount. "
                        "(6) billing.item_count must equal the number of elements in the line_items array. "
                        "(7) If customer.account_type is 'trial', billing.discount_amount must be 0."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": [
                                        "invoice_date",
                                        "due_date",
                                        "customer",
                                        "line_items",
                                        "billing",
                                    ],
                                    "properties": {
                                        "invoice_date": {
                                            "type": "string",
                                            "format": "date",
                                        },
                                        "due_date": {
                                            "type": "string",
                                            "format": "date",
                                            "description": "Must be strictly after invoice_date",
                                        },
                                        "customer": {
                                            "type": "object",
                                            "required": ["id", "name", "account_type"],
                                            "properties": {
                                                "id": {"type": "integer"},
                                                "name": {"type": "string"},
                                                "account_type": {
                                                    "type": "string",
                                                    "enum": [
                                                        "trial",
                                                        "standard",
                                                        "premium",
                                                        "enterprise",
                                                    ],
                                                },
                                            },
                                        },
                                        "line_items": {
                                            "type": "array",
                                            "minItems": 1,
                                            "items": {
                                                "type": "object",
                                                "required": [
                                                    "description",
                                                    "quantity",
                                                    "unit_price",
                                                    "line_total",
                                                ],
                                                "properties": {
                                                    "description": {"type": "string"},
                                                    "quantity": {
                                                        "type": "integer",
                                                        "minimum": 1,
                                                    },
                                                    "unit_price": {
                                                        "type": "number",
                                                        "minimum": 0,
                                                    },
                                                    "line_total": {
                                                        "type": "number",
                                                        "description": "Must equal quantity * unit_price",
                                                    },
                                                },
                                            },
                                        },
                                        "billing": {
                                            "type": "object",
                                            "required": [
                                                "subtotal",
                                                "tax_rate",
                                                "tax_amount",
                                                "discount_amount",
                                                "total",
                                                "item_count",
                                                "currency",
                                            ],
                                            "properties": {
                                                "subtotal": {
                                                    "type": "number",
                                                    "minimum": 0,
                                                    "description": "Sum of all line_items[N].line_total",
                                                },
                                                "tax_rate": {
                                                    "type": "number",
                                                    "minimum": 0,
                                                    "maximum": 1,
                                                },
                                                "tax_amount": {
                                                    "type": "number",
                                                    "description": "Must equal subtotal * tax_rate",
                                                },
                                                "discount_amount": {
                                                    "type": "number",
                                                    "minimum": 0,
                                                    "description": "Must be 0 when customer.account_type is 'trial'",
                                                },
                                                "total": {
                                                    "type": "number",
                                                    "description": "Must equal subtotal + tax_amount - discount_amount",
                                                },
                                                "item_count": {
                                                    "type": "integer",
                                                    "description": "Must equal number of elements in line_items",
                                                },
                                                "currency": {
                                                    "type": "string",
                                                    "enum": ["USD", "EUR", "GBP", "INR"],
                                                },
                                            },
                                        },
                                    },
                                }
                            }
                        },
                    },
                }
            }
        },
    }

    # Planted violations (7):
    # 1  due_date before invoice_date
    # 2  line_items[0].line_total arithmetic wrong  (80 ≠ 3 × 25)
    # 3  billing.subtotal wrong                     (200 ≠ 80+49.99+60 = 189.99)
    # 4  billing.tax_amount wrong                   (14.40 ≠ 200 × 0.08 = 16.00)
    # 5  billing.item_count wrong                   (4 ≠ 3)
    # 6  billing.discount_amount > 0 for trial      (25 ≠ 0)
    # 7  billing.total inconsistent                 (195 ≠ 200+14.40-25 = 189.40)
    payload: Dict[str, Any] = {
        "invoice_date": "2024-03-15",
        "due_date": "2024-03-10",        # VIOLATION 1 — before invoice_date
        "customer": {
            "id": 1042,
            "name": "Acme Corp",
            "account_type": "trial",     # triggers VIOLATION 6
        },
        "line_items": [
            {
                "description": "Cloud Storage 100 GB/mo",
                "quantity": 3,
                "unit_price": 25.00,
                "line_total": 80.00,     # VIOLATION 2 — should be 3 × 25.00 = 75.00
            },
            {
                "description": "API Calls 1M/mo",
                "quantity": 1,
                "unit_price": 49.99,
                "line_total": 49.99,     # valid
            },
            {
                "description": "Support Package",
                "quantity": 2,
                "unit_price": 30.00,
                "line_total": 60.00,     # valid
            },
        ],
        "billing": {
            "subtotal": 200.00,          # VIOLATION 3 — sum of line_totals = 189.99
            "tax_rate": 0.08,
            "tax_amount": 14.40,         # VIOLATION 4 — should be 200.00 × 0.08 = 16.00
            "discount_amount": 25.00,   # VIOLATION 6 — trial account; must be 0
            "total": 195.00,             # VIOLATION 7 — should be 200+14.40-25 = 189.40
            "item_count": 4,             # VIOLATION 5 — 3 line_items, not 4
            "currency": "USD",
        },
    }

    violations = [
        PlantedViolation(
            field_path="due_date",
            violation_type="cross_field_constraint",
            description=(
                "due_date '2024-03-10' is before invoice_date '2024-03-15'; "
                "due_date must be strictly after invoice_date."
            ),
            expected_value="date after 2024-03-15",
            actual_value="2024-03-10",
        ),
        PlantedViolation(
            field_path="line_items[0].line_total",
            violation_type="cross_field_constraint",
            description=(
                "line_items[0].line_total is 80.00 but "
                "quantity(3) × unit_price(25.00) = 75.00."
            ),
            expected_value="75.00",
            actual_value="80.00",
        ),
        PlantedViolation(
            field_path="billing.subtotal",
            violation_type="cross_field_constraint",
            description=(
                "billing.subtotal is 200.00 but the sum of line_totals "
                "(80.00 + 49.99 + 60.00) = 189.99."
            ),
            expected_value="189.99",
            actual_value="200.00",
        ),
        PlantedViolation(
            field_path="billing.tax_amount",
            violation_type="cross_field_constraint",
            description=(
                "billing.tax_amount is 14.40 but "
                "billing.subtotal(200.00) × tax_rate(0.08) = 16.00."
            ),
            expected_value="16.00",
            actual_value="14.40",
        ),
        PlantedViolation(
            field_path="billing.item_count",
            violation_type="cross_field_constraint",
            description=(
                "billing.item_count is 4 but there are 3 elements in line_items."
            ),
            expected_value="3",
            actual_value="4",
        ),
        PlantedViolation(
            field_path="billing.discount_amount",
            violation_type="cross_field_constraint",
            description=(
                "billing.discount_amount is 25.00 but customer.account_type "
                "is 'trial'; trial accounts must have discount_amount = 0."
            ),
            expected_value="0",
            actual_value="25.00",
        ),
        PlantedViolation(
            field_path="billing.total",
            violation_type="cross_field_constraint",
            description=(
                "billing.total is 195.00 but "
                "subtotal(200.00) + tax_amount(14.40) - discount_amount(25.00) = 189.40."
            ),
            expected_value="189.40",
            actual_value="195.00",
        ),
    ]

    return TaskScenario(
        task_name="validate_cross_field_constraints",
        task_description=(
            "You are given an API spec for POST /invoices and a request payload. "
            "The spec defines cross-field constraints that cannot be checked by "
            "standard JSON Schema validation. You must actively compute and "
            "cross-reference values to find violations. "
            "Constraints to check: "
            "(1) due_date must be strictly after invoice_date. "
            "(2) Each line_items[N].line_total must equal quantity × unit_price. "
            "(3) billing.subtotal must equal the sum of all line_items[N].line_total. "
            "(4) billing.tax_amount must equal billing.subtotal × billing.tax_rate. "
            "(5) billing.total must equal billing.subtotal + billing.tax_amount - billing.discount_amount. "
            "(6) billing.item_count must equal the number of elements in line_items. "
            "(7) If customer.account_type is 'trial', billing.discount_amount must be 0. "
            "Use violation_type='cross_field_constraint' for these. "
            "Report one violation per step. Submit field_path='DONE' when finished."
        ),
        api_spec=api_spec,
        payload=payload,
        violations=violations,
        max_steps=18,
    )


# ── Expert Task 3 — authentication & security schema validation ───────────
#
# Validates a complex authentication API payload covering OAuth2 tokens,
# permission scopes, rate limits, and security constraints.  Two variants
# selected by seed parity.


def _auth_scenario_a() -> TaskScenario:
    """Auth scenario A: OAuth2 token introspection response with 6 violations."""
    api_spec: Dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {"title": "Auth Service API", "version": "2.0.0"},
        "paths": {
            "/auth/token": {
                "post": {
                    "summary": "Issue an OAuth2 access token",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["client_id", "client_secret", "grant_type", "scope"],
                                    "properties": {
                                        "client_id": {
                                            "type": "string",
                                            "pattern": "^[a-zA-Z0-9_-]{8,64}$",
                                        },
                                        "client_secret": {
                                            "type": "string",
                                            "minLength": 32,
                                        },
                                        "grant_type": {
                                            "type": "string",
                                            "enum": [
                                                "authorization_code",
                                                "client_credentials",
                                                "refresh_token",
                                                "password",
                                            ],
                                        },
                                        "scope": {
                                            "type": "array",
                                            "minItems": 1,
                                            "items": {
                                                "type": "string",
                                                "enum": [
                                                    "read:users",
                                                    "write:users",
                                                    "read:orders",
                                                    "write:orders",
                                                    "admin",
                                                ],
                                            },
                                        },
                                        "expires_in": {
                                            "type": "integer",
                                            "minimum": 60,
                                            "maximum": 86400,
                                        },
                                        "redirect_uri": {
                                            "type": "string",
                                            "format": "uri",
                                        },
                                        "mfa_token": {
                                            "type": "string",
                                            "pattern": "^[0-9]{6}$",
                                            "description": "6-digit numeric MFA code",
                                        },
                                    },
                                }
                            }
                        },
                    },
                }
            }
        },
    }

    payload: Dict[str, Any] = {
        "client_id": "abc",
        "client_secret": "short",
        "grant_type": "implicit",
        "scope": ["read:users", "delete:everything"],
        "expires_in": 0,
        "redirect_uri": "not-a-valid-uri",
        "mfa_token": "12AB56",
    }

    violations = [
        PlantedViolation(
            field_path="client_id",
            violation_type="format_error",
            description="'abc' has length 3, below minLength pattern requirement of 8 characters.",
            expected_value="string matching ^[a-zA-Z0-9_-]{8,64}$",
            actual_value="'abc'",
        ),
        PlantedViolation(
            field_path="client_secret",
            violation_type="format_error",
            description="'short' has length 5, below minLength of 32.",
            expected_value="string, minLength: 32",
            actual_value="'short' (length 5)",
        ),
        PlantedViolation(
            field_path="grant_type",
            violation_type="invalid_enum",
            description="'implicit' not in allowed enum [authorization_code, client_credentials, refresh_token, password].",
            expected_value="one of: authorization_code, client_credentials, refresh_token, password",
            actual_value="'implicit'",
        ),
        PlantedViolation(
            field_path="scope[1]",
            violation_type="invalid_enum",
            description="'delete:everything' not in allowed scope enum.",
            expected_value="one of: read:users, write:users, read:orders, write:orders, admin",
            actual_value="'delete:everything'",
        ),
        PlantedViolation(
            field_path="expires_in",
            violation_type="format_error",
            description="'expires_in' is 0, below minimum of 60.",
            expected_value="integer, minimum: 60",
            actual_value="0",
        ),
        PlantedViolation(
            field_path="mfa_token",
            violation_type="format_error",
            description="'12AB56' contains letters; must match ^[0-9]{6}$ (6 digits only).",
            expected_value="string matching ^[0-9]{6}$",
            actual_value="'12AB56'",
        ),
    ]

    return TaskScenario(
        task_name="validate_auth_request",
        task_description=(
            "You are given an OpenAPI specification for POST /auth/token (OAuth2 token issuance) "
            "and an API request payload. Find all violations: invalid enum values for grant_type "
            "and scope items, format/pattern violations (client_id pattern, client_secret length, "
            "mfa_token digits-only, redirect_uri URI format), and out-of-range numeric values. "
            "Use dot-notation for nested paths and bracket notation for arrays (e.g. 'scope[1]'). "
            "Submit field_path='DONE' when finished."
        ),
        api_spec=api_spec,
        payload=payload,
        violations=violations,
        max_steps=14,
    )


def _auth_scenario_b() -> TaskScenario:
    """Auth scenario B: API key management request with 6 different violations."""
    api_spec: Dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {"title": "API Key Management", "version": "1.0.0"},
        "paths": {
            "/api-keys": {
                "post": {
                    "summary": "Create a new API key",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["name", "permissions", "environment", "ttl_days"],
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "minLength": 3,
                                            "maxLength": 50,
                                            "pattern": "^[a-z0-9-]+$",
                                            "description": "Lowercase alphanumeric with hyphens only",
                                        },
                                        "permissions": {
                                            "type": "array",
                                            "minItems": 1,
                                            "maxItems": 5,
                                            "items": {
                                                "type": "string",
                                                "enum": ["read", "write", "delete", "admin"],
                                            },
                                        },
                                        "environment": {
                                            "type": "string",
                                            "enum": ["development", "staging", "production"],
                                        },
                                        "ttl_days": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 365,
                                        },
                                        "ip_whitelist": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "format": "ipv4",
                                            },
                                        },
                                        "rate_limit": {
                                            "type": "integer",
                                            "minimum": 10,
                                            "maximum": 10000,
                                            "description": "Requests per minute",
                                        },
                                    },
                                }
                            }
                        },
                    },
                }
            }
        },
    }

    payload: Dict[str, Any] = {
        "name": "My API Key!",
        "permissions": ["read", "write", "superuser"],
        "environment": "local",
        "ttl_days": 0,
        "ip_whitelist": ["192.168.1.1", "999.0.0.1"],
        "rate_limit": 5,
    }

    violations = [
        PlantedViolation(
            field_path="name",
            violation_type="format_error",
            description="'My API Key!' contains spaces and '!'; must match ^[a-z0-9-]+$ (lowercase, digits, hyphens only).",
            expected_value="string matching ^[a-z0-9-]+$",
            actual_value="'My API Key!'",
        ),
        PlantedViolation(
            field_path="permissions[2]",
            violation_type="invalid_enum",
            description="'superuser' not in allowed enum [read, write, delete, admin].",
            expected_value="one of: read, write, delete, admin",
            actual_value="'superuser'",
        ),
        PlantedViolation(
            field_path="environment",
            violation_type="invalid_enum",
            description="'local' not in enum [development, staging, production].",
            expected_value="one of: development, staging, production",
            actual_value="'local'",
        ),
        PlantedViolation(
            field_path="ttl_days",
            violation_type="format_error",
            description="'ttl_days' is 0, below minimum of 1.",
            expected_value="integer, minimum: 1",
            actual_value="0",
        ),
        PlantedViolation(
            field_path="ip_whitelist[1]",
            violation_type="format_error",
            description="'999.0.0.1' is not a valid IPv4 address (first octet 999 > 255).",
            expected_value="string, format: ipv4",
            actual_value="'999.0.0.1'",
        ),
        PlantedViolation(
            field_path="rate_limit",
            violation_type="format_error",
            description="'rate_limit' is 5, below minimum of 10.",
            expected_value="integer, minimum: 10",
            actual_value="5",
        ),
    ]

    return TaskScenario(
        task_name="validate_auth_request",
        task_description=(
            "You are given an OpenAPI specification for POST /api-keys (API key creation) "
            "and a request payload. Find all violations: invalid enum values for permissions "
            "items and environment, format/pattern violations (name pattern, ip_whitelist format), "
            "and out-of-range numeric values (ttl_days, rate_limit). "
            "Use dot-notation for nested paths and bracket notation for arrays (e.g. 'permissions[2]'). "
            "Submit field_path='DONE' when finished."
        ),
        api_spec=api_spec,
        payload=payload,
        violations=violations,
        max_steps=14,
    )


def generate_auth_scenario(seed: Optional[int] = None) -> TaskScenario:
    """Auth scenario: two variants selected by seed parity.

    seed=None or even → OAuth2 token (variant A).
    Odd seed → API key management (variant B).
    """
    if seed is None or seed % 2 == 0:
        return _auth_scenario_a()
    return _auth_scenario_b()


# ── Registry ──────────────────────────────────────────────────────────────

TASK_GENERATORS = {
    "find_type_mismatches": generate_easy_scenario,
    "validate_nested_objects": generate_medium_scenario,
    "detect_breaking_changes": generate_hard_scenario,
    "validate_response_schema": generate_format_validation_scenario,
    "validate_cross_field_constraints": generate_cross_field_scenario,
    "validate_auth_request": generate_auth_scenario,
}

AVAILABLE_TASKS = list(TASK_GENERATORS.keys())


def generate_scenario_for_task(
    task_name: str, seed: Optional[int] = None
) -> TaskScenario:
    """Return a ``TaskScenario`` for the requested task.

    Parameters
    ----------
    task_name:
        One of the keys in ``AVAILABLE_TASKS``.
    seed:
        Optional integer seed for deterministic randomisation.
        seed=None → canonical fixed scenario (backward-compatible).
        seed=int  → reproducible randomised variant.

    Raises
    ------
    ValueError
        If *task_name* is not recognised.
    """
    generator = TASK_GENERATORS.get(task_name)
    if generator is None:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {AVAILABLE_TASKS}"
        )
    return generator(seed=seed)
