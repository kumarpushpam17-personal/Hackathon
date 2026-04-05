"""
Spec generator for the API Contract Validator Environment.

Generates OpenAPI specifications, payloads with planted violations,
and ground-truth violation records for three difficulty levels.

Each generator returns a ``TaskScenario`` containing everything the
environment needs for one episode.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


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


# ── Easy Task ─────────────────────────────────────────────────────────────


def generate_easy_scenario() -> TaskScenario:
    """Generate a scenario with obvious type mismatches and missing fields.

    Violations are shallow (top-level fields) and easy to spot:
    - String where integer is expected
    - Missing required fields
    - Wrong primitive types
    """
    api_spec = {
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
                                        "username": {"type": "string", "minLength": 3},
                                        "email": {"type": "string", "format": "email"},
                                        "age": {"type": "integer", "minimum": 0},
                                        "is_active": {"type": "boolean"},
                                        "role": {
                                            "type": "string",
                                            "enum": ["admin", "editor", "viewer"],
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

    payload = {
        "username": "ab",
        "age": "twenty-five",
        "is_active": "yes",
        "role": "superadmin",
    }

    violations = [
        PlantedViolation(
            field_path="email",
            violation_type="missing_required",
            description="Required field 'email' is missing from the payload.",
            expected_value="(present, type: string)",
            actual_value="(missing)",
        ),
        PlantedViolation(
            field_path="age",
            violation_type="type_mismatch",
            description="Field 'age' should be integer but received string.",
            expected_value="integer",
            actual_value="string ('twenty-five')",
        ),
        PlantedViolation(
            field_path="is_active",
            violation_type="type_mismatch",
            description="Field 'is_active' should be boolean but received string.",
            expected_value="boolean",
            actual_value="string ('yes')",
        ),
        PlantedViolation(
            field_path="role",
            violation_type="invalid_enum",
            description="Field 'role' value 'superadmin' is not in allowed enum.",
            expected_value="one of: admin, editor, viewer",
            actual_value="superadmin",
        ),
    ]

    return TaskScenario(
        task_name="find_type_mismatches",
        task_description=(
            "You are given an OpenAPI specification and an API request payload. "
            "Find all violations in the payload: type mismatches, missing required "
            "fields, and invalid enum values. Report one violation per step using "
            "the field's dot-notation path. Submit field_path='DONE' when finished."
        ),
        api_spec=api_spec,
        payload=payload,
        violations=violations,
        max_steps=10,
    )


# ── Medium Task ───────────────────────────────────────────────────────────


def generate_medium_scenario() -> TaskScenario:
    """Generate a scenario with nested object and array violations.

    Violations span nested objects and arrays:
    - Missing required fields inside nested objects
    - Wrong types in array items
    - Invalid enum in nested context
    - Extra unexpected fields
    """
    api_spec = {
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
                                                    "product_id": {
                                                        "type": "string"
                                                    },
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

    payload = {
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
            expected_value="(present, type: string)",
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


# ── Hard Task ─────────────────────────────────────────────────────────────


def generate_hard_scenario() -> TaskScenario:
    """Generate a scenario requiring detection of breaking API changes.

    The agent receives two spec versions and must identify breaking changes:
    - Removed fields that were previously available
    - Type changes on existing fields
    - Narrowed enums (values removed)
    - New required fields added (breaking for existing clients)
    - Changed format constraints
    """
    api_spec = {
        "description": "Compare v1 (old) and v2 (new) of the Product Catalog API. "
        "Identify all BREAKING changes that would cause existing v1 "
        "clients to fail when calling the v2 API.",
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
                                            "tags": {
                                                "type": "integer",
                                            },
                                            "sku": {"type": "string"},
                                            "weight_grams": {
                                                "type": "integer"
                                            },
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

    # For the hard task the "payload" carries a sample v1 request that would
    # break under v2 — it shows the agent what a real client is sending.
    payload = {
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
            description="Field 'weight_kg' removed and replaced by 'weight_grams' with different type.",
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
            description="Response field 'discount_percent' removed in v2 — clients depending on it will break.",
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
            "fields. Use the format 'METHOD /path.field' for paths (e.g. "
            "'POST /products.price'). Submit field_path='DONE' when finished."
        ),
        api_spec=api_spec,
        payload=payload,
        violations=violations,
        max_steps=20,
    )


# ── Registry ──────────────────────────────────────────────────────────────

TASK_GENERATORS = {
    "find_type_mismatches": generate_easy_scenario,
    "validate_nested_objects": generate_medium_scenario,
    "detect_breaking_changes": generate_hard_scenario,
}

AVAILABLE_TASKS = list(TASK_GENERATORS.keys())


def generate_scenario_for_task(task_name: str) -> TaskScenario:
    """Return a ``TaskScenario`` for the requested task.

    Raises ``ValueError`` if *task_name* is not recognised.
    """
    generator = TASK_GENERATORS.get(task_name)
    if generator is None:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {AVAILABLE_TASKS}"
        )
    return generator()
