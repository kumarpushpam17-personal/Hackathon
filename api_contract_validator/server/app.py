"""
FastAPI application for the API Contract Validator Environment.

Exposes the ValidatorEnvironment over HTTP and WebSocket endpoints
using ``openenv.core.env_server.http_server.create_app``.

Endpoints created automatically:
    - POST /reset   — Reset the environment
    - POST /step    — Execute an action
    - GET  /state   — Get current environment state
    - GET  /health  — Health check
    - WS   /ws      — WebSocket for persistent sessions
    - GET  /docs    — Swagger UI
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from exc

try:
    from ..models import ValidatorAction, ValidatorObservation
    from .environment import ValidatorEnvironment
except (ImportError, ModuleNotFoundError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import ValidatorAction, ValidatorObservation
    from server.environment import ValidatorEnvironment


app = create_app(
    ValidatorEnvironment,
    ValidatorAction,
    ValidatorObservation,
    env_name="api_contract_validator",
    max_concurrent_envs=10,
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Entry point for ``uv run server`` or direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
