"""
FastAPI application for the API Contract Validator Environment.

Exposes the ValidatorEnvironment over HTTP and WebSocket endpoints
using ``openenv.core.env_server.http_server.create_app``.

Endpoints created automatically:
    - GET  /        — Landing page (this module)
    - POST /reset   — Reset the environment
    - POST /step    — Execute an action
    - GET  /state   — Get current environment state
    - GET  /health  — Health check
    - WS   /ws      — WebSocket for persistent sessions
    - GET  /docs    — Swagger UI (interactive — try every endpoint)
"""

from fastapi.responses import HTMLResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from exc

try:
    from ..models import ValidatorAction, ValidatorObservation
    from .environment import ValidatorEnvironment
    from .logging_setup import configure_logging
except (ImportError, ModuleNotFoundError):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import ValidatorAction, ValidatorObservation
    from server.environment import ValidatorEnvironment
    from server.logging_setup import configure_logging

configure_logging()

app = create_app(
    ValidatorEnvironment,
    ValidatorAction,
    ValidatorObservation,
    env_name="api_contract_validator",
    max_concurrent_envs=10,
)


_LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Enterprise Contract Guardian — OpenEnv</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         margin: 0; padding: 2rem; background: #0b1020; color: #e7ecf3; line-height: 1.55; }
  .container { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
  h2 { font-size: 1.05rem; margin: 1.6rem 0 .5rem; color: #9bb0d6;
       text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
  .tag { display: inline-block; padding: 2px 10px; border-radius: 999px;
         background: #16a34a; color: white; font-size: 0.78rem; font-weight: 600;
         margin-left: .4rem; vertical-align: middle; }
  a { color: #6db9ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  code { background: #1c2540; padding: 2px 6px; border-radius: 4px; font-size: .9em; }
  pre { background: #1c2540; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: .88em; }
  table { width: 100%; border-collapse: collapse; margin-top: .5rem; }
  td, th { padding: .55rem .75rem; text-align: left;
           border-bottom: 1px solid #1c2540; font-size: .92em; vertical-align: top; }
  th { color: #9bb0d6; font-weight: 600; font-size: .78em;
       text-transform: uppercase; letter-spacing: .04em; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .card { background: #131a35; padding: 1rem 1.2rem; border-radius: 8px;
          border: 1px solid #1c2540; }
  .footer { margin-top: 2rem; color: #7c8db0; font-size: .82em; }
  .pill { display:inline-block; padding:1px 8px; border-radius:6px;
          background:#1c2540; font-size:.78em; color:#9bb0d6; margin-right:4px; }
</style>
</head>
<body>
<div class="container">
  <h1>Enterprise Contract Guardian <span class="tag">RUNNING</span></h1>
  <p>An <b>OpenEnv</b> RL environment that trains agents to detect API contract
  violations, trace downstream blast radius across microservices, and propose
  backward-compatible fixes.</p>
  <p style="color:#9bb0d6;">
    <span class="pill">Theme #3.1</span>
    <span class="pill">Scaler AI Labs Bonus</span>
    <span class="pill">openenv-core 0.2.3</span>
    <span class="pill">9 tasks · 3 phases · 14 reward signals</span>
  </p>

  <h2>Endpoints</h2>
  <table>
    <tr><th>Method</th><th>Path</th><th>Purpose</th></tr>
    <tr><td>GET</td><td><a href="/health">/health</a></td><td>Liveness check</td></tr>
    <tr><td>GET</td><td><a href="/docs">/docs</a></td>
        <td>Interactive Swagger UI &mdash; click "Try it out" on any endpoint</td></tr>
    <tr><td>POST</td><td><code>/reset</code></td>
        <td>Start a new episode (optional <code>task_name</code>, <code>seed</code>)</td></tr>
    <tr><td>POST</td><td><code>/step</code></td>
        <td>Submit one ValidatorAction</td></tr>
    <tr><td>GET</td><td><code>/state</code></td>
        <td>Current environment state</td></tr>
    <tr><td>WS</td><td><code>/ws</code></td>
        <td>WebSocket session</td></tr>
  </table>

  <h2>Try it from your terminal</h2>
  <pre>HOST=https://pushpam14-api-contract-validator.hf.space

curl $HOST/health
# {"status":"healthy"}

curl -X POST $HOST/reset -H "Content-Type: application/json" \\
     -d '{"task_name":"trace_downstream_blast_radius","seed":1}'</pre>

  <h2>Tasks</h2>
  <div class="grid">
    <div class="card"><b>Phase 1 &mdash; Detection</b><br>
      <code>find_type_mismatches</code><br>
      <code>validate_nested_objects</code><br>
      <code>detect_breaking_changes</code><br>
      <code>validate_response_schema</code><br>
      <code>validate_cross_field_constraints</code><br>
      <code>validate_auth_request</code></div>
    <div class="card"><b>Phase 2 &mdash; Impact Tracing</b><br>
      <code>trace_downstream_blast_radius</code><br><br>
      <b>Phase 3 &mdash; Fix &amp; Verify</b><br>
      <code>propose_backward_compat_fix</code><br>
      <code>multi_service_cascade_fix</code></div>
  </div>

  <h2>Source &amp; documentation</h2>
  <p>
    <a href="https://github.com/kumarpushpam17-personal/Hackathon">GitHub repo</a>
    &nbsp;·&nbsp;
    <a href="/docs">Swagger UI (interactive)</a>
    &nbsp;·&nbsp;
    <a href="/redoc">ReDoc API spec</a>
  </p>

  <p class="footer">Built with <code>openenv-core</code>, FastAPI, and the
  OpenEnv <i>composable rubric</i> pattern. Reward signal stays independent
  per phase, so per-component contributions are visible in training logs.</p>
</div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> HTMLResponse:
    """Landing page — what the HF Space iframe shows by default."""
    return HTMLResponse(content=_LANDING_HTML)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Entry point for ``uv run server`` or direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
