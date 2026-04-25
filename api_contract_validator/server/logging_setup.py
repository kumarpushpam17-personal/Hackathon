"""Structured JSON logging for API Contract Validator.

Outputs one JSON object per log record to stdout (visible in docker logs)
and, when writable, to logs/episodes.jsonl for persistent episode history.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    """Configure root logger: JSON to stdout + logs/episodes.jsonl."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. during testing)

    root.setLevel(log_level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_JsonFormatter())
    root.addHandler(sh)

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "episodes.jsonl", encoding="utf-8")
        fh.setFormatter(_JsonFormatter())
        root.addHandler(fh)
    except OSError:
        pass  # read-only filesystem (HF Spaces free tier) — stdout only
