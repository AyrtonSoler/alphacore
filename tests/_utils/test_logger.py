"""RED tests for common.logger — structured JSON logging config.

Acceptance: log records include service, timestamp, runbook_url fields.
"""

from __future__ import annotations

import io
import json

from alphacore._utils.logger import configure_logging, get_logger


def test_log_record_includes_service_timestamp_runbook() -> None:
    """A single info() call emits one JSON line carrying the required fields."""
    buf = io.StringIO()
    configure_logging(service="research-lab", runbook_url="https://rb/x", stream=buf)

    log = get_logger("test.logger")
    log.info("pipeline_started", step="curator")

    record = json.loads(buf.getvalue())
    assert record["service"] == "research-lab"
    assert record["runbook_url"] == "https://rb/x"
    assert record["timestamp"]  # present and non-empty
    assert record["event"] == "pipeline_started"
    assert record["step"] == "curator"


def test_log_record_carries_level() -> None:
    """The structured record exposes the log level so JSON sinks can filter on it."""
    buf = io.StringIO()
    configure_logging(service="research-lab", runbook_url="https://rb/x", stream=buf)

    log = get_logger("test.logger")
    log.warning("slow_query")

    record = json.loads(buf.getvalue())
    assert record["level"] == "warning"
