"""Structured JSON logging for the research lab.

Every log record is rendered as a single JSON line carrying the originating
``service`` name, an ISO-8601 ``timestamp``, a ``runbook_url`` for on-call triage,
and the log ``level``. JSON output keeps records machine-parseable by downstream
sinks (CI logs, the cockpit, the Daily Brief) without bespoke parsers.

Notes
-----
Configuration is process-global (``structlog.configure``). Call
:func:`configure_logging` once at an I/O boundary (CLI entrypoint, app startup),
never inside pure logic.

"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast

import structlog

if TYPE_CHECKING:
    from typing import TextIO

    from structlog.typing import EventDict, WrappedLogger


def _make_context_processor(service: str, runbook_url: str) -> structlog.typing.Processor:
    """Build a structlog processor that stamps every record with static context.

    Parameters
    ----------
    service:
        Logical service name emitting the record (e.g. ``"research-lab"``).
    runbook_url:
        Link an on-call engineer follows when this service misbehaves.

    Returns
    -------
    structlog.typing.Processor
        A processor injecting ``service`` and ``runbook_url`` into each event.

    """

    def processor(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
        event_dict["service"] = service
        event_dict["runbook_url"] = runbook_url
        return event_dict

    return processor


def configure_logging(service: str, runbook_url: str, *, stream: TextIO | None = None) -> None:
    """Configure process-global structured JSON logging.

    Parameters
    ----------
    service:
        Logical service name stamped onto every record.
    runbook_url:
        On-call runbook link stamped onto every record.
    stream:
        Destination text stream. Defaults to ``sys.stdout``; tests pass a
        :class:`io.StringIO` to capture output.

    Returns
    -------
    None

    Notes
    -----
    ``cache_logger_on_first_use=False`` so repeated reconfiguration (as in tests)
    takes effect immediately rather than binding to the first-seen config.

    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _make_context_processor(service, runbook_url),
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stdout),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for ``name``.

    Parameters
    ----------
    name:
        Logger name, conventionally ``__name__`` of the calling module.

    Returns
    -------
    structlog.stdlib.BoundLogger
        A logger honouring the active :func:`configure_logging` settings.

    """
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))
