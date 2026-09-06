"""
Request-ID middleware.

Attaches a unique request ID to every incoming request (reusing an
`X-Request-ID` header if the caller/load-balancer already set one), stores
it on `request.state` for use in route handlers, injects it into log
records via a logging filter, and echoes it back in the response headers.

This is the difference between "a prediction failed somewhere" and
"prediction abc-123 failed" when correlating client reports with server
logs in production.
"""
from __future__ import annotations

import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"


class _RequestIdLogFilter(logging.Filter):
    """Injects the current request ID (if any) into every log record emitted
    during request handling, so JSON logs can be correlated by request."""

    def __init__(self):
        super().__init__()
        self.request_id: str | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = self.request_id
        return True


_log_filter = _RequestIdLogFilter()
logging.getLogger().addFilter(_log_filter)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))
        request.state.request_id = request_id
        _log_filter.request_id = request_id

        response: Response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id

        _log_filter.request_id = None
        return response
