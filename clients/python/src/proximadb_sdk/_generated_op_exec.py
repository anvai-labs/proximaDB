"""One outcome vocabulary for every generated openapi op the adapters run.

Extracted from the document family (TD-SDK-3 batch 2) so timeseries and
observability (batch 3) share the same executor instead of cloning it:
transport failures → :class:`NetworkError`, documented error envelopes and
undocumented statuses → :class:`ServerError` (with the status), malformed
bodies on documented statuses → :class:`ServerError`, and a 404 → the
caller's not-found contract (``get``-style ``None`` or delete-style
``False``) on BOTH the parsed-``ErrorResponse`` path and the
``UnexpectedStatus`` path (ops that do not document 404 raise before a
response is parsed). The generated exceptions never escape the SDK.
"""

from __future__ import annotations

from typing import Any

import httpx

from ._generated.rest.errors import UnexpectedStatus
from ._generated.rest.models.error_response import ErrorResponse
from .exceptions import NetworkError, ProximaDBError, ServerError

#: Sentinel distinguishing "no not-found mapping configured" from a
#: legitimate ``None`` not-found value.
NOT_FOUND_UNSET = object()


def map_op_response(
    op: str,
    response: Any,
    *,
    not_found: Any = NOT_FOUND_UNSET,
) -> dict[str, Any]:
    """Map a generated op ``Response`` to a dict, or raise.

    ``ErrorResponse`` → :class:`ServerError` (with the documented detail
    and status); no body → :class:`ServerError`. A 404 maps to
    ``not_found`` when the caller configured one (``get`` → ``None``; the
    deletes pass an ack-shaped ``{"success": False}`` and surface
    ``False``) — the established not-found contracts.
    """
    parsed = response.parsed
    if isinstance(parsed, ErrorResponse):
        if response.status_code == 404 and not_found is not NOT_FOUND_UNSET:
            return not_found
        detail = getattr(getattr(parsed, "error", None), "message", None)
        raise ServerError(
            f"{op} rejected by server" f"{f': {detail}' if detail else ''}",
            status_code=response.status_code,
        )
    if parsed is None:
        # raise_on_unexpected_status=True makes this unreachable in practice;
        # guard anyway so a future regression can't silently return None.
        raise ServerError(
            f"{op} returned no body",
            status_code=response.status_code,
        )
    return parsed.to_dict()


def execute_generated_op(
    op: str,
    call: Any,
    *,
    not_found: Any = NOT_FOUND_UNSET,
):
    """Run a generated op and map outcomes to SDK exceptions.

    ``call`` is a zero-argument callable invoking the generated
    ``sync_detailed`` (attribute lookup happens at call time so the
    offline test seam — patching the generated module — keeps working).
    """
    try:
        response = call()
    except httpx.HTTPError as e:
        raise NetworkError(f"{op} transport error: {e}", original_error=e) from e
    except UnexpectedStatus as e:
        if e.status_code == 404 and not_found is not NOT_FOUND_UNSET:
            return not_found
        # raise_on_unexpected_status=True raises BEFORE the response is
        # returned, so undocumented statuses (e.g. HTTP 500) map to
        # ServerError here — the generated exception never escapes the SDK.
        raise ServerError(
            f"{op} unexpected status: {e.status_code}",
            status_code=e.status_code,
        ) from e
    except (KeyError, TypeError) as e:
        # Deferred request-model construction (bodies build inside the
        # callables) surfaces missing/ill-typed REQUIRED fields here — map
        # to the SDK vocabulary instead of a raw KeyError/TypeError.
        raise ProximaDBError(f"{op} request payload is malformed: {e}") from e
    except ValueError as e:
        # json.JSONDecodeError subclasses ValueError: a non-JSON body on a
        # DOCUMENTED status (LB/gateway garbage at 500) must not escape as
        # a decoding error — it is still a server-side failure.
        raise ServerError(
            f"{op} returned a malformed body: {e}",
            status_code=None,
        ) from e
    return map_op_response(op, response, not_found=not_found)
