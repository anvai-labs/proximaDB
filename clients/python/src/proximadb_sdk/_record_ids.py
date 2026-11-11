"""Client-side default record IDs.

A record written without an ``id`` gets a fresh UUID4 here. The SDK used to
fill a missing id with ``f"record_{index}"`` (its position in the batch), so
two separate ID-less inserts both wrote ``record_0`` and the second silently
overwrote the first. A UUID is unique per call on every transport (REST,
gRPC, embedded, local fallback) and the caller can read it back from the
payload it sent.
"""

from __future__ import annotations

import uuid


def new_record_id() -> str:
    """Return a fresh record id for a record that was written without one."""
    return str(uuid.uuid4())
