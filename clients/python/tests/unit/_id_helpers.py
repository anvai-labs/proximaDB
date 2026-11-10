"""Shared assertions for client-minted record ids."""

import uuid


def is_uuid(value) -> bool:
    """True when ``value`` is a canonical dashed UUID string.

    Default record ids are UUID4s; the old position-derived defaults
    (``record_0``, ``vec_0``) repeated across calls and overwrote.
    """
    try:
        return str(uuid.UUID(str(value))) == str(value)
    except ValueError:
        return False
