"""Backward-compatible alias for API schemas.

Some routers import from `app.schemas`. The actual models live in `app.api_schemas`.
This shim keeps imports stable without touching router code.
"""

from .api_schemas import *  # noqa: F401,F403
