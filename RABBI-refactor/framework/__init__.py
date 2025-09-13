"""
Lightweight framework layer for RABBI refactor.
This package provides minimal, non-invasive scaffolding around the existing
RABBI-Neighbor codebase: dependency injection, runner orchestration,
standardized results, and visualization adapters.

Note: This package must not rename or change any existing functions/classes
in RABBI-Neighbor. It only coordinates and adapts.
"""

__all__ = [
    "PolicyRegistry",
    "Container",
    "RunResult",
    "MultiKResult",
    "Visualizer",
]

# Lazy imports to avoid importing heavy deps at package import time
try:  # pragma: no cover - optional lazy imports
    from .di import PolicyRegistry, Container  # noqa: F401
    from .results import RunResult, MultiKResult  # noqa: F401
    from .viz import Visualizer  # noqa: F401
except Exception:  # noqa: BLE001 - tolerate missing deps during partial setups
    pass
