"""Compatibility shim for project-local imports.

The canonical implementation remains in experiments/01-minimal-agent. This
shim exposes its public types to modules executed from the project directory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_source = Path(__file__).resolve().parents[2] / "experiments" / "01-minimal-agent" / "agent_loop.py"
_spec = importlib.util.spec_from_file_location("_minimal_agent_loop", _source)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {_source}")
_module = importlib.util.module_from_spec(_spec)
# Dataclasses with postponed annotations resolve their module during execution.
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

__all__ = ["Model", "ToolCall", "ModelResponse", "Tool", "ToolError",
           "MaxStepsExceeded", "ToolRegistry", "AgentResult", "Agent"]
for _name in __all__:
    globals()[_name] = getattr(_module, _name)
