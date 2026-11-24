"""AgentSandbox backend state (SDK handle + backend lifecycle).

The kernel works on host-agnostic :class:`bench_core.schemas.BenchSandbox` (built
fresh via ``BenchSandbox.from_instance`` from the lean contract
:class:`env_provider.SandboxInstance`); the manager + adapter keep the Sandbox
CR handle here and translate out of it. Workflow task metrics live in
:mod:`bench_core.schemas` (the kernel) -- not duplicated here.

The backend status enum + creation metrics are shared with e2b/docker via
:mod:`env_provider._base`; this module re-exports them under the agentsandbox
names so the manager/adapter imports read the same as the other backends. Only
:class:`SandboxState` (the handle + lifecycle flags) is defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from env_provider._base import (
    BackendCreationMetrics as CreationMetrics,
)
from env_provider._base import (
    BackendSandboxStatus as SandboxStatus,
)

__all__ = ["SandboxState", "SandboxStatus", "CreationMetrics"]


@dataclass
class SandboxState:
    """Per-sandbox AgentSandbox backend state.

    Carries the handle (``sandbox_obj`` -- the Sandbox CR's name, which is also
    the backing pod's name; the manager speaks to the cluster through
    namespace-scoped API clients, so a name is the full handle) + backend
    lifecycle flags + creation timing. Workflow task metrics and per-step
    timing live on the kernel's ``BenchSandbox``, not here: the adapter
    translates a ``SandboxState`` into a handle-free
    :class:`env_provider.SandboxInstance`, then the kernel rebuilds its own
    metrics state via ``BenchSandbox.from_instance``. The kernel never sees an
    agentsandbox ``SandboxState``.
    """

    sandbox_id: int  # Sequence number (1, 2, 3...) -- kernel index, NOT the CR name
    sandbox_obj: object | None = None  # Handle: the Sandbox CR name (str)
    batch_id: int = -1  # Batch ID

    creation_metrics: CreationMetrics = field(default_factory=CreationMetrics)

    is_alive: bool = True  # Sandbox alive status
    stopped_by_cleanup: bool = False  # Removed by normal benchmark cleanup (set by the base)
    torn_down: bool = False  # Server-side delete succeeded (cleanup skips; set by the base)
    browser_started: bool = False  # OpenClaw browser backend started flag (read by the adapter as warmup_done)
