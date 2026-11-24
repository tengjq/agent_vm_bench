"""AgentSandbox :class:`EnvironmentProvider` adapter.

Wraps :class:`env_provider.agentsandbox.manager.SandboxManager` behind the
kernel's :class:`env_provider.EnvironmentProvider` contract. The manager owns
the Sandbox CR handles (``SandboxState.sandbox_obj`` -- the CR name, which is
also the backing pod's name); the adapter translates those into host-agnostic
:class:`SandboxInstance` objects and routes ``exec`` calls back through the
manager's handle table -- the kernel never sees a kubernetes type.

This is the only agentsandbox-specific code the kernel ever loads. It is
lazy-imported by ``bench_core.bench._build_provider`` (and
``bench_core.bench_cli`` on the gitcode side), so ``bench_core`` itself never
depends on the kubernetes SDK -- the layering rule (kernel must not import
provider packages) holds.

Backend: kubernetes-sigs/agent-sandbox. A sandbox is one ``Sandbox`` CR
(``agents.x-k8s.io/v1beta1``); the controller cold-creates the backing pod and
reports readiness through the CR's ``Ready`` condition. Commands exec through
the API server's pod-exec stream -- no warm pools, no sandbox router, no
per-sandbox port-forwards.

Capability set: exec-only plus :class:`env_provider.EphemeralCapable`
(``create_one``/``kill_one`` for the exec_ephemeral replay mode -- the
oversub sweep's default). The CRD has no memory-snapshot pause/resume: its
``operatingMode: Suspended`` tears the pod down and keeps only volumes, so a
resumed sandbox loses its processes (and any browser backend) -- that is not
the contract's "pause (memory-snapshot)" semantics, so this provider does NOT
implement :class:`LifecycleCapable` and stays on ``exec_only`` /
``exec_ephemeral``. It also does not implement ``SnapshotSizeCapable`` (no
snapshots). ``vmm_type`` stays None: sandboxes are plain pods, there is no
host VMM process for vm-monitor to sample.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import Any

from bench_core.config import KernelConfig
from env_provider import (
    CommandResult,
    CreationMetrics,
    EnvironmentProvider,
    SandboxInstance,
    SandboxStatus,
)
from env_provider._base import BackendSandboxStatus

from .config import Config
from .manager import SandboxManager
from .schemas import SandboxState

logger = logging.getLogger(__name__)

# BackendSandboxStatus.value -> kernel SandboxStatus. The manager sets a
# BackendSandboxStatus on each state's creation_metrics (PENDING/CREATING/
# CREATED/PORT_READY/ACTIVE/FAILED/PORT_FAILED/OFFLINE/KILLED -- the shared
# backend-granular enum from env_provider._base, re-exported via schemas).
# Keyed by the enum's value string so the lookup stays correct when the enum
# class is re-bound across the provider/state module boundary (mirrors e2b).
# Unknown values map to FAILED defensively.
_STATUS_MAP: dict[str, SandboxStatus] = {
    BackendSandboxStatus.PENDING.value: SandboxStatus.PENDING,
    BackendSandboxStatus.CREATING.value: SandboxStatus.CREATING,
    BackendSandboxStatus.CREATED.value: SandboxStatus.CREATED,
    BackendSandboxStatus.PORT_READY.value: SandboxStatus.READY,
    BackendSandboxStatus.ACTIVE.value: SandboxStatus.ACTIVE,
    BackendSandboxStatus.FAILED.value: SandboxStatus.FAILED,
    BackendSandboxStatus.PORT_FAILED.value: SandboxStatus.READY_FAILED,
    BackendSandboxStatus.OFFLINE.value: SandboxStatus.OFFLINE,
    BackendSandboxStatus.KILLED.value: SandboxStatus.KILLED,
}


class AgentSandboxProvider(EnvironmentProvider):
    """EnvironmentProvider backed by an AgentSandbox :class:`SandboxManager`.

    The adapter holds the kernel's :class:`KernelConfig` (shared stress params)
    plus the agentsandbox :class:`Config` (cluster access, image, ownership
    group) plus the stop event. The :class:`SandboxManager` (and its k8s
    clients) is constructed lazily on first use -- so the kernel can run
    host-side preflight and the header print before any cluster call. The
    kernel never sees the manager or SDK types directly.
    """

    name = "agentsandbox"

    def __init__(self, kernel_config: KernelConfig, config: Config, stop_event: threading.Event) -> None:
        self._kernel_config = kernel_config
        self._config = config
        self._stop_event = stop_event
        # Guards the lazy build below (first access can race under a
        # concurrent create wave -- mirror sandboxsched).
        self._manager_lock = threading.Lock()
        self._manager: SandboxManager | None = None

    @property
    def manager(self) -> SandboxManager:
        """The wrapped SandboxManager, constructed on first access.

        Lazy so the kernel's prepare_env / header-print can run before any
        cluster client is built. Tests inject a mock by setting ``_manager``
        directly.
        """
        with self._manager_lock:
            if self._manager is None:
                self._manager = SandboxManager(self._kernel_config, self._config, self._stop_event)
        return self._manager

    # ------------------------------------------------------------------ lifecycle
    def create_all(self, *, templates: dict[int, str | None] | None = None) -> Mapping[int, SandboxInstance]:
        return self._translate(self.manager.create_all(templates=templates))

    def detect_existing(self) -> Mapping[int, SandboxInstance]:
        return self._translate(self.manager.detect_existing())

    def list_sandbox_states(self) -> Mapping[str, int] | None:
        # No manager = nothing was ever created (preflight/cleanup-only runs):
        # building one here would load a kubeconfig just to poll a monitor
        # series.
        if self._manager is None:
            return None
        return self.manager.list_states()

    def check_alive(self, inst: SandboxInstance) -> bool:
        state = self.manager.sandbox_states.get(inst.index)
        if state is None:
            return False
        return self.manager.check_alive(state)

    def cleanup_all(self) -> None:
        # If the manager was never built (e.g. preflight failed before create),
        # there is nothing to tear down.
        if self._manager is None:
            return
        self._manager.cleanup_all()

    def cleanup_existing(self) -> int:
        # Delegate to the manager's list->attach->delete path, which skips the
        # readiness probe (a not-Ready CR must not stall teardown on the
        # 300s probe wait).
        return self.manager.cleanup_existing()

    # ------------------------------------------------------------------ setup hooks
    def prepare_env(self) -> None:
        # Cluster access (kubeconfig) is resolved by the manager at first
        # build; no host-side environment setup is needed here.
        return None

    def prepare(self, inst: SandboxInstance) -> None:
        """Start the browser backend + clear cache before warmup."""
        state = self.manager.sandbox_states.get(inst.index)
        if state is None:
            return
        self.manager.clear_browser_cache(state)
        self.manager.start_browser_backend(state)

    # ------------------------------------------------------------------ command exec
    def exec(
        self,
        inst: SandboxInstance,
        command: str,
        *,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        state = self.manager.sandbox_states.get(inst.index)
        if state is None or state.sandbox_obj is None:
            raise RuntimeError(f"No agentsandbox handle for sandbox index {inst.index}")
        if timeout is None:
            return self._exec(state, command, cwd=cwd, env=env)
        # The k8s exec stream is a blocking call; run it in a worker thread and
        # raise on timeout so the kernel's runners can classify the failure
        # (they catch Exception and match "timed out") -- same watchdog as the
        # docker adapter.
        return self._exec_with_timeout(state, command, timeout, cwd=cwd, env=env)

    # ------------------------------------------------------------------ ephemeral (EphemeralCapable)
    def create_one(
        self, index: int, *, template: str | None = None, metadata: dict[str, str] | None = None
    ) -> SandboxInstance:
        """Create a single sandbox on demand (trajectory mode, EphemeralCapable).

        Mirrors the single-sandbox path of the base's create batch: build a
        state, run ``_create_single`` (forwarding ``metadata`` + ``template``),
        populate ``_slot_templates`` so ``_to_instance`` stamps the resolved
        template onto ``SandboxInstance.template``, run the base readiness
        probe, and map the result via ``_apply_ready`` -- so timing is
        consistent with ``create_all``. ``metadata`` lands as pod annotations
        (operator visibility only -- see the manager's ``_apply_pod_metadata``).
        """
        mgr = self.manager
        state = mgr._new_state(index)
        mgr._states[index] = state
        result = mgr._create_single(state, metadata=metadata, template=template)
        label = f"{mgr._noun}{index}"
        if result["success"]:
            mgr._slot_templates[index] = result.get("template")
            ready = mgr._ready_checker().check(
                mgr._handle_of(state),
                self._kernel_config.workflow_type,
                label,
            )
            mgr._apply_ready(state, ready, create_elapsed=result["create_elapsed"])
        else:
            state.creation_metrics.status = BackendSandboxStatus.FAILED
            state.creation_metrics.error_msg = result["error"]
            state.is_alive = False
            raise RuntimeError(f"create_one({index}) failed: {result['error']}")
        return self._to_instance(state)

    def kill_one(self, inst: SandboxInstance) -> None:
        """Tear down a single sandbox (trajectory mode, EphemeralCapable).

        On a successful server-side delete the state is marked ``torn_down``
        so the final ``cleanup_all`` skips it (re-killing an already-dead
        sandbox only yields misleading warnings). A failed delete leaves the
        flag False so cleanup retries it -- no leak. The handle is kept either
        way for ``check_alive`` / post-mortem diagnostics.
        """
        state = self.manager.sandbox_states.get(inst.index)
        if state is None:
            return  # never created (runner finally-path safety)
        try:
            if self.manager._handle_of(state) is not None:
                self.manager._kill_one(state)
                state.torn_down = True
        finally:
            inst.is_alive = False
            state.is_alive = False

    # ------------------------------------------------------------------ translation
    def _translate(self, states: Mapping[int, SandboxState]) -> dict[int, SandboxInstance]:
        """Translate ``{index: SandboxState}`` -> ``{index: SandboxInstance}``."""
        return {index: self._to_instance(state) for index, state in states.items()}

    def _to_instance(self, state: SandboxState) -> SandboxInstance:
        cm = state.creation_metrics
        cr_name = str(state.sandbox_obj) if state.sandbox_obj is not None else ""
        status = _STATUS_MAP.get(cm.status.value, SandboxStatus.FAILED)
        return SandboxInstance(
            id=cr_name,
            index=state.sandbox_id,
            numa_node=None,  # agentsandbox has no client-side NUMA binding
            ready=(status == SandboxStatus.READY),
            is_alive=state.is_alive,
            warmup_done=state.browser_started,
            creation_metrics=CreationMetrics(
                submit_time=cm.submit_time,
                ready_time=cm.port_ready_time,
                create_elapsed=cm.create_elapsed,
                ready_check_elapsed=cm.port_wait_elapsed,
                total_elapsed=cm.total_elapsed,
                status=status,
                error=cm.error_msg,
                ready_check_error=cm.port_check_error,
            ),
            template=self.manager._slot_templates.get(state.sandbox_id),
        )

    # ------------------------------------------------------------------ internals
    def _exec(
        self, state: SandboxState, command: str, *, cwd: str | None = None, env: dict[str, str] | None = None
    ) -> CommandResult:
        exit_code, stdout, stderr = self.manager.pod_exec(
            str(state.sandbox_obj), command, timeout=None, cwd=cwd, env=env
        )
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    def _exec_with_timeout(
        self,
        state: SandboxState,
        command: str,
        timeout: int,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        box: dict[str, Any] = {}

        def _worker() -> None:
            try:
                box["result"] = self._exec(state, command, cwd=cwd, env=env)
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
                box["error"] = exc

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            # The pod-side process may keep running (the exec stream is not
            # interruptible from the host); the benchmark's consecutive-failure
            # handling marks such a sandbox offline anyway.
            raise TimeoutError(f"agentsandbox exec timed out after {timeout}s: {command[:80]}")
        if "error" in box:
            raise box["error"]
        return box["result"]


def from_config(kernel_config: KernelConfig, config: Config, stop_event: threading.Event) -> AgentSandboxProvider:
    """Build an :class:`AgentSandboxProvider` from a KernelConfig + Config.

    The SandboxManager is constructed lazily on first use, so this is cheap
    and does not talk to the cluster yet.
    """
    return AgentSandboxProvider(kernel_config, config, stop_event)


def build_provider(config: KernelConfig, raw_config: dict) -> AgentSandboxProvider:
    """Construct an :class:`AgentSandboxProvider` from a raw YAML dict.

    ``config`` is the already-built :class:`KernelConfig` (shared stress params
    from :meth:`KernelConfig.from_raw`); the agentsandbox backend Config is
    rebuilt here from the ``agentsandbox:`` block of the same raw dict. Both
    are passed to the provider: the kernel drives ``config``, the manager reads
    backend knobs from the agentsandbox Config.
    """
    stop_event = threading.Event()
    agentsandbox_config = Config.from_raw(raw_config) if raw_config else Config()
    return from_config(config, agentsandbox_config, stop_event)
