"""Tests for the AgentSandbox provider adapter (:class:`AgentSandboxProvider`).

Mocks the :class:`SandboxManager` so the adapter's translation + exec routing +
ephemeral delegation is tested without the SDK. Mirrors ``test_sched_provider.
py``'s approach: the manager is the seam, the adapter maps backend types ->
kernel contract.
"""

from __future__ import annotations

import importlib
import time
from threading import Event
from unittest.mock import MagicMock

import pytest

from bench_core.config import KernelConfig
from env_provider import EphemeralCapable, LifecycleCapable, SandboxInstance, SandboxStatus
from env_provider._base import BackendCreationMetrics, BackendSandboxStatus
from env_provider.agentsandbox import AgentSandboxProvider, build_provider
from env_provider.agentsandbox.config import Config
from env_provider.agentsandbox.schemas import SandboxState


# -------------------------------------------------------------------- fixtures
def _kernel_config():
    cfg = KernelConfig()
    cfg.total_count = 2
    cfg.workflow_type = "coding"
    return cfg


def _state(index: int, *, ready: bool = True, cr_name: str = "agentsbx-1") -> SandboxState:
    cm = BackendCreationMetrics()
    cm.status = BackendSandboxStatus.PORT_READY if ready else BackendSandboxStatus.FAILED
    cm.create_elapsed = 1.0
    cm.total_elapsed = 1.5
    cm.port_wait_elapsed = 0.5
    cm.port_ready_time = 100.0
    cm.submit_time = 99.0
    return SandboxState(sandbox_id=index, sandbox_obj=cr_name, creation_metrics=cm, is_alive=ready)


@pytest.fixture
def provider():
    p = AgentSandboxProvider(_kernel_config(), Config(sandbox_group="sg-1"), Event())
    p._manager = MagicMock()
    return p


def _kernel_cli_module():
    """The kernel module owning ``_build_provider`` (layout differs by repo:
    the GitHub side keeps the dispatch in ``bench_core.bench``, the gitcode
    side in ``bench_core.bench_cli``)."""
    try:
        from bench_core import bench_cli
    except ImportError:
        from bench_core import bench as bench_cli
    return bench_cli


# -------------------------------------------------------------------- identity
class TestIdentity:
    def test_name_and_capabilities(self):
        p = AgentSandboxProvider(_kernel_config(), Config(sandbox_group="sg-1"), Event())
        assert p.name == "agentsandbox"
        assert isinstance(p, EphemeralCapable)
        # Suspended = pod torn down, volumes only (no process/memory state) --
        # not the contract's pause semantics, so no LifecycleCapable.
        assert not isinstance(p, LifecycleCapable)
        assert p.vmm_type is None  # pods: no host VMM process for vm-monitor
        assert p.default_replay_mode == "exec_only"

    def test_manager_lazy(self):
        p = AgentSandboxProvider(_kernel_config(), Config(sandbox_group="sg-1"), Event())
        assert p._manager is None  # built on first access, never in __init__


# ---------------------------------------------------------------- translation
class TestTranslate:
    def test_to_instance_maps_ready(self, provider):
        provider._manager._slot_templates = {1: "python:3.11"}
        inst = provider._to_instance(_state(1, ready=True, cr_name="agentsbx-abc"))
        assert inst.id == "agentsbx-abc"  # the CR name is the stable id
        assert inst.index == 1
        assert inst.ready is True
        assert inst.is_alive is True
        assert inst.numa_node is None  # agentsandbox has no client-side NUMA
        assert inst.template == "python:3.11"
        assert inst.creation_metrics.status == SandboxStatus.READY

    def test_to_instance_maps_failed(self, provider):
        provider._manager._slot_templates = {}
        inst = provider._to_instance(_state(2, ready=False))
        assert inst.ready is False
        assert inst.creation_metrics.status == SandboxStatus.FAILED

    def test_no_handle_yields_empty_id(self, provider):
        provider._manager._slot_templates = {}
        st = _state(1)
        st.sandbox_obj = None
        inst = provider._to_instance(st)
        assert inst.id == ""

    def test_create_all_translates_manager_states(self, provider):
        provider._manager._slot_templates = {}
        provider._manager.create_all.return_value = {1: _state(1)}
        instances = provider.create_all()
        provider._manager.create_all.assert_called_once_with(templates=None)
        assert set(instances) == {1}
        assert instances[1].ready is True

    def test_detect_existing_delegates(self, provider):
        provider._manager._slot_templates = {}
        provider._manager.detect_existing.return_value = {1: _state(1, ready=False)}
        instances = provider.detect_existing()
        assert instances[1].ready is False


# ------------------------------------------------------- lifecycle delegation
class TestLifecycle:
    def test_check_alive_delegates(self, provider):
        st = _state(1)
        provider._manager.sandbox_states = {1: st}
        provider._manager.check_alive.return_value = True
        assert provider.check_alive(SandboxInstance(id="x", index=1)) is True

    def test_check_alive_missing_state(self, provider):
        provider._manager.sandbox_states = {}
        assert provider.check_alive(SandboxInstance(id="x", index=9)) is False

    def test_cleanup_all_no_manager_is_noop(self):
        p = AgentSandboxProvider(_kernel_config(), Config(sandbox_group="sg-1"), Event())
        p.cleanup_all()  # must not raise / must not build the manager
        assert p._manager is None

    def test_list_sandbox_states_none_without_manager(self):
        p = AgentSandboxProvider(_kernel_config(), Config(sandbox_group="sg-1"), Event())
        assert p.list_sandbox_states() is None

    def test_list_sandbox_states_delegates(self, provider):
        provider._manager.list_states.return_value = {"running": 2}
        assert provider.list_sandbox_states() == {"running": 2}

    def test_cleanup_existing_delegates(self, provider):
        provider._manager.cleanup_existing.return_value = 3
        assert provider.cleanup_existing() == 3

    def test_prepare_runs_browser_helpers(self, provider):
        st = _state(1)
        provider._manager.sandbox_states = {1: st}
        provider.prepare(SandboxInstance(id="x", index=1))
        provider._manager.clear_browser_cache.assert_called_once_with(st)
        provider._manager.start_browser_backend.assert_called_once_with(st)


# ---------------------------------------------------------------------- exec
class TestExec:
    def test_exec_routes_through_pod_exec(self, provider):
        st = _state(1)
        provider._manager.sandbox_states = {1: st}
        provider._manager.pod_exec.return_value = (0, "out\n", "err\n")

        result = provider.exec(SandboxInstance(id="agentsbx-1", index=1), "echo out")

        assert (result.exit_code, result.stdout, result.stderr) == (0, "out\n", "err\n")
        provider._manager.pod_exec.assert_called_once_with("agentsbx-1", "echo out", timeout=None, cwd=None, env=None)

    def test_exec_missing_handle_raises(self, provider):
        provider._manager.sandbox_states = {}
        with pytest.raises(RuntimeError, match="No agentsandbox handle"):
            provider.exec(SandboxInstance(id="x", index=1), "true")

    def test_exec_timeout_raises_timeout_error(self, provider):
        st = _state(1)
        provider._manager.sandbox_states = {1: st}

        def _slow(*args, **kwargs):  # noqa: ARG001
            time.sleep(5)

        provider._manager.pod_exec.side_effect = _slow
        with pytest.raises(TimeoutError, match="timed out"):
            provider.exec(SandboxInstance(id="agentsbx-1", index=1), "sleep 60", timeout=1)


# ------------------------------------------------------------------ ephemeral
class TestEphemeral:
    def test_create_one_success(self, provider):
        provider._manager._new_state.return_value = _state(1)
        provider._manager._create_single.return_value = {"success": True, "create_elapsed": 1.0, "template": "t"}
        provider._manager._handle_of.return_value = "agentsbx-1"
        provider._manager._ready_checker.return_value.check.return_value = {
            "success": True,
            "wait_elapsed": 0.5,
            "error": "",
        }
        provider._manager._apply_ready = MagicMock()
        provider._manager._slot_templates = {}

        inst = provider.create_one(1, template="t", metadata={"trajectory_id": "tr-42"})

        assert inst.index == 1
        assert inst.template == "t"
        provider._manager._create_single.assert_called_once()
        _, kwargs = provider._manager._create_single.call_args
        assert kwargs.get("template") == "t"
        assert kwargs.get("metadata") == {"trajectory_id": "tr-42"}
        provider._manager._apply_ready.assert_called_once()

    def test_create_one_failure_raises(self, provider):
        provider._manager._new_state.return_value = _state(1, ready=False)
        provider._manager._create_single.return_value = {"success": False, "create_elapsed": 0.0, "error": "boom"}
        with pytest.raises(RuntimeError, match="create_one\\(1\\) failed: boom"):
            provider.create_one(1)

    def test_kill_one_marks_torn_down(self, provider):
        st = _state(1)
        provider._manager.sandbox_states = {1: st}
        provider._manager._handle_of.return_value = "agentsbx-1"
        inst = SandboxInstance(id="agentsbx-1", index=1, ready=True)
        provider.kill_one(inst)
        provider._manager._kill_one.assert_called_once_with(st)
        assert st.torn_down is True
        assert inst.is_alive is False
        assert st.is_alive is False

    def test_kill_one_failure_leaves_retry_flag(self, provider):
        st = _state(1)
        provider._manager.sandbox_states = {1: st}
        provider._manager._kill_one.side_effect = RuntimeError("api down")
        inst = SandboxInstance(id="agentsbx-1", index=1, ready=True)
        with pytest.raises(RuntimeError):
            provider.kill_one(inst)
        assert st.torn_down is False  # final cleanup_all retries the kill
        assert inst.is_alive is False

    def test_kill_one_unknown_index_is_noop(self, provider):
        provider._manager.sandbox_states = {}
        provider.kill_one(SandboxInstance(id="ghost", index=99))  # must not raise
        provider._manager._kill_one.assert_not_called()


# ------------------------------------------------------------------ build/roster
class TestBuildProvider:
    def test_build_from_raw(self):
        raw = {"agentsandbox": {"sandbox_group": "sg-1", "template": "python:3.11", "namespace": "bench"}}
        p = build_provider(_kernel_config(), raw)
        assert isinstance(p, AgentSandboxProvider)
        assert p._config.sandbox_group == "sg-1"
        assert p._config.template == "python:3.11"
        assert p._config.namespace == "bench"
        assert p._manager is None  # still lazy

    def test_build_from_empty_raw(self):
        p = build_provider(_kernel_config(), {})
        assert isinstance(p, AgentSandboxProvider)
        assert p._config.sandbox_group == ""  # fails fast at create, not build

    def test_roster_entry_matches_module_class(self):
        from bench_core.bench import _PROVIDERS

        module_path, class_name = _PROVIDERS["agentsandbox"]
        assert module_path == "env_provider.agentsandbox"
        module = importlib.import_module(module_path)
        assert getattr(module, class_name) is AgentSandboxProvider

    def test_build_provider_dispatch_wired(self):
        cli = _kernel_cli_module()
        raw = {"agentsandbox": {"sandbox_group": "sg-1"}}
        provider = cli._build_provider("agentsandbox", _kernel_config(), raw)
        assert isinstance(provider, AgentSandboxProvider)
        assert provider._config.sandbox_group == "sg-1"
