"""Seam tests for the real agentsandbox :class:`SandboxManager`.

The provider/adapter tests (``test_provider.py``) mock the manager itself, so
the real manager's SDK wiring -- ``_create_single`` (ownership labels + stale
CR removal + Ready-condition wait), ``_list_existing`` (label-selector
scoping), ``_attach`` (CR-name-is-the-handle), ``pod_exec`` (stream drain +
returncode), ``_kill_one`` (idempotent 404) -- has no coverage there. These
tests inject fakes for the k8s clients (via the ``_build_clients`` seam) and a
fake exec stream (via the module's ``k8s_stream`` global), then drive the real
base lifecycle (create_all / detect_existing / cleanup_all) through them --
entirely SDK-free, so they run wherever the rest of the suite runs.
"""

from __future__ import annotations

import copy
from threading import Event

import pytest

from bench_core.config import KernelConfig
from bench_core.workflow_registry import ensure_workflow_registered
from env_provider.agentsandbox import manager as manager_module
from env_provider.agentsandbox.config import Config
from env_provider.agentsandbox.manager import SandboxManager

ensure_workflow_registered("coding")


# ------------------------------------------------------------------ k8s fakes
class _FakeCustom:
    """Stand-in for CustomObjectsApi over an in-memory Sandbox CR store."""

    def __init__(self, *, auto_ready: bool = True, finished_reason: str | None = None):
        self.created: list[dict] = []  # request bodies (as sent)
        self.deleted: list[str] = []
        self.objects: dict[str, dict] = {}  # CR name -> CR dict (with status)
        self.last_selector: str | None = None
        self.delete_error: Exception | None = None
        self.auto_ready = auto_ready  # become Ready on create (skip the poll wait)
        self.finished_reason = finished_reason  # inject a terminal pod on create

    def create_namespaced_custom_object(self, group, version, namespace, plural, body):
        name = body["metadata"]["name"]
        if name in self.objects:
            raise manager_module.ApiException(status=409)
        obj = copy.deepcopy(body)
        obj["status"] = {"conditions": []}
        if self.auto_ready:
            obj["status"]["conditions"] = [{"type": "Ready", "status": "True", "reason": "DependenciesReady"}]
        elif self.finished_reason:
            obj["status"]["conditions"] = [{"type": "Finished", "status": "True", "reason": self.finished_reason}]
        self.objects[name] = obj
        self.created.append(copy.deepcopy(body))
        return obj

    def get_namespaced_custom_object(self, group, version, namespace, plural, name):
        if name not in self.objects:
            raise manager_module.ApiException(status=404)
        return self.objects[name]

    def list_namespaced_custom_object(self, group, version, namespace, plural, label_selector=None):
        self.last_selector = label_selector
        pairs = dict(part.split("=", 1) for part in (label_selector or "").split(",") if "=" in part)
        items = [
            obj
            for obj in self.objects.values()
            if all((obj["metadata"].get("labels") or {}).get(key) == value for key, value in pairs.items())
        ]
        return {"items": items}

    def delete_namespaced_custom_object(self, group, version, namespace, plural, name):
        if self.delete_error is not None:
            raise self.delete_error
        if name not in self.objects:
            raise manager_module.ApiException(status=404)
        self.deleted.append(name)
        del self.objects[name]


# --------------------------------------------------------------------- helpers
def _kc(total_count: int = 2, workflow: str = "coding") -> KernelConfig:
    return KernelConfig(total_count=total_count, workflow_type=workflow)


def _config(**kw) -> Config:
    kw.setdefault("sandbox_group", "sg-1")
    kw.setdefault("ready_timeout", 5)
    return Config(**kw)


class _FakeCore:
    """Stand-in for CoreV1Api: pod_exec only reads the exec method off it."""

    def connect_get_namespaced_pod_exec(self, *args, **kwargs):  # noqa: ARG002
        return None  # the fake stream fn swallows it


def _manager(
    monkeypatch, custom: _FakeCustom | None = None, config: Config | None = None
) -> tuple[SandboxManager, _FakeCustom]:
    custom = custom if custom is not None else _FakeCustom()
    monkeypatch.setattr(SandboxManager, "_build_clients", lambda self: (_FakeCore(), custom))
    return SandboxManager(_kc(), config if config is not None else _config(), Event()), custom


def _seed_cr(
    custom: _FakeCustom, name: str, *, group: str = "sg-1", ready: bool = True, suspended: bool = False
) -> dict:
    """Seed a stored CR (ownership labels + conditions) as the controller would."""
    conditions = [{"type": "Ready", "status": "True" if ready else "False"}]
    spec: dict = {}
    if suspended:
        spec["operatingMode"] = "Suspended"
    obj = {
        "metadata": {"name": name, "labels": {"managed_by": "bench-core", "sandbox_group": group}},
        "spec": spec,
        "status": {"conditions": conditions},
    }
    custom.objects[name] = obj
    return obj


class _FakeStreamClient:
    """Stand-in for the kubernetes WSClient (one frame, then server close).

    Mirrors the real client's contract: ``peek_*`` returns the buffer without
    consuming it, ``read_*`` drains it, ``returncode`` carries the exit status.
    ``hangs=True`` models a wedged stream (never closes, no output);
    ``drop_status=True`` models a channel that closed without a status frame
    (the real client raises a bare TypeError reading ``returncode`` then).
    """

    def __init__(self, *, stdout="", stderr="", returncode=0, hangs=False, drop_status=False):
        self._stdout_buf = stdout
        self._stderr_buf = stderr
        self._returncode = returncode
        self._hangs = hangs
        self._drop_status = drop_status
        self._updated = False
        self.close_called = False

    def is_open(self):
        return self._hangs or not self._updated

    def update(self, timeout=None):  # noqa: ARG002
        if not self._hangs:
            self._updated = True

    def peek_stdout(self):
        return self._stdout_buf

    def peek_stderr(self):
        return self._stderr_buf

    def read_stdout(self):
        out, self._stdout_buf = self._stdout_buf, ""
        return out

    def read_stderr(self):
        err, self._stderr_buf = self._stderr_buf, ""
        return err

    @property
    def returncode(self):
        if self._drop_status:
            raise TypeError("'NoneType' object is not subscriptable")  # real client's parse failure
        return self._returncode

    def close(self):
        self.close_called = True


def _fake_stream(monkeypatch, client: _FakeStreamClient) -> dict:
    """Swap the module's ``k8s_stream`` for a capturing fake; return captures.

    Every exec gets a FRESH copy of ``client``: a real WSClient's buffers drain
    once and the channel closes after the first read loop, so a second probe
    on a shared instance would see an already-closed stream with empty output.
    """
    captured: dict = {}

    def _stream(fn, *args, **kwargs):
        captured["fn"] = fn
        captured["args"] = args
        captured["kwargs"] = kwargs
        fresh = copy.deepcopy(client)
        captured["last_client"] = fresh
        return fresh

    monkeypatch.setattr(manager_module, "k8s_stream", _stream)
    return captured


def _fake_probe_stream(monkeypatch) -> None:
    """A plain success stream for workflow readiness probes.

    The coding probe is ``uname -a`` and requires a NON-empty stdout, so the
    fake returns one line of it with exit 0.
    """
    monkeypatch.setattr(
        manager_module, "k8s_stream", lambda fn, *args, **kwargs: _FakeStreamClient(stdout="Linux", returncode=0)
    )


# ------------------------------------------------------------- _create_single
class TestCreateSingle:
    def test_success_builds_cr_with_ownership_labels(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        state = mgr._new_state(1)
        result = mgr._create_single(state)

        assert result["success"] is True
        assert result["template"] == "python:3.11"
        assert state.creation_metrics.status.value == "created"
        assert state.sandbox_obj == "agentsbx-1"

        body = custom.created[0]
        assert body["apiVersion"] == "agents.x-k8s.io/v1beta1"
        assert body["kind"] == "Sandbox"
        assert body["metadata"]["name"] == "agentsbx-1"
        assert body["metadata"]["labels"] == {"managed_by": "bench-core", "sandbox_group": "sg-1"}
        container = body["spec"]["podTemplate"]["spec"]["containers"][0]
        assert container["image"] == "python:3.11"
        # SandboxSpec embeds its lifecycle struct inline -> TOP-LEVEL spec props
        assert body["spec"]["shutdownPolicy"] == "Delete"
        assert "shutdownTime" in body["spec"]
        assert body["spec"]["podTemplate"]["spec"]["automountServiceAccountToken"] is False

    def test_resources_and_pod_knobs_forwarded(self, monkeypatch):
        config = _config(
            template="ubuntu-openclaw:24.04",
            resources={"cpu": "2", "memory": "4Gi"},
            runtime_class_name="gvisor",
            image_pull_policy="IfNotPresent",
            pod_container="main",
        )
        mgr, custom = _manager(monkeypatch, config=config)
        result = mgr._create_single(mgr._new_state(1))

        assert result["success"] is True
        container = custom.created[0]["spec"]["podTemplate"]["spec"]["containers"][0]
        assert container["image"] == "ubuntu-openclaw:24.04"
        assert container["name"] == "main"
        assert container["imagePullPolicy"] == "IfNotPresent"
        assert container["resources"] == {
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "2", "memory": "4Gi"},
        }
        assert custom.created[0]["spec"]["podTemplate"]["spec"]["runtimeClassName"] == "gvisor"

    def test_stale_same_name_replaced(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        mgr._create_single(mgr._new_state(1))  # first run leaves agentsbx-1 behind
        assert "agentsbx-1" in custom.objects

        result = mgr._create_single(mgr._new_state(1))  # second run recreates

        assert result["success"] is True
        assert custom.deleted == ["agentsbx-1"]
        assert len(custom.created) == 2

    def test_stale_vanished_between_get_and_delete(self, monkeypatch):
        # The GET passes (owned CR), the DELETE removes it but reports 404
        # (the CR vanished server-side in between -- controller expiry,
        # concurrent cleanup): the create must proceed, not fail.
        custom = _FakeCustom()
        mgr, _mgr2 = _manager(monkeypatch, custom=custom)
        _seed_cr(custom, "agentsbx-1", ready=True)
        original_delete = custom.delete_namespaced_custom_object

        def _delete_then_404(group, version, namespace, plural, name):  # noqa: ARG001
            original_delete(group=group, version=version, namespace=namespace, plural=plural, name=name)
            raise manager_module.ApiException(status=404)

        custom.delete_namespaced_custom_object = _delete_then_404

        result = mgr._create_single(mgr._new_state(1))

        assert result["success"] is True
        assert len(custom.created) == 1

    def test_stale_foreign_owner_refused(self, monkeypatch):
        # A name collision with another team's CR (same prefix, shared
        # namespace) is refused -- _delete_stale must never delete it.
        mgr, custom = _manager(monkeypatch)
        obj = _seed_cr(custom, "agentsbx-1", group="sg-other")
        assert obj["metadata"]["labels"]["managed_by"] == "bench-core"  # right manager, wrong group

        result = mgr._create_single(mgr._new_state(1))

        assert result["success"] is False
        assert "foreign ownership labels" in result["error"]
        assert custom.created == [] and custom.deleted == []

    def test_missing_sandbox_group_fails_fast(self, monkeypatch):
        mgr, custom = _manager(monkeypatch, config=_config(sandbox_group=""))
        state = mgr._new_state(1)
        result = mgr._create_single(state)

        assert result["success"] is False
        assert "sandbox_group" in result["error"]
        assert custom.created == []  # nothing sent to the cluster
        # NOTE: status stays CREATING on this direct seam call -- the base's
        # batch loop is what maps a failed result onto BackendSandboxStatus.
        # FAILED (mirrors sandboxsched/docker); create_one does the same.

    def test_terminal_pod_fails_fast(self, monkeypatch):
        custom = _FakeCustom(auto_ready=False, finished_reason="PodFailed")
        mgr, _custom = _manager(monkeypatch, custom=custom, config=_config(ready_timeout=30))
        result = mgr._create_single(mgr._new_state(1))

        assert result["success"] is False
        assert "finished: PodFailed" in result["error"]

    def test_ready_timeout_returns_failure(self, monkeypatch):
        custom = _FakeCustom(auto_ready=False)
        mgr, _custom = _manager(monkeypatch, custom=custom, config=_config(ready_timeout=0))
        result = mgr._create_single(mgr._new_state(1))

        assert result["success"] is False
        assert "not Ready within 0s" in result["error"]

    def test_metadata_lands_as_pod_annotations_not_labels(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        result = mgr._create_single(mgr._new_state(1), metadata={"trajectory_id": "tr-42", "runner": "r1"})

        assert result["success"] is True
        body = custom.created[0]
        assert body["spec"]["podTemplate"]["metadata"]["annotations"] == {"trajectory_id": "tr-42", "runner": "r1"}
        # ownership labels stay exact -- caller metadata must not override them
        assert body["metadata"]["labels"] == {"managed_by": "bench-core", "sandbox_group": "sg-1"}

    def test_template_override_used_as_image(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        result = mgr._create_single(mgr._new_state(1), template="custom-img:2")

        assert result["success"] is True
        assert result["template"] == "custom-img:2"
        assert custom.created[0]["spec"]["podTemplate"]["spec"]["containers"][0]["image"] == "custom-img:2"


# --------------------------------------------------------------- list / detect
class TestListAndDetect:
    def test_list_existing_scopes_to_ownership_selector(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        _seed_cr(custom, "agentsbx-1")  # owned
        _seed_cr(custom, "agentsbx-2", group="sg-other")  # another team's fleet
        _seed_cr(custom, "unrelated")  # right group, but not bench-core's tag
        custom.objects["unrelated"]["metadata"]["labels"]["managed_by"] = "someone-else"

        listed = mgr._list_existing()

        assert [mgr._external_id(item) for item in listed] == ["agentsbx-1"]
        assert custom.last_selector == "managed_by=bench-core,sandbox_group=sg-1"

    def test_external_id_and_attach(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        obj = _seed_cr(custom, "agentsbx-3")
        assert mgr._external_id(obj) == "agentsbx-3"
        assert mgr._attach(obj) == "agentsbx-3"  # the CR name IS the handle

    def test_detect_existing_readies_via_coding_probe(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        _seed_cr(custom, "agentsbx-1", ready=True)
        _seed_cr(custom, "agentsbx-2", ready=True)
        captured = _fake_stream(monkeypatch, _FakeStreamClient(stdout="Linux", returncode=0))

        states = mgr.detect_existing()

        assert set(states) == {1, 2}
        assert all(state.creation_metrics.status.value == "port_ready" for state in states.values())
        # captured holds the LAST probe: a detected CR name exec'd in the config namespace
        assert captured["args"][0] in {"agentsbx-1", "agentsbx-2"}
        assert captured["args"][1] == "default"


# -------------------------------------------------------------------- killing
class TestKill:
    def test_kill_one_deletes_cr(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        _seed_cr(custom, "agentsbx-1")
        state = mgr._new_state(1, external_id="agentsbx-1")
        mgr._kill_one(state)
        assert custom.deleted == ["agentsbx-1"]

    def test_kill_one_404_is_idempotent(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch)
        state = mgr._new_state(1, external_id="agentsbx-gone")
        mgr._kill_one(state)  # must not raise

    def test_kill_one_other_error_raises(self, monkeypatch):
        custom = _FakeCustom()
        custom.delete_error = manager_module.ApiException(status=500)
        mgr, _custom = _manager(monkeypatch, custom=custom)
        state = mgr._new_state(1, external_id="agentsbx-1")
        with pytest.raises(manager_module.ApiException):
            mgr._kill_one(state)

    def test_cleanup_all_sets_killed_status(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        _seed_cr(custom, "agentsbx-1", ready=True)
        _seed_cr(custom, "agentsbx-2", ready=True)
        _fake_probe_stream(monkeypatch)

        mgr.detect_existing()
        mgr.cleanup_all()

        assert sorted(custom.deleted) == ["agentsbx-1", "agentsbx-2"]
        assert all(state.creation_metrics.status.value == "killed" for state in mgr._states.values())


# ------------------------------------------------------------------- liveness
class TestCheckAlive:
    def test_ready_cr_is_alive(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        _seed_cr(custom, "agentsbx-1", ready=True)
        assert mgr.check_alive(mgr._new_state(1, external_id="agentsbx-1")) is True

    def test_unready_cr_is_not_alive(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        _seed_cr(custom, "agentsbx-1", ready=False)
        assert mgr.check_alive(mgr._new_state(1, external_id="agentsbx-1")) is False

    def test_missing_cr_is_not_alive(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch)
        assert mgr.check_alive(mgr._new_state(1, external_id="agentsbx-gone")) is False

    def test_dead_state_short_circuits(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        _seed_cr(custom, "agentsbx-1", ready=True)
        state = mgr._new_state(1, external_id="agentsbx-1")
        state.is_alive = False
        assert mgr.check_alive(state) is False


# --------------------------------------------------------------- fleet states
class TestStateOf:
    def test_ready_is_running(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        obj = _seed_cr(custom, "agentsbx-1", ready=True)
        assert mgr._state_of(obj) == "running"

    def test_suspended_is_paused(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        obj = _seed_cr(custom, "agentsbx-1", ready=False, suspended=True)
        assert mgr._state_of(obj) == "paused"

    def test_provisioning_is_other(self, monkeypatch):
        mgr, custom = _manager(monkeypatch)
        obj = _seed_cr(custom, "agentsbx-1", ready=False)
        assert mgr._state_of(obj) == "other"


# ------------------------------------------------------------------- pod_exec
class TestPodExec:
    def test_output_and_exit_code(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch)
        client = _FakeStreamClient(stdout="hello\n", stderr="warn\n", returncode=3)
        captured = _fake_stream(monkeypatch, client)

        exit_code, stdout, stderr = mgr.pod_exec("agentsbx-1", "echo hello")

        assert (exit_code, stdout, stderr) == (3, "hello\n", "warn\n")
        assert captured["last_client"].close_called is True

    def test_passes_pod_name_and_namespace(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch)
        captured = _fake_stream(monkeypatch, _FakeStreamClient())
        mgr.pod_exec("agentsbx-7", "true")
        assert captured["args"][:2] == ("agentsbx-7", "default")

    def test_cwd_env_wrapped(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch)
        captured = _fake_stream(monkeypatch, _FakeStreamClient())
        mgr.pod_exec("agentsbx-1", "echo hi", cwd="/tmp/w", env={"A": "1"})
        # env prefix + cd wrap, both shell-quoted (mirrors the docker adapter)
        assert captured["kwargs"]["command"] == ["/bin/sh", "-c", "cd /tmp/w && A=1 echo hi"]

    def test_bare_command_unwrapped(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch)
        captured = _fake_stream(monkeypatch, _FakeStreamClient())
        mgr.pod_exec("agentsbx-1", "uname -a")
        assert captured["kwargs"]["command"] == ["/bin/sh", "-c", "uname -a"]
        assert captured["kwargs"]["stdin"] is False

    def test_timeout_forwarded_as_request_timeout(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch)
        captured = _fake_stream(monkeypatch, _FakeStreamClient())
        mgr.pod_exec("agentsbx-1", "sleep 60", timeout=10)
        assert captured["kwargs"]["_request_timeout"] == 10

    def test_pod_container_configured(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch, config=_config(pod_container="main"))
        captured = _fake_stream(monkeypatch, _FakeStreamClient())
        mgr.pod_exec("agentsbx-1", "true")
        assert captured["kwargs"]["container"] == "main"

    def test_pod_exec_timeout_bounds_hanging_stream(self, monkeypatch):
        # _request_timeout alone only bounds the websocket connect on the real
        # client; the read-loop deadline is what bounds a wedged-but-open
        # stream (the probe/browser-helper callers have no outer watchdog).
        mgr, _custom = _manager(monkeypatch)
        _fake_stream(monkeypatch, _FakeStreamClient(hangs=True))
        with pytest.raises(TimeoutError, match="did not finish within 1s"):
            mgr.pod_exec("agentsbx-1", "sleep infinity", timeout=1)

    def test_pod_exec_dropped_status_frame_raises_diagnosable(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch)
        _fake_stream(monkeypatch, _FakeStreamClient(returncode=0, drop_status=True))
        with pytest.raises(RuntimeError, match="without an exit status frame"):
            mgr.pod_exec("agentsbx-1", "true")


# ------------------------------------------------------------ browser helpers
class TestBrowserHelpers:
    def test_start_browser_backend_success(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch)
        _fake_stream(monkeypatch, _FakeStreamClient(returncode=0))
        state = mgr._new_state(1)
        ok, err = mgr.start_browser_backend(state)
        assert ok is True
        assert err == ""
        assert state.browser_started is True

    def test_start_browser_backend_failure(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch)
        _fake_stream(monkeypatch, _FakeStreamClient(stdout="boom", returncode=1))
        state = mgr._new_state(1)
        ok, err = mgr.start_browser_backend(state)
        assert ok is False
        assert "exit_code=1" in err
        assert state.browser_started is False

    def test_clear_browser_cache(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch)
        _fake_stream(monkeypatch, _FakeStreamClient(returncode=0))
        assert mgr.clear_browser_cache(mgr._new_state(1)) is True


# ------------------------------------------------------------ SDK-free guard
class TestBuildClients:
    def test_without_sdk_raises_clear_error(self, monkeypatch):
        # No _build_clients patch here -- the real seam must hit the SDK guard.
        monkeypatch.setattr(manager_module, "k8s_client", None)
        with pytest.raises(RuntimeError, match="agentsandbox"):
            SandboxManager(_kc(), _config(), Event())

    def test_header_extras(self, monkeypatch):
        mgr, _custom = _manager(monkeypatch, config=_config(template="img:1", resources={"cpu": "1", "memory": "1Gi"}))
        extras = mgr._create_header_extras()
        assert any("Image: img:1" in line for line in extras)
        assert any("Namespace: default" in line for line in extras)
