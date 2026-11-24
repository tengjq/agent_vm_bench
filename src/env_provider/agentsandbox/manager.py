"""AgentSandbox sandbox manager (SDK seams over :class:`BaseSandboxManager`).

The workflow-agnostic create/detect/cleanup skeleton is inherited from
:class:`env_provider._base.BaseSandboxManager`; this module supplies only the
kubernetes SDK seams (Sandbox CR create/list/get/delete + the pod-exec probe)
and the agentsandbox-specific methods the base can't own: ``check_alive``
(CR Ready condition), ``pod_exec`` (API-server exec stream), and the browser
backend helpers (hot-start openclaw, clear user-data). Readiness probing is
delegated to :class:`env_provider._ready.ReadyChecker` via the base, exactly
like the docker backend.

Sandbox model: a sandbox is one ``Sandbox`` CR of the kubernetes-sigs/
agent-sandbox API (``agents.x-k8s.io/v1beta1``). The controller cold-creates
the backing pod from ``spec.podTemplate``; the CR's ``Ready`` condition becomes
True once that pod is Running + Ready with an IP. ``_create_single`` therefore
waits for the CR condition itself (the pod must exist before any workflow
probe can exec into it), and the base's ReadyChecker probes the workflow on
top. The CR name doubles as the exec handle -- the backing pod is named after
the Sandbox -- so ``state.sandbox_obj`` holds the name string and the manager
speaks to the cluster through namespace-scoped API clients.

The ``kubernetes`` package is an optional extra: the import is guarded so the
package (and its tests) load SDK-free, and all exception handling rides the
module-level :data:`ApiException` (the real SDK's exception when installed, a
stand-in otherwise).
"""

from __future__ import annotations

import logging
import shlex
import time
from datetime import datetime, timedelta, timezone
from threading import Event
from typing import Any

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
    from kubernetes.stream import stream as k8s_stream

    ApiException = k8s_client.ApiException
except ImportError:  # pragma: no cover - SDK is an optional extra
    # The kubernetes SDK is only needed to actually talk to a cluster. Keeping
    # this module importable without it lets the provider package and its tests
    # load on SDK-free environments: tests inject fakes through
    # ``_build_clients`` and may raise the stand-in exception below.
    k8s_client = None  # type: ignore[assignment]
    k8s_config = None  # type: ignore[assignment]
    k8s_stream = None  # type: ignore[assignment]

    class ApiException(Exception):  # type: ignore[no-redef]
        """SDK-free stand-in for ``kubernetes.client.ApiException``."""

        status = 0

        def __init__(self, status: int = 0) -> None:
            self.status = status
            super().__init__(f"status={status}")


from env_provider._base import BackendSandboxStatus, BaseSandboxManager

from .config import Config
from .schemas import SandboxState

logger = logging.getLogger(__name__)

# The kubernetes-sigs/agent-sandbox Sandbox API (api/v1beta1 of that repo).
SANDBOX_API_GROUP = "agents.x-k8s.io"
SANDBOX_API_VERSION = "v1beta1"
SANDBOX_PLURAL = "sandboxes"

# Ownership labels stamped on every sandbox this manager creates, and the
# label selector ``--cleanup`` / ``--detect`` / the fleet gauge scope their
# listing by. The namespace may hold other teams' sandboxes, so without the
# pair a listing would surface -- and ``--cleanup`` would delete -- fleets this
# config does not own. The tag is config-derived (sandbox_group), so a fleet
# stays visible to ``--cleanup`` / ``--detect`` from a later process with the
# same sandbox_group (mirrors sandboxsched's metadata filter).
LABEL_MANAGED_BY = "managed_by"
LABEL_SANDBOX_GROUP = "sandbox_group"
_MANAGED_BY_VALUE = "bench-core"

# CR-condition wait cadence for the create path (see _wait_ready) and the
# exec-stream read cadence.
_POLL_INTERVAL = 1.0


def wrap_command(command: str, *, cwd: str | None, env: dict[str, str] | None) -> str:
    """Wrap a command for cwd/env (the k8s exec stream has no native kwargs).

    ``env`` prefixes the command itself (so the vars apply to it, not to a
    preceding ``cd``); ``cwd`` runs ``cd`` first so both apply to the command.
    No wrapping when neither is set. Shared by the manager's exec paths and
    the adapter's ``exec`` so every command sees the same semantics.
    """
    if cwd is None and not env:
        return command
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in (env or {}).items())
    full = f"{env_prefix} {command}".strip()
    if cwd is not None:
        return f"cd {shlex.quote(cwd)} && {full}"
    return full


class SandboxManager(BaseSandboxManager):
    """Sandbox CR lifecycle: SDK seams over the shared lifecycle template.

    Shared stress params (total_count, create_batch_*) are read from
    ``kernel_config``; backend knobs (image, prefix, sandbox_group, resources,
    TTL) from ``agentsandbox_config``. Readiness is delegated to
    :class:`ReadyChecker` via the base (provider-transparent
    ``_ready_config`` -- no agentsandbox readiness knobs).
    """

    _handle_attr = "sandbox_obj"
    _noun = "Sandbox"
    _id_attr = "sandbox_id"
    # k8s deletes are hard removes like docker's; preserve docker's KILLED
    # status on cleanup so stats read the same across backends.
    _set_killed_on_cleanup = True

    def __init__(self, kernel_config: Any, agentsandbox_config: Config, stop_event: Event) -> None:
        super().__init__(kernel_config, stop_event)
        self.config = agentsandbox_config
        core, custom = self._build_clients()
        self.core = core
        self.custom = custom

    def _build_clients(self) -> tuple[Any, Any]:
        """Load the kubeconfig + build the API clients.

        Single seam tests override to inject fakes without touching a real
        cluster (or even having the SDK installed). ``kubeconfig``/``context``
        on the Config override the client's own default resolution.
        """
        if k8s_client is None:
            raise RuntimeError("kubernetes SDK not installed; install it with: pip install -e '.[agentsandbox]'")
        if self.config.kubeconfig or self.config.context:
            k8s_config.load_kube_config(config_file=self.config.kubeconfig or None, context=self.config.context or None)
        else:
            k8s_config.load_kube_config()
        return k8s_client.CoreV1Api(), k8s_client.CustomObjectsApi()

    # --------------------------------------------------------- subclass seams
    def _new_state(self, index: int, *, batch_id: int = -1, external_id: str = "") -> SandboxState:
        # On detect, external_id is the real CR name; on create, generate the
        # prefixed name so the create path can replace a stale same-name CR.
        return SandboxState(
            sandbox_id=index,
            sandbox_obj=external_id or f"{self.config.container_prefix}-{index}",
            batch_id=batch_id,
        )

    def _create_single(
        self, state: SandboxState, *, metadata: dict[str, str] | None = None, template: str | None = None
    ) -> dict:
        """Create one Sandbox CR and wait for its Ready condition.

        Records submit->ready timing; the base runs the workflow readiness
        probe after this returns success and maps the result onto
        ``creation_metrics``. ``metadata`` lands as pod annotations (operator
        visibility only -- annotations accept arbitrary values, unlike label
        values); the ownership labels are merged separately and cannot be
        overridden -- they scope ``--cleanup`` / ``--detect`` to this config's
        fleet. A missing ``sandbox_group`` fails fast here (mirror
        sandboxsched) rather than listing every sandbox in the namespace later.
        """
        state.creation_metrics.status = BackendSandboxStatus.CREATING
        state.creation_metrics.submit_time = time.time()

        resolved = template or self.config.template

        if not self.config.sandbox_group:
            err = "sandbox_group is required (Sandbox ownership labels); set it under the agentsandbox: block"
            state.creation_metrics.create_ready_time = time.time()
            return {"success": False, "create_elapsed": 0.0, "error": err, "template": resolved}

        name = str(state.sandbox_obj)
        try:
            self._delete_stale(name)
            body = self._sandbox_body(name, resolved)
            self._apply_pod_metadata(body, metadata)
            self.custom.create_namespaced_custom_object(
                group=SANDBOX_API_GROUP,
                version=SANDBOX_API_VERSION,
                namespace=self.config.namespace,
                plural=SANDBOX_PLURAL,
                body=body,
            )
            self._wait_ready(name)
            now = time.time()
            state.creation_metrics.create_ready_time = now
            state.creation_metrics.create_elapsed = now - state.creation_metrics.submit_time
            state.creation_metrics.status = BackendSandboxStatus.CREATED
            return {
                "success": True,
                "create_elapsed": state.creation_metrics.create_elapsed,
                "error": "",
                "template": resolved,
            }
        except Exception as e:
            state.creation_metrics.create_ready_time = time.time()
            return {
                "success": False,
                "create_elapsed": 0.0,
                "error": str(e),
                "template": resolved,
            }

    def _list_existing(self) -> list:
        """List Sandbox CRs scoped to the ownership labels (see LABEL_*)."""
        listed = self.custom.list_namespaced_custom_object(
            group=SANDBOX_API_GROUP,
            version=SANDBOX_API_VERSION,
            namespace=self.config.namespace,
            plural=SANDBOX_PLURAL,
            label_selector=self._ownership_selector(),
        )
        return listed.get("items") or []

    def _external_id(self, listed: Any) -> str:
        return str((listed.get("metadata") or {}).get("name") or "")

    def _attach(self, listed: Any) -> Any:
        # The CR name IS the handle: the backing pod is named after the
        # Sandbox, and every API call takes (name, namespace-from-config).
        return self._external_id(listed)

    def _kill_one(self, state: SandboxState) -> None:
        try:
            self.custom.delete_namespaced_custom_object(
                group=SANDBOX_API_GROUP,
                version=SANDBOX_API_VERSION,
                namespace=self.config.namespace,
                plural=SANDBOX_PLURAL,
                name=str(state.sandbox_obj),
            )
        except ApiException as e:
            if e.status != 404:
                raise  # already gone: treat as success (idempotent kill)

    def _exec_probe(self, handle: Any, cmd: str, timeout: int) -> tuple[int, str, str]:
        return self.pod_exec(str(handle), cmd, timeout=timeout)

    def _create_header_extras(self) -> list[str]:
        return [
            f"Image: {self.config.template}",
            f"Spec:  {self.config.resources or 'cluster defaults'}",
            f"Namespace: {self.config.namespace}",
        ]

    # ------------------------------------------------------------- CR plumbing
    def _ownership_labels(self) -> dict[str, str]:
        """The CR labels stamping ownership (see LABEL_*)."""
        return {LABEL_MANAGED_BY: _MANAGED_BY_VALUE, LABEL_SANDBOX_GROUP: self.config.sandbox_group}

    def _ownership_selector(self) -> str:
        """The label selector matching sandboxes this config owns (see LABEL_*)."""
        return f"{LABEL_MANAGED_BY}={_MANAGED_BY_VALUE},{LABEL_SANDBOX_GROUP}={self.config.sandbox_group}"

    def _delete_stale(self, name: str) -> None:
        """Remove a stale same-name Sandbox CR (recreate path); ignore 404.

        Mirrors docker's same-name container removal: a leftover CR from an
        earlier run would 409 (AlreadyExists) the create. The ownership
        labels are checked first -- a name collision with another team's CR
        (same prefix, shared namespace) is refused, never deleted.
        """
        try:
            existing = self._get(name)
        except ApiException as e:
            if e.status != 404:
                raise
            return
        labels = (existing.get("metadata") or {}).get("labels") or {}
        if (
            labels.get(LABEL_MANAGED_BY) != _MANAGED_BY_VALUE
            or labels.get(LABEL_SANDBOX_GROUP) != self.config.sandbox_group
        ):
            raise RuntimeError(
                f"Sandbox '{name}' exists with foreign ownership labels {labels};"
                " refusing to delete it (container_prefix collision on a shared namespace?)"
            )
        self.custom.delete_namespaced_custom_object(
            group=SANDBOX_API_GROUP,
            version=SANDBOX_API_VERSION,
            namespace=self.config.namespace,
            plural=SANDBOX_PLURAL,
            name=name,
        )

    def _sandbox_body(self, name: str, image: str) -> dict[str, Any]:
        """Build the Sandbox CR body (pod spec + server-side TTL)."""
        pod_spec: dict[str, Any] = {
            "containers": [self._container_spec(image)],
            # No in-sandbox workload needs the namespace SA token; upstream
            # defaults it false for template-provisioned sandboxes -- match
            # that secure-by-default posture for direct Sandbox creation too.
            "automountServiceAccountToken": False,
        }
        if self.config.runtime_class_name:
            pod_spec["runtimeClassName"] = self.config.runtime_class_name
        # datetime.UTC (UP017) needs py311; the kernel's runtime floor is py310.
        expires = datetime.now(timezone.utc) + timedelta(seconds=self.config.sandbox_ttl)  # noqa: UP017
        shutdown_time = expires.isoformat(timespec="seconds")
        # SandboxSpec embeds Lifecycle json:",inline", so shutdownTime/
        # shutdownPolicy are TOP-LEVEL spec properties in the v1beta1 CRD --
        # a nested spec.lifecycle object would be silently pruned by the API
        # server and the TTL would never land.
        return {
            "apiVersion": f"{SANDBOX_API_GROUP}/{SANDBOX_API_VERSION}",
            "kind": "Sandbox",
            "metadata": {"name": name, "labels": self._ownership_labels()},
            "spec": {
                "podTemplate": {"spec": pod_spec},
                "shutdownTime": shutdown_time,
                "shutdownPolicy": "Delete",
            },
        }

    def _container_spec(self, image: str) -> dict[str, Any]:
        spec: dict[str, Any] = {"name": self.config.pod_container or "sandbox", "image": image}
        if self.config.image_pull_policy:
            spec["imagePullPolicy"] = self.config.image_pull_policy
        if self.config.resources:
            spec["resources"] = {"requests": dict(self.config.resources), "limits": dict(self.config.resources)}
        return spec

    def _apply_pod_metadata(self, body: dict[str, Any], metadata: dict[str, str] | None) -> None:
        """Caller metadata -> pod annotations (visibility only).

        The create_one (ephemeral) path forwards runner/trajectory labels for
        operator visibility; annotations accept arbitrary values, so they go
        on the pod template instead of the CR labels (whose values must be
        k8s-valid and whose scoping keys must not be overridable).
        """
        if not metadata:
            return
        pod_meta = body["spec"]["podTemplate"].setdefault("metadata", {})
        pod_meta["annotations"] = {**(pod_meta.get("annotations") or {}), **metadata}

    def _wait_ready(self, name: str) -> dict[str, Any]:
        """Poll the Sandbox CR until its Ready condition is True.

        A 1s poll (not a watch) keeps the create path simple; the granularity
        loss is well under the fleet-creation timing noise. A terminal pod
        (Finished condition: PodFailed/PodSucceeded) fails fast -- a broken
        image would otherwise burn the whole ``ready_timeout``. Raises on
        timeout or terminal failure (mapped to create failure by
        ``_create_single``).
        """
        deadline = time.monotonic() + self.config.ready_timeout
        while True:
            sandbox = self._get(name)
            if self._ready_condition(sandbox) is True:
                return sandbox
            finished = self._finished_reason(sandbox)
            if finished:
                raise RuntimeError(f"Sandbox '{name}' backing pod finished: {finished}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Sandbox '{name}' not Ready within {self.config.ready_timeout}s")
            time.sleep(_POLL_INTERVAL)

    def _get(self, name: str) -> dict[str, Any]:
        return self.custom.get_namespaced_custom_object(
            group=SANDBOX_API_GROUP,
            version=SANDBOX_API_VERSION,
            namespace=self.config.namespace,
            plural=SANDBOX_PLURAL,
            name=name,
        )

    @staticmethod
    def _conditions(sandbox: dict[str, Any]) -> list[dict[str, Any]]:
        return (sandbox.get("status") or {}).get("conditions") or []

    def _ready_condition(self, sandbox: dict[str, Any]) -> bool:
        """True only when the CR's Ready condition is True (else False)."""
        for cond in self._conditions(sandbox):
            if cond.get("type") == "Ready":
                return cond.get("status") == "True"
        return False

    def _finished_reason(self, sandbox: dict[str, Any]) -> str | None:
        """The Finished condition's reason when the pod went terminal."""
        for cond in self._conditions(sandbox):
            if cond.get("type") == "Finished" and cond.get("status") == "True":
                return str(cond.get("reason", "Finished"))
        return None

    def _state_of(self, listed: Any) -> str:
        """Normalized fleet state of one listed Sandbox CR (base hook override).

        The base reads ``listed.state``, which a CR dict lacks -- without the
        override the monitor's fleet-state series would be all-unknown. Ready
        True -> ``running``; operatingMode Suspended -> ``paused`` (volumes
        retained, backing pod gone); everything else -> ``other``.
        """
        try:
            if self._ready_condition(listed) is True:
                return "running"
            if ((listed.get("spec") or {}).get("operatingMode") or "Running") == "Suspended":
                return "paused"
        except Exception:
            pass
        return "other"

    # ------------------------------------------------- agentsandbox-specific
    def pod_exec(
        self,
        pod_name: str,
        command: str,
        *,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Run ``command`` in the sandbox pod via the API server's exec stream.

        Returns ``(exit_code, stdout, stderr)``. The k8s exec stream has no
        cwd/env kwargs, so commands are wrapped the same way the docker
        adapter wraps them (``cd`` + env prefix, both shell-quoted).
        ``timeout`` bounds the WHOLE session: the connect request gets
        ``_request_timeout`` (which on the real client only bounds the
        websocket connect, not the stream) and the read loop enforces a
        monotonic deadline. None = unbounded -- kernel-facing adapter execs
        run under their own watchdog thread instead.
        """
        wrapped = wrap_command(command, cwd=cwd, env=env)
        request = k8s_stream(
            self.core.connect_get_namespaced_pod_exec,
            pod_name,
            self.config.namespace,
            container=self.config.pod_container or None,
            command=["/bin/sh", "-c", wrapped],
            stderr=True,
            stdout=True,
            stdin=False,
            tty=False,
            _preload_content=False,
            **({"_request_timeout": timeout} if timeout else {}),
        )
        deadline = time.monotonic() + timeout if timeout else None
        stdout = ""
        stderr = ""
        try:
            while request.is_open():
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"exec stream did not finish within {timeout}s: {command[:80]}")
                request.update(timeout=_POLL_INTERVAL)
                if request.peek_stdout():
                    stdout += request.read_stdout()
                if request.peek_stderr():
                    stderr += request.read_stderr()
            # Drain frames buffered before the server closed the channel.
            if request.peek_stdout():
                stdout += request.read_stdout()
            if request.peek_stderr():
                stderr += request.read_stderr()
        finally:
            request.close()
        try:
            exit_code = request.returncode
        except TypeError:
            # A stream that closed without a status frame (pod deleted
            # mid-exec, connection dropped) makes the client's channel-3
            # parser raise a bare TypeError -- surface it diagnosably.
            raise RuntimeError(
                "exec stream closed without an exit status frame (pod gone or connection dropped?)"
            ) from None
        if exit_code is None:
            exit_code = 0  # only reachable on clients whose returncode stays unset
        return exit_code, stdout, stderr

    def check_alive(self, state: SandboxState) -> bool:
        """Liveness via the Sandbox CR (Ready condition True)."""
        if not state.sandbox_obj or not state.is_alive:
            return False
        try:
            return self._ready_condition(self._get(str(state.sandbox_obj)))
        except Exception:
            return False

    def start_browser_backend(self, state: SandboxState) -> tuple[bool, str]:
        """Start OpenClaw browser backend (hot start) inside the sandbox pod."""
        handle = state.sandbox_obj
        if not handle:
            return False, "No sandbox handle"
        try:
            exit_code, stdout, stderr = self.pod_exec(
                str(handle), "openclaw browser status && start || openclaw browser start", timeout=60
            )
            if exit_code == 0:
                state.browser_started = True
                return True, ""
            output = (stdout + stderr)[:200]
            return False, f"exit_code={exit_code}, output={output}"
        except Exception as e:
            return False, str(e)

    def clear_browser_cache(self, state: SandboxState) -> bool:
        """Clear browser cache for a clean test (rm user-data)."""
        handle = state.sandbox_obj
        if not handle:
            return False
        try:
            exit_code, _stdout, _stderr = self.pod_exec(
                str(handle), "rm -rf /root/.openclaw/browser/openclaw/user-data", timeout=30
            )
            return exit_code == 0
        except Exception:
            return False

    # ----------------------------------------------------- adapter alias (state)
    @property
    def sandbox_states(self) -> dict[int, SandboxState]:
        """Adapter-facing alias for the base state registry."""
        return self._states
