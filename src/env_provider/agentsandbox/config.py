"""AgentSandbox backend configuration (backend-only fields).

Host-agnostic stress params (total_count, create_batch, task_batch,
benchmark_*, browser, test, warmup, coding, document, report, workflow_type)
live on :class:`bench_core.config.KernelConfig`, read from the shared sections
of the unified YAML by :meth:`KernelConfig.from_raw`. This Config holds only
the agentsandbox-specific knobs -- cluster access (kubeconfig/context/
namespace), the sandbox image, the CR name prefix, the REQUIRED ownership
``sandbox_group``, pod resources, and the ready/TTL budgets -- read from the
``agentsandbox:`` block of that YAML.

Backend model: the provider drives the kubernetes-sigs/agent-sandbox
``Sandbox`` CRD (``agents.x-k8s.io/v1beta1``) through the ``kubernetes``
python client (the optional ``agentsandbox`` extra) and execs commands through
the API server's pod-exec stream, so a bench fleet needs no warm pools, no
sandbox router, and no per-sandbox port-forwards -- only a cluster with the
agent-sandbox controller installed (helm) and a kubeconfig that can reach it.

The ``kubernetes`` package is intentionally not imported here (config loading
is dataclass work only); the manager guards its import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Config:
    """AgentSandbox backend configuration (backend-specific fields only)."""

    # Cluster access. Empty ``kubeconfig``/``context`` defer to the kubernetes
    # client's own defaults (``$KUBECONFIG`` env, else ``~/.kube/config``,
    # current context). ``kubeconfig`` points at an explicit file when the
    # bench host keeps its cluster credentials elsewhere.
    kubeconfig: str = ""
    context: str = ""
    namespace: str = "default"

    # Sandbox image (= the backing pod's container image; the multi-template
    # replay path overrides it per slot) and the Sandbox CR name prefix (names
    # ``f"{container_prefix}-{index}"``; sharding appends ``-s<i>``).
    template: str = "python:3.11"
    container_prefix: str = "agentsbx"

    # REQUIRED ownership tag: every Sandbox CR is labeled
    # ``managed_by=bench-core`` + ``sandbox_group=<this>`` at create, and
    # ``--cleanup`` / ``--detect`` listing scopes to the same pair. A missing
    # group fails fast at create -- without the pair, listing on a shared
    # namespace would surface (and --cleanup would kill) other fleets.
    sandbox_group: str = ""

    # Optional pod knobs. ``resources`` ({"cpu": "2", "memory": "2Gi"}) is set
    # as both requests and limits on the sandbox container;
    # ``runtime_class_name`` selects an isolation runtime (gVisor/Kata);
    # ``image_pull_policy`` empty omits the field (kubelet default).
    resources: dict[str, str] | None = None
    runtime_class_name: str = ""
    image_pull_policy: str = ""

    # Pod container the exec stream runs in; empty -> the pod's single
    # container (multi-container images must name the exec target here).
    pod_container: str = ""

    # Seconds to wait for the Sandbox CR's Ready condition after create. The
    # kernel's own ReadyChecker (uname probe for replay/coding) runs on top of
    # this -- the CR-level wait only proves the pod came up.
    ready_timeout: int = 300

    # Server-side TTL seconds: spec.shutdownTime = now + sandbox_ttl
    # (shutdownPolicy Delete), so a fleet whose cleanup failed is reclaimed by
    # the agent-sandbox controller instead of occupying cluster resources
    # forever. Both are TOP-LEVEL spec properties (SandboxSpec embeds its
    # lifecycle struct inline); NOT a create-wait budget (that is
    # ready_timeout). The default 86400 (24h) keeps a leaked fleet reclaimable
    # within a day; lower it if that is a concern.
    sandbox_ttl: int = 86400

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> Config:
        """Build from the unified YAML's ``agentsandbox:`` block.

        Missing keys fall back to the dataclass defaults, so a minimal block
        (just ``sandbox_group``) still loads.
        """
        backend = raw.get("agentsandbox") or {}
        return cls(
            kubeconfig=str(backend.get("kubeconfig", "") or ""),
            context=str(backend.get("context", "") or ""),
            namespace=str(backend.get("namespace", "default")),
            template=str(backend.get("template", "python:3.11")),
            container_prefix=str(backend.get("container_prefix", "agentsbx")),
            sandbox_group=str(backend.get("sandbox_group", "")),
            resources=backend.get("resources"),
            runtime_class_name=str(backend.get("runtime_class_name", "") or ""),
            image_pull_policy=str(backend.get("image_pull_policy", "") or ""),
            pod_container=str(backend.get("pod_container", "") or ""),
            ready_timeout=int(backend.get("ready_timeout", 300)),
            sandbox_ttl=int(backend.get("sandbox_ttl", 86400)),
        )
