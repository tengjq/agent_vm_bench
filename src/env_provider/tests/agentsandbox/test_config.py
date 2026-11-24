"""Tests for the agentsandbox backend Config (``agentsandbox:`` block)."""

from __future__ import annotations

from env_provider.agentsandbox.config import Config


class TestConfigFromRaw:
    def test_reads_block_fields(self):
        raw = {
            "agentsandbox": {
                "kubeconfig": "/root/.kube/config-bench",
                "context": "bench-cluster",
                "namespace": "bench-ns",
                "template": "ubuntu-openclaw-coding-python:24.04",
                "container_prefix": "asbx-replay",
                "sandbox_group": "sg-k8s",
                "resources": {"cpu": "2", "memory": "4Gi"},
                "runtime_class_name": "gvisor",
                "image_pull_policy": "IfNotPresent",
                "pod_container": "main",
                "ready_timeout": 600,
                "sandbox_ttl": 3600,
            }
        }
        cfg = Config.from_raw(raw)
        assert cfg.kubeconfig == "/root/.kube/config-bench"
        assert cfg.context == "bench-cluster"
        assert cfg.namespace == "bench-ns"
        assert cfg.template == "ubuntu-openclaw-coding-python:24.04"
        assert cfg.container_prefix == "asbx-replay"
        assert cfg.sandbox_group == "sg-k8s"
        assert cfg.resources == {"cpu": "2", "memory": "4Gi"}
        assert cfg.runtime_class_name == "gvisor"
        assert cfg.image_pull_policy == "IfNotPresent"
        assert cfg.pod_container == "main"
        assert cfg.ready_timeout == 600
        assert cfg.sandbox_ttl == 3600

    def test_missing_block_uses_defaults(self):
        cfg = Config.from_raw({})
        assert cfg.kubeconfig == ""  # defer to the kubernetes client's defaults
        assert cfg.context == ""
        assert cfg.namespace == "default"
        assert cfg.template == "python:3.11"
        assert cfg.container_prefix == "agentsbx"
        assert cfg.sandbox_group == ""  # required at create -- fail fast there
        assert cfg.resources is None
        assert cfg.runtime_class_name == ""
        assert cfg.image_pull_policy == ""
        assert cfg.pod_container == ""
        assert cfg.ready_timeout == 300
        assert cfg.sandbox_ttl == 86400

    def test_empty_block_uses_defaults(self):
        cfg = Config.from_raw({"agentsandbox": {}})
        assert cfg.sandbox_group == ""
        assert cfg.namespace == "default"
