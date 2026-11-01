"""Tests for vm-monitor startup hardening (vm_monitor/cli.py) and the
numastat-fallback circuit breaker (vm_monitor/base.py).

Hardening goals on a memory-pressured oversub host: oom_score_adj=-1000
(OOM-killer immunity, inherited by capture children) + optional CPU pinning
(sched_setaffinity). Both degrade to a warning -- hardening must never keep
the monitor from starting. The circuit breaker keeps the per-VM sampling hot
path fork-free when the numa_maps fast path fails under VM churn.
"""

from __future__ import annotations

import logging

import pytest

import vm_monitor.base as base
import vm_monitor.cli as cli
from vm_monitor.base import VMMonitorBase


class _StubMonitor(VMMonitorBase):
    """Concrete VMMonitorBase that touches no real /proc or /sys at runtime."""

    def get_process_names(self):  # noqa: N802 - ABC seam
        return ("test_process",)

    def extract_vm_id(self, pid, cmdline):  # noqa: N802 - ABC seam
        return "vm0"

    def get_vms_realtime(self):  # noqa: N802 - ABC seam
        return []

    def get_monitor_title(self):
        return "StubMonitor"

    def get_no_vm_message(self):
        return "No VMs detected"

    def get_csv_filename_prefix(self):
        return "stub_monitor"


_EMPTY_NUMASTAT = {
    "total_mb": 0.0,
    "huge_mb": 0.0,
    "private_mb": 0.0,
    "heap_mb": 0.0,
    "stack_mb": 0.0,
    "per_node": {},
}


# ---------- cpu list parsing ----------


def test_parse_cpu_list_single_and_ranges():
    assert cli._parse_cpu_list("318,319") == [318, 319]
    assert cli._parse_cpu_list("316-319") == [316, 317, 318, 319]
    assert cli._parse_cpu_list("0, 2, 8-9") == [0, 2, 8, 9]
    assert cli._parse_cpu_list("7") == [7]


def test_parse_cpu_list_rejects_malformed():
    with pytest.raises(ValueError):
        cli._parse_cpu_list("8-3")  # inverted range
    with pytest.raises(ValueError):
        cli._parse_cpu_list("  ")  # empty
    with pytest.raises(ValueError):
        cli._parse_cpu_list("a,b")  # non-numeric


# ---------- _harden_process ----------


def test_harden_sets_affinity_and_tolerates_bad_spec(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(cli.os, "sched_setaffinity", lambda pid, cpus: seen.setdefault("args", (pid, set(cpus))))
    cli._harden_process(0, "318,319")  # oom_score_adj=0 -> no /proc write
    assert seen["args"] == (0, {318, 319})
    out = capsys.readouterr().out
    assert "HARDENING" in out and "318" in out

    # A malformed spec degrades to a warning -- never raises, never pins.
    cli._harden_process(0, "not-a-cpu")
    assert "WARN" in capsys.readouterr().out


def test_harden_writes_oom_score_adj(monkeypatch, capsys):
    written = {}

    class _FakeFH:
        def __init__(self, path, mode):
            written["path"] = str(path)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, s):
            written["value"] = s

    real_open = open

    def fake_open(path, mode="r", *a, **k):
        if str(path) == "/proc/self/oom_score_adj":
            return _FakeFH(path, mode)
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    cli._harden_process(-1000, None)
    assert written == {"path": "/proc/self/oom_score_adj", "value": "-1000"}
    assert "HARDENING" in capsys.readouterr().out


def test_harden_oom_write_failure_degrades(monkeypatch, capsys):
    def boom(path, mode="r", *a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr("builtins.open", boom)
    cli._harden_process(-1000, None)  # must not raise
    assert "WARN" in capsys.readouterr().out


# ---------- numastat fallback circuit breaker ----------


def test_numastat_broken_short_circuits_without_fork(monkeypatch):
    mon = _StubMonitor()
    mon._numastat_broken = True

    def no_fork(*a, **k):
        raise AssertionError("subprocess must not be spawned when breaker is open")

    monkeypatch.setattr(base.subprocess, "run", no_fork)
    monkeypatch.setattr(mon, "get_vm_memory_from_numa_maps", lambda pid: {})
    assert mon.get_vm_memory_from_numastat(123) == _EMPTY_NUMASTAT


def test_numastat_budget_exhausted_trips_breaker(monkeypatch):
    mon = _StubMonitor()
    mon._numastat_budget = 0

    def no_fork(*a, **k):
        raise AssertionError("budget exhaustion must not fork")

    monkeypatch.setattr(base.subprocess, "run", no_fork)
    monkeypatch.setattr(mon, "get_vm_memory_from_numa_maps", lambda pid: {})
    mon.get_vm_memory_from_numastat(123)
    assert mon._numastat_broken is True
    assert mon._numastat_warned is True


def test_numastat_fork_failure_trips_breaker(monkeypatch):
    mon = _StubMonitor()
    mon._numastat_budget = 8
    calls = {"n": 0}

    def fail_run(*a, **k):
        calls["n"] += 1
        raise OSError("Cannot allocate memory")

    monkeypatch.setattr(base.subprocess, "run", fail_run)
    monkeypatch.setattr(mon, "get_vm_memory_from_numa_maps", lambda pid: {})
    assert mon.get_vm_memory_from_numastat(1)["total_mb"] == 0.0
    assert mon.get_vm_memory_from_numastat(2)["total_mb"] == 0.0
    assert calls["n"] == 1  # breaker opened after the first fork failure
    assert mon._numastat_broken is True


def test_numastat_budget_refills_per_cycle():
    mon = _StubMonitor()
    mon._numastat_budget = 0
    mon._numastat_broken = False
    # Hermetic cycle: every collector off, stubbed /proc readers, no VMs.
    mon.disable_collectors(set(base._ALL_SAMPLE_COLLECTORS))
    mon._read_meminfo = lambda: {}
    mon._read_vmstat = lambda: {}
    mon._read_proc_stat = lambda: {}
    mon.get_vms_realtime = lambda: []
    mon.collect_sample()
    assert mon._numastat_budget == base._NUMASTAT_MAX_SPAWNS_PER_CYCLE


# ---------- slow-cycle warning ----------


def test_warn_slow_cycle_threshold_and_cooldown(monkeypatch, caplog):
    mon = _StubMonitor()
    mon.interval = 2
    monkeypatch.setattr(base.time, "monotonic", lambda: 100.0)
    with caplog.at_level(logging.WARNING, logger="vm_monitor.base"):
        mon._warn_slow_cycle(1.0)  # under threshold -> silent
        assert not [r for r in caplog.records if "slow sampling" in r.message]
        mon._warn_slow_cycle(30.0)  # over threshold -> one WARNING
        mon._warn_slow_cycle(30.0)  # inside cooldown -> still one
    assert sum(1 for r in caplog.records if "slow sampling" in r.message) == 1
