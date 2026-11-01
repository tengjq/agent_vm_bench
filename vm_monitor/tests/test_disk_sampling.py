"""Tests for the 1-second disk sub-sampling in vm_monitor/base.py.

collect_disk_stats divides by the ACTUAL monotonic elapsed time between reads
(not the nominal sampling interval), and _disk_subsample_sleep ticks disk + ublk
together every 1s so disk bandwidth stays a true per-second rate at any
interval. These tests patch /sys reads + time.monotonic/time.sleep so they run
on any host (no real block devices needed).
"""

from __future__ import annotations

import io

import vm_monitor.base as base
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


def _stat_line(*, sectors_read=0, sectors_written=0, inflight=0, ms_io=0) -> str:
    """Build a minimal 11-field /sys/block/<dev>/stat line.

    Fields: reads_completed, reads_merged, sectors_read, read_ms,
    writes_completed, writes_merged, sectors_written, write_ms, inflight,
    ms_io, weighted_ms.
    """
    return f"0 0 {sectors_read} 0 0 0 {sectors_written} 0 {inflight} {ms_io} 0"


def test_collect_disk_stats_divides_by_actual_elapsed_not_interval(monkeypatch):
    """r_mb_s uses the real monotonic delta between reads, so it is correct
    even when self.interval differs from the actual sampling cadence."""
    mon = _StubMonitor()
    mon.target_disks = ["sda"]
    mon.interval = 5  # deliberately != the 2s elapsed below

    # First call: baseline (prev=None -> zero rates). Second: 2048 sectors
    # (1 MiB) over a 2s window -> 0.5 MB/s.
    snaps = iter([_stat_line(sectors_read=0), _stat_line(sectors_read=2048)])

    def fake_open(path, *a, **k):
        assert path == "/sys/block/sda/stat"
        return io.StringIO(next(snaps))

    monkeypatch.setattr("builtins.open", fake_open)

    # monotonic: 1000.0 on the baseline read, 1002.0 on the second read
    # (2s elapsed). If the code used self.interval=5 instead, r_mb_s would be
    # 1.0/5 = 0.2, not 0.5.
    ticks = iter([1000.0, 1002.0])
    monkeypatch.setattr(base.time, "monotonic", lambda: next(ticks))

    mon.collect_disk_stats()  # baseline (zero rates, prev=None)
    mon.collect_disk_stats()  # 1 MiB over 2s

    row = mon.disk_history[-1]["disks"]["sda"]
    assert row["r_mb"] == 1.0  # 2048 sectors * 512 B = 1 MiB
    assert row["r_mb_s"] == 0.5  # 1 MiB / 2s actual elapsed (NOT 0.2 from interval=5)


def test_collect_disk_stats_baseline_row_has_zero_rates(monkeypatch):
    """First read (prev=None) yields a zero-rate baseline row + sets the
    monotonic baseline; the second read computes a real delta from it."""
    mon = _StubMonitor()
    mon.target_disks = ["sda"]

    snaps = iter([_stat_line(sectors_read=0), _stat_line(sectors_read=4096)])
    monkeypatch.setattr("builtins.open", lambda p, *a, **k: io.StringIO(next(snaps)))
    ticks = iter([10.0, 11.0])
    monkeypatch.setattr(base.time, "monotonic", lambda: next(ticks))

    mon.collect_disk_stats()
    assert mon.disk_history[-1]["disks"]["sda"]["r_mb_s"] == 0.0  # baseline

    mon.collect_disk_stats()
    # 4096 sectors = 2 MiB over 1s elapsed -> 2.0 MB/s
    assert mon.disk_history[-1]["disks"]["sda"]["r_mb_s"] == 2.0


def test_disk_subsample_tick_calls_disk_and_ublk_together():
    """One tick runs disk + ublk back-to-back (histories stay index-aligned);
    disabled collectors are skipped."""
    mon = _StubMonitor()
    calls = {"disk": 0, "ublk": 0}

    mon.collect_disk_stats = lambda: calls.__setitem__("disk", calls["disk"] + 1)  # type: ignore[assignment]
    mon.collect_ublk_count = lambda: calls.__setitem__("ublk", calls["ublk"] + 1)  # type: ignore[assignment]

    mon._disk_subsample_tick()
    assert calls["disk"] == 1
    assert calls["ublk"] == 1

    calls["disk"] = calls["ublk"] = 0
    mon.disable_collectors(["disk", "ublk"])
    mon._disk_subsample_tick()
    assert calls["disk"] == 0
    assert calls["ublk"] == 0


def test_disk_subsample_loop_ticks_until_stop(monkeypatch):
    """The dedicated thread body ticks (disk+ublk together) until the stop
    event is set, then exits -- a stalled collect_sample can no longer starve
    the 1s disk/ublk stream."""
    mon = _StubMonitor()
    mon.running = True
    calls = {"disk": 0, "ublk": 0}

    monkeypatch.setattr(mon, "collect_disk_stats", lambda: calls.__setitem__("disk", calls["disk"] + 1))
    monkeypatch.setattr(mon, "collect_ublk_count", lambda: calls.__setitem__("ublk", calls["ublk"] + 1))
    # elapsed 0 -> the inter-tick wait is 1.0s; the mocked wait walks a scripted
    # stop sequence instead of really sleeping.
    monkeypatch.setattr(base.time, "monotonic", lambda: 0.0)
    waits = iter([False, True])

    def fake_wait(timeout=None):
        stop = next(waits)
        if stop:
            mon._disk_thread_stop.set()
        return stop

    monkeypatch.setattr(mon._disk_thread_stop, "wait", fake_wait)

    mon._disk_subsample_loop()
    assert calls["disk"] == 2
    assert calls["ublk"] == 2


def test_disk_subsample_loop_exits_when_not_running():
    """A cleared self.running ends the loop without any tick."""
    mon = _StubMonitor()
    mon.running = False
    calls = {"disk": 0}
    mon.collect_disk_stats = lambda: calls.__setitem__("disk", calls["disk"] + 1)  # type: ignore[assignment]
    mon._disk_subsample_loop()
    assert calls["disk"] == 0


def test_stop_disk_thread_is_idempotent():
    """Stopping a never-started thread is a no-op."""
    mon = _StubMonitor()
    mon._stop_disk_thread()
    assert mon._disk_thread is None
