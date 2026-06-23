"""Tests for process_manager lifecycle logic: failure cooldown and
download-aware startup timeout.

These exercise the pure decision helpers plus ensure_model/_switch_model
behavior, without spawning real MLX subprocesses.
"""

import importlib
import time

import pytest
from fastapi import HTTPException


@pytest.fixture()
def pm(tmp_config, log_dir):
    """Reload config + process_manager against the temp config, fresh state."""
    import mlx_serve.config

    importlib.reload(mlx_serve.config)
    import mlx_serve.events as events

    events.configure(log_dir)

    import mlx_serve.process_manager as process_manager

    importlib.reload(process_manager)
    return process_manager


# ---------------------------------------------------------------------------
# Bug 2 — failure cooldown / circuit breaker
# ---------------------------------------------------------------------------


def test_failure_cooldown_blocks_recent_failure(pm):
    pm._record_failure("test-text-model", "startup_timeout")
    assert pm._failure_cooldown_remaining("test-text-model") > 0


def test_failure_cooldown_expires_after_window(pm):
    pm._record_failure("test-text-model", "startup_timeout")
    # Pretend the failure happened a full cooldown ago.
    pm._last_failure_at = time.monotonic() - (pm.config.FAILURE_COOLDOWN + 1)
    assert pm._failure_cooldown_remaining("test-text-model") == 0


def test_failure_cooldown_only_affects_failed_model(pm):
    pm._record_failure("test-text-model", "startup_timeout")
    assert pm._failure_cooldown_remaining("test-vision-model") == 0


def test_clear_failure_resets_cooldown(pm):
    pm._record_failure("test-text-model", "startup_timeout")
    pm._clear_failure()
    assert pm._failure_cooldown_remaining("test-text-model") == 0


@pytest.mark.asyncio
async def test_ensure_model_short_circuits_during_cooldown(pm, monkeypatch):
    """A request for a recently-failed model returns 503 immediately and
    does NOT spawn another subprocess (the infinite-respawn bug)."""
    spawned = False

    async def fake_switch(model_name):
        nonlocal spawned
        spawned = True

    monkeypatch.setattr(pm, "_switch_model", fake_switch)
    pm._record_failure("test-text-model", "startup_timeout")

    with pytest.raises(HTTPException) as exc:
        await pm.ensure_model("test-text-model")

    assert exc.value.status_code == 503
    assert spawned is False


# ---------------------------------------------------------------------------
# Bug 1 — download-aware readiness timeout + downloading state/event
# ---------------------------------------------------------------------------


def test_readiness_timeout_uses_download_timeout_when_not_cached(pm, monkeypatch):
    monkeypatch.setattr(pm, "_is_model_cached", lambda cfg: False)
    cfg = pm.config.MODELS["test-text-model"]
    assert pm._readiness_timeout(cfg) == pm.config.DOWNLOAD_TIMEOUT


def test_readiness_timeout_uses_startup_timeout_when_cached(pm, monkeypatch):
    monkeypatch.setattr(pm, "_is_model_cached", lambda cfg: True)
    cfg = pm.config.MODELS["test-text-model"]
    assert pm._readiness_timeout(cfg) == pm.config.STARTUP_TIMEOUT


def test_diagnose_failure_reports_actual_timeout(pm):
    """On a startup/download timeout, the reported timeout_seconds reflects the
    deadline actually used, not the bare STARTUP_TIMEOUT."""

    class StillRunning:
        returncode = None

        def poll(self):
            return None  # process alive -> this is a timeout, not a crash

    pm._process = StillRunning()
    detail = pm._diagnose_failure(timeout_seconds=pm.config.DOWNLOAD_TIMEOUT)
    assert detail["reason"] == "startup_timeout"
    assert detail["timeout_seconds"] == pm.config.DOWNLOAD_TIMEOUT


@pytest.mark.asyncio
async def test_switch_emits_downloading_event_when_not_cached(pm, monkeypatch):
    """When a model isn't cached, the switch enters DOWNLOADING and emits a
    model.downloading event so the hang is visible to the user."""
    import mlx_serve.events as events

    monkeypatch.setattr(pm, "_is_model_cached", lambda cfg: False)

    class FakeProc:
        pid = 4321

        def poll(self):
            return None

    monkeypatch.setattr(pm.subprocess, "Popen", lambda *a, **k: FakeProc())
    # Don't run the real health loop (would poll a nonexistent server).
    monkeypatch.setattr(pm.asyncio, "create_task", lambda coro: coro.close())

    await pm._switch_model("test-text-model")

    downloading = events.get_events(event_type="model.downloading")
    assert any(e["model"] == "test-text-model" for e in downloading)
    assert pm._state == pm.ModelState.DOWNLOADING
