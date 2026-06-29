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


def test_build_command_includes_per_model_subprocess_options(pm, monkeypatch, tmp_path):
    executable = tmp_path / "mlx_lm.server"
    executable.write_text("")
    monkeypatch.setattr(pm, "_MLX_LM_SERVER", executable)

    cfg = pm.config.ModelConfig(
        name="ornith",
        type="text",
        hf_path="mlx-community/Ornith-1.0-35B-bf16",
        context_length=8192,
        chat_template_args={"enable_thinking": False},
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.05,
        prompt_cache_size=2,
        extra_args=["--log-level", "DEBUG"],
    )

    cmd = pm._build_command(cfg)

    assert cmd[:2] == [str(executable), "--model"]
    assert cmd[cmd.index("--max-tokens") + 1] == "8192"
    assert [
        cmd[cmd.index("--chat-template-args")],
        cmd[cmd.index("--chat-template-args") + 1],
    ] == ["--chat-template-args", '{"enable_thinking": false}']
    assert cmd[cmd.index("--temp") + 1] == "0.6"
    assert cmd[cmd.index("--top-p") + 1] == "0.95"
    assert cmd[cmd.index("--top-k") + 1] == "20"
    assert cmd[cmd.index("--min-p") + 1] == "0.05"
    assert cmd[cmd.index("--prompt-cache-size") + 1] == "2"
    assert cmd[-2:] == ["--log-level", "DEBUG"]


def test_build_command_uses_vlm_supported_options_for_vision(pm, monkeypatch, tmp_path):
    executable = tmp_path / "mlx_vlm.server"
    executable.write_text("")
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", executable)

    cfg = pm.config.ModelConfig(
        name="qwen3-vl",
        type="vision",
        hf_path="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        context_length=32768,
        max_kv_cache_size=8192,
        temperature=0.6,
        top_p=0.95,
        prompt_cache_size=2,
        extra_args=["--prefill-step-size", "512"],
    )

    cmd = pm._build_command(cfg)

    assert cmd[:2] == [str(executable), "--model"]
    assert "--max-tokens" not in cmd
    assert "--temp" not in cmd
    assert "--top-p" not in cmd
    assert "--prompt-cache-size" not in cmd
    assert cmd[cmd.index("--max-kv-size") + 1] == "8192"
    assert cmd[-2:] == ["--prefill-step-size", "512"]


def test_log_filename_sanitizes_hugging_face_model_ids(pm):
    assert (
        pm._log_filename("mlx-community/Qwen3.6-35B-A3B-nvfp4")
        == "mlx-community_Qwen3.6-35B-A3B-nvfp4.log"
    )


@pytest.mark.asyncio
async def test_switch_emits_downloading_event_when_not_cached(pm, monkeypatch):
    """When a model isn't cached, the switch enters DOWNLOADING and emits a
    model.downloading event so the hang is visible to the user."""
    import mlx_serve.events as events

    monkeypatch.setattr(pm, "_is_model_cached", lambda cfg: False)
    monkeypatch.setattr(pm, "_MLX_LM_SERVER", pm.pathlib.Path(__file__))

    class FakeProc:
        pid = 4321

        def poll(self):
            return None

    monkeypatch.setattr(pm.subprocess, "Popen", lambda *a, **k: FakeProc())
    # Don't run the real health loop (would poll a nonexistent server).
    monkeypatch.setattr(pm.asyncio, "create_task", lambda coro: coro.close())

    try:
        await pm._switch_model("test-text-model")

        downloading = events.get_events(event_type="model.downloading")
        assert any(e["model"] == "test-text-model" for e in downloading)
        assert pm._state == pm.ModelState.DOWNLOADING
    finally:
        pm._process = None
        pm._active_model = None
        pm._state = pm.ModelState.IDLE
