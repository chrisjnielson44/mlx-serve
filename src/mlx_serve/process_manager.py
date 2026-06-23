"""
process_manager.py — subprocess lifecycle for the active mlx model.

State machine: IDLE -> LOADING -> READY / FAILED -> IDLE (on inactivity/shutdown)
"""

import asyncio
import contextlib
import logging
import pathlib
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from enum import Enum

import httpx
import psutil

from . import config, events, metrics

logger = logging.getLogger("mlx-serve.process")

# Resolve executable paths relative to the running venv so subprocesses
# inherit the same environment that has mlx_lm / mlx_vlm installed.
_VENV_BIN = pathlib.Path(sys.executable).parent
_MLX_LM_SERVER = _VENV_BIN / "mlx_lm.server"
_MLX_VLM_SERVER = _VENV_BIN / "mlx_vlm.server"

_LOG_DIR = pathlib.Path(tempfile.gettempdir()) / "mlx-manager-logs"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ModelState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"


_state: ModelState = ModelState.IDLE
_active_model: str | None = None
_process: subprocess.Popen | None = None
_ready_event: asyncio.Event = asyncio.Event()
_switch_lock: asyncio.Lock = asyncio.Lock()
_last_request_at: datetime | None = None
_started_at: datetime = datetime.now(UTC)
_stderr_log_path: pathlib.Path | None = None
_stderr_log_handle = None
_inactivity_timeout: int = config.INACTIVITY_TIMEOUT  # overridable per-request
_loading_started_at: float | None = None  # monotonic time when loading began

# Failure tracking (circuit breaker / cooldown)
_last_failure_at: float | None = None  # monotonic timestamp of last failure
_last_failed_model: str | None = None
_last_failure_reason: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _record_failure(model_name: str, reason: str) -> None:
    """Record a model load failure for cooldown tracking."""
    global _last_failure_at, _last_failed_model, _last_failure_reason
    _last_failure_at = time.monotonic()
    _last_failed_model = model_name
    _last_failure_reason = reason


def _clear_failure() -> None:
    """Reset failure tracking state."""
    global _last_failure_at, _last_failed_model, _last_failure_reason
    _last_failure_at = None
    _last_failed_model = None
    _last_failure_reason = None


def _failure_cooldown_remaining(model_name: str) -> float:
    """Return seconds remaining in the failure cooldown for this model, or 0.0."""
    if _last_failure_at is None or _last_failed_model != model_name:
        return 0.0
    elapsed = time.monotonic() - _last_failure_at
    if elapsed >= config.FAILURE_COOLDOWN:
        return 0.0
    return config.FAILURE_COOLDOWN - elapsed


def _is_model_cached(model_cfg: config.ModelConfig) -> bool:
    """Return True if the model's weights are already in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.file_download import _CACHED_NO_EXIST

        result = try_to_load_from_cache(model_cfg.hf_path, "config.json")
        return isinstance(result, str) and result is not _CACHED_NO_EXIST
    except Exception:
        return False


def _readiness_timeout(model_cfg: config.ModelConfig) -> int:
    """Return the appropriate readiness timeout for this model."""
    if not _is_model_cached(model_cfg):
        return config.DOWNLOAD_TIMEOUT
    return config.STARTUP_TIMEOUT


def _build_command(model_cfg: config.ModelConfig) -> list[str]:
    executable = _MLX_VLM_SERVER if model_cfg.type == "vision" else _MLX_LM_SERVER
    if not executable.exists():
        pkg = "mlx-vlm" if model_cfg.type == "vision" else "mlx-lm"
        extra = "vision" if model_cfg.type == "vision" else "text"
        raise RuntimeError(
            f"{pkg} is not installed (expected: {executable}). Run: pip install mlx-serve[{extra}]"
        )
    cmd = [
        str(executable),
        "--model",
        model_cfg.hf_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(config.MLX_PORT),
    ]
    if model_cfg.context_length > 0:
        cmd += ["--max-tokens", str(model_cfg.context_length)]
    if model_cfg.max_kv_cache_size > 0:
        cmd += ["--max-kv-cache-size", str(model_cfg.max_kv_cache_size)]
    return cmd


def _diagnose_failure(timeout_seconds: int | None = None) -> dict:
    """Inspect subprocess state and stderr to determine failure reason.

    ``timeout_seconds`` is the readiness deadline that was actually applied
    (download vs. startup), reported back when the failure is a timeout.
    """
    detail: dict = {}

    if _process is not None and _process.poll() is not None:
        detail["exit_code"] = _process.returncode
        detail["reason"] = "process_crash"
    else:
        detail["reason"] = "startup_timeout"
        detail["timeout_seconds"] = (
            timeout_seconds if timeout_seconds is not None else config.STARTUP_TIMEOUT
        )

    if _stderr_log_path and _stderr_log_path.exists():
        lines = _stderr_log_path.read_text().splitlines()
        detail["stderr_tail"] = lines[-15:]
        text = "\n".join(lines).lower()
        if any(kw in text for kw in ("out of memory", "memoryerror", "malloc", "mlock failed")):
            detail["reason"] = "oom"
        elif any(kw in text for kw in ("no such file", "not found", "does not exist")):
            detail["reason"] = "model_not_found"
        elif "address already in use" in text:
            detail["reason"] = "port_conflict"

    vm = psutil.virtual_memory()
    detail["ram_percent_at_failure"] = vm.percent
    if vm.percent > 90:
        detail["memory_pressure"] = "critical"

    return detail


async def _terminate_current() -> None:
    global _process, _state, _active_model, _stderr_log_handle, _inactivity_timeout
    prev_model = _active_model
    if _process is not None:
        logger.info(f"Terminating subprocess for model={prev_model} pid={_process.pid}")
        _process.terminate()
        try:
            await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, _process.wait),
                timeout=5,
            )
        except TimeoutError:
            logger.warning(f"Subprocess did not exit gracefully, killing pid={_process.pid}")
            _process.kill()
            await asyncio.get_running_loop().run_in_executor(None, _process.wait)
        _process = None
    if _stderr_log_handle is not None:
        _stderr_log_handle.close()
        _stderr_log_handle = None
    _state = ModelState.IDLE
    _active_model = None
    _inactivity_timeout = config.INACTIVITY_TIMEOUT  # reset per-request override

    if prev_model is not None:
        # Memory snapshot after unload to show freed memory
        snap = metrics.take_memory_snapshot(active_model=None, event="model_unloaded")
        events.emit(
            events.EventType.MODEL_UNLOADED,
            model=prev_model,
            detail={
                "ram_used_gb": snap.ram_used_gb,
                "ram_available_gb": snap.ram_available_gb,
            },
        )


async def _health_check_loop() -> None:
    """Poll /health every 2s until READY or startup timeout."""
    global _state
    model_cfg = config.MODELS.get(_active_model) if _active_model else None
    readiness_timeout = _readiness_timeout(model_cfg) if model_cfg else config.STARTUP_TIMEOUT
    deadline = datetime.now(UTC) + timedelta(seconds=readiness_timeout)
    async with httpx.AsyncClient() as client:
        while datetime.now(UTC) < deadline:
            # Detect early process exit
            if _process is not None and _process.poll() is not None:
                _state = ModelState.FAILED
                _ready_event.set()
                detail = _diagnose_failure(timeout_seconds=readiness_timeout)
                load_ms = (
                    (time.monotonic() - _loading_started_at) * 1000 if _loading_started_at else None
                )
                _record_failure(_active_model, detail.get("reason", "unknown"))
                events.emit(
                    events.EventType.MODEL_FAILED,
                    model=_active_model,
                    detail=detail,
                    duration_ms=load_ms,
                )
                if detail.get("reason") == "oom":
                    events.emit(
                        events.EventType.OOM_DETECTED,
                        model=_active_model,
                        detail=detail,
                    )
                logger.error(
                    f"Model {_active_model} failed to load: {detail.get('reason', 'unknown')} "
                    f"(logs: {_stderr_log_path})"
                )
                return
            try:
                resp = await client.get(
                    f"http://127.0.0.1:{config.MLX_PORT}/health",
                    timeout=2,
                )
                if resp.status_code == 200:
                    _state = ModelState.READY
                    _ready_event.set()
                    _clear_failure()
                    load_ms = (
                        (time.monotonic() - _loading_started_at) * 1000
                        if _loading_started_at
                        else None
                    )

                    # Memory snapshot after model is loaded
                    pid = _process.pid if _process else None
                    snap = metrics.take_memory_snapshot(
                        active_model=_active_model,
                        event="model_loaded",
                        subprocess_pid=pid,
                    )
                    events.emit(
                        events.EventType.MODEL_READY,
                        model=_active_model,
                        detail={
                            "ram_used_gb": snap.ram_used_gb,
                            "subprocess_rss_mb": snap.subprocess_rss_mb,
                            "metal_active_mb": snap.metal_active_mb,
                        },
                        duration_ms=load_ms,
                    )
                    events.emit(
                        events.EventType.MODEL_SWITCH_COMPLETE,
                        model=_active_model,
                        duration_ms=load_ms,
                    )
                    logger.info(
                        f"Model {_active_model} ready in {load_ms:.0f}ms | "
                        f"RAM={snap.ram_used_gb}GB | subprocess_rss={snap.subprocess_rss_mb}MB"
                    )
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(2)

    # Timed out
    _state = ModelState.FAILED
    _ready_event.set()
    detail = _diagnose_failure(timeout_seconds=readiness_timeout)
    load_ms = (time.monotonic() - _loading_started_at) * 1000 if _loading_started_at else None
    _record_failure(_active_model, detail.get("reason", "unknown"))
    events.emit(
        events.EventType.MODEL_FAILED,
        model=_active_model,
        detail=detail,
        duration_ms=load_ms,
    )
    logger.error(
        f"Model {_active_model} timed out after {readiness_timeout}s (logs: {_stderr_log_path})"
    )


async def _switch_model(model_name: str) -> None:
    global _process, _state, _active_model, _stderr_log_path, _stderr_log_handle
    global _loading_started_at

    async with _switch_lock:
        # Another coroutine may have already loaded this model while we waited
        if _active_model == model_name and _state == ModelState.READY:
            return

        prev_model = _active_model

        if prev_model is not None:
            events.emit(
                events.EventType.MODEL_SWITCH_START,
                model=model_name,
                detail={"from_model": prev_model, "to_model": model_name},
            )
            logger.info(f"Switching model: {prev_model} -> {model_name}")

        await _terminate_current()

        model_cfg = config.MODELS[model_name]
        cached = _is_model_cached(model_cfg)
        _ready_event.clear()
        _state = ModelState.DOWNLOADING if not cached else ModelState.LOADING
        _active_model = model_name
        _loading_started_at = time.monotonic()

        if not cached:
            events.emit(
                events.EventType.MODEL_DOWNLOADING,
                model=model_name,
                detail={"hf_path": model_cfg.hf_path},
            )

        _LOG_DIR.mkdir(exist_ok=True)
        _stderr_log_path = _LOG_DIR / f"{model_name}.log"
        _stderr_log_handle = _stderr_log_path.open("w")

        cmd = _build_command(model_cfg)
        logger.info(f"Spawning: {' '.join(cmd)}")

        _process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=_stderr_log_handle,
        )

        events.emit(
            events.EventType.MODEL_LOADING,
            model=model_name,
            detail={"hf_path": model_cfg.hf_path, "type": model_cfg.type, "pid": _process.pid},
        )
        events.emit(
            events.EventType.SUBPROCESS_SPAWNED,
            model=model_name,
            detail={"pid": _process.pid, "command": cmd},
        )

        # Snapshot before model loads to capture baseline memory
        metrics.take_memory_snapshot(active_model=model_name, event="model_loading")

        asyncio.create_task(_health_check_loop())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def ensure_model(model_name: str) -> bool:
    """
    Ensure the named model is READY before the caller proxies a request.
    Returns True if the model was freshly loaded (cold start).
    """
    global _last_request_at

    if _active_model == model_name and _state == ModelState.READY:
        _last_request_at = datetime.now(UTC)
        return False  # already warm

    cooldown = _failure_cooldown_remaining(model_name)
    if cooldown > 0:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail=(
                f"Model {model_name} recently failed ({_last_failure_reason}); "
                f"retry in {cooldown:.0f}s"
            ),
        )

    cold_start = True

    if _active_model != model_name or _state in (ModelState.IDLE, ModelState.FAILED):
        await _switch_model(model_name)

    if _state in (ModelState.LOADING, ModelState.DOWNLOADING):
        model_cfg = config.MODELS[model_name]
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(_ready_event.wait(), timeout=_readiness_timeout(model_cfg) + 5)

    if _state == ModelState.FAILED:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail=f"Model {model_name} failed to load")

    _last_request_at = datetime.now(UTC)
    return cold_start


def set_keep_alive(seconds: int) -> None:
    """Override the inactivity timeout for the current model session."""
    global _inactivity_timeout
    _inactivity_timeout = seconds


async def start_inactivity_watcher() -> None:
    """Background task: unload idle models every 30s check."""
    while True:
        await asyncio.sleep(30)
        if (
            _state == ModelState.READY
            and _last_request_at is not None
            and datetime.now(UTC) - _last_request_at > timedelta(seconds=_inactivity_timeout)
        ):
            idle_secs = int((datetime.now(UTC) - _last_request_at).total_seconds())
            events.emit(
                events.EventType.MODEL_INACTIVITY_UNLOAD,
                model=_active_model,
                detail={"idle_seconds": idle_secs},
            )
            logger.info(f"Model {_active_model} idle for {idle_secs}s, unloading")
            await _terminate_current()


async def unload() -> None:
    """Terminate the active subprocess and reset to IDLE."""
    await _terminate_current()


# ---------------------------------------------------------------------------
# Status accessors
# ---------------------------------------------------------------------------


def get_status() -> dict:
    unload_at = None
    if _last_request_at is not None and _inactivity_timeout > 0 and _state == ModelState.READY:
        unload_at = (_last_request_at + timedelta(seconds=_inactivity_timeout)).isoformat()
    return {
        "active_model": _active_model,
        "state": _state.value,
        "pid": _process.pid if _process else None,
        "last_request_at": _last_request_at.isoformat() if _last_request_at else None,
        "unload_at": unload_at,
        "uptime_seconds": int((datetime.now(UTC) - _started_at).total_seconds()),
        "inactivity_timeout_seconds": _inactivity_timeout,
    }


def get_process_pid() -> int | None:
    """Return the PID of the active subprocess, or None."""
    return _process.pid if _process else None


def get_active_model_name() -> str | None:
    """Return the name of the currently active subprocess model."""
    return _active_model
