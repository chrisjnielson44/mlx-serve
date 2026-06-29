"""
model_pool.py — Multi-model concurrent loading for MLX inference.

Keeps multiple models in memory simultaneously so that switching between
them is instantaneous (no reload delay). Models are evicted based on:
  1. Memory pressure (RAM usage threshold)
  2. LRU (least recently used)
  3. Explicit eviction

State machine per model:
  NOT_LOADED -> LOADING -> READY / FAILED -> EVICTING -> NOT_LOADED
"""

import asyncio
import contextlib
import json
import logging
import pathlib
import subprocess
import sys
import time
from datetime import UTC, datetime
from enum import Enum

import httpx
import psutil

from . import config, events

logger = logging.getLogger("mlx-serve.pool")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_pool: dict[str, "PoolModel"] = {}
_load_lock = asyncio.Lock()
_initialized = False
_next_port = config.MLX_PORT + 1  # port allocator for pool models


class PoolModelState(Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class PoolModel:
    """Represents a single model in the pool."""

    def __init__(self, name: str, hf_path: str, model_type: str):
        self.name = name
        self.hf_path = hf_path
        self.model_type = model_type
        self.state = PoolModelState.NOT_LOADED
        self.last_used_at: datetime | None = None
        self.loaded_at: datetime | None = None
        self.subprocess: subprocess.Popen | None = None
        self.pid: int | None = None
        self.port: int | None = None
        self.error: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hf_path": self.hf_path,
            "type": self.model_type,
            "state": self.state.value,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "pid": self.pid,
            "port": self.port,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def initialize() -> None:
    """Initialize the model pool from config."""
    global _pool, _initialized

    if _initialized:
        return

    async with _load_lock:
        if _initialized:
            return

        # Pre-load models marked with keep_in_pool=True
        for name, model_cfg in config.MODELS.items():
            if model_cfg.keep_in_pool and model_cfg.type in ("text", "vision"):
                _pool[name] = PoolModel(name, model_cfg.hf_path, model_cfg.type)
                logger.info(f"Added {name} to model pool (keep_in_pool=True)")

        _initialized = True
        logger.info(
            f"Model pool initialized: {len(_pool)} models in pool, "
            f"max={config.POOL_CONFIG.max_models}"
        )


async def unload_all() -> None:
    """Unload every model from the pool and reset state."""
    global _pool
    async with _load_lock:
        for name in list(_pool.keys()):
            pm = _pool[name]
            if pm.subprocess and pm.subprocess.poll() is None:
                logger.info(f"Unloading pool model {name} (pid={pm.pid})")
                pm.subprocess.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(None, pm.subprocess.wait),
                        timeout=5,
                    )
                except TimeoutError:
                    pm.subprocess.kill()
                    await asyncio.get_running_loop().run_in_executor(None, pm.subprocess.wait)
        _pool.clear()
        _initialized = False
        logger.info("Model pool cleared")


def get_pool_status() -> dict:
    """Return status of all models in the pool."""
    return {
        "models": {name: m.to_dict() for name, m in _pool.items()},
        "total_models": len(_pool),
        "max_models": config.POOL_CONFIG.max_models,
        "memory_threshold": config.POOL_CONFIG.memory_threshold,
        "auto_evict": config.POOL_CONFIG.auto_evict,
    }


def get_pool_model(name: str) -> PoolModel | None:
    """Get a model from the pool by name."""
    return _pool.get(name)


def list_pool_models() -> list[dict]:
    """List all models in the pool."""
    return [m.to_dict() for m in _pool.values()]


# ---------------------------------------------------------------------------
# Load / Evict
# ---------------------------------------------------------------------------


async def load_model(name: str) -> bool:
    """
    Load a model into the pool. Returns True if freshly loaded (cold start).

    If the model is already in the pool and ready, returns False.
    If the model is not in the pool, adds it first (if under max_models limit).
    """
    global _pool

    async with _load_lock:
        # Check if model exists in pool
        if name in _pool:
            pm = _pool[name]
            if pm.state == PoolModelState.READY:
                pm.last_used_at = datetime.now(UTC)
                return False  # Already loaded
            elif pm.state == PoolModelState.LOADING:
                logger.info(f"Waiting for {name} to finish loading...")
                return False

        # Check if we can add more models
        if len(_pool) >= config.POOL_CONFIG.max_models:
            if config.POOL_CONFIG.auto_evict:
                evicted = await _evict_lru()
                if not evicted:
                    raise RuntimeError(
                        f"Model pool full ({config.POOL_CONFIG.max_models} models) "
                        f"and cannot evict."
                    )
            else:
                raise RuntimeError(
                    f"Model pool full ({config.POOL_CONFIG.max_models} models). "
                    f"Set auto_evict=True to enable automatic eviction."
                )

        # Validate model config
        if name not in config.MODELS:
            raise ValueError(f"Model '{name}' not found in config")

        model_cfg = config.MODELS[name]
        if model_cfg.type not in ("text", "vision"):
            raise ValueError(
                f"Model '{name}' is type '{model_cfg.type}', only 'text' and 'vision' "
                f"models can be loaded into the pool"
            )

        pm = PoolModel(name, model_cfg.hf_path, model_cfg.type)
        _pool[name] = pm

    # Spawn subprocess outside lock
    try:
        await _spawn_subprocess(pm, model_cfg)
        pm.state = PoolModelState.READY
        pm.last_used_at = datetime.now(UTC)

        logger.info(f"Model {name} loaded into pool on port {pm.port}")
        events.emit(
            events.EventType.POOL_LOADED,
            model=name,
            detail={"port": pm.port},
        )
        return True

    except Exception as e:
        pm.state = PoolModelState.FAILED
        pm.error = str(e)
        logger.error(f"Failed to load {name} into pool: {e}")
        events.emit(
            events.EventType.MODEL_FAILED,
            model=name,
            detail={"source": "pool", "error": str(e)},
        )
        raise


async def _spawn_subprocess(pm: PoolModel, model_cfg: config.ModelConfig) -> None:
    """Spawn a subprocess for a pool model on a unique port."""
    global _next_port

    is_vision = model_cfg.type == "vision"
    executable = (
        _VENV_BIN / "mlx_vlm.server" if is_vision else _VENV_BIN / "mlx_lm.server"
    )

    if not executable.exists():
        pkg = "mlx-vlm" if is_vision else "mlx-lm"
        raise RuntimeError(f"{pkg} is not installed (expected: {executable})")

    port = _next_port
    _next_port += 1

    cmd = [
        str(executable),
        "--model",
        model_cfg.hf_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    if is_vision:
        if model_cfg.max_kv_cache_size > 0:
            cmd += ["--max-kv-size", str(model_cfg.max_kv_cache_size)]
        if model_cfg.extra_args:
            cmd += [str(arg) for arg in model_cfg.extra_args]
    else:
        if model_cfg.context_length > 0:
            cmd += ["--max-tokens", str(model_cfg.context_length)]
        if model_cfg.chat_template_args:
            cmd += ["--chat-template-args", json.dumps(model_cfg.chat_template_args)]
        if model_cfg.temperature is not None:
            cmd += ["--temp", str(model_cfg.temperature)]
        if model_cfg.top_p is not None:
            cmd += ["--top-p", str(model_cfg.top_p)]
        if model_cfg.top_k is not None:
            cmd += ["--top-k", str(model_cfg.top_k)]
        if model_cfg.min_p is not None:
            cmd += ["--min-p", str(model_cfg.min_p)]
        if model_cfg.extra_args:
            cmd += [str(arg) for arg in model_cfg.extra_args]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    pm.subprocess = proc
    pm.pid = proc.pid
    pm.port = port

    # Wait for readiness
    deadline = time.time() + config.STARTUP_TIMEOUT
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Subprocess exited with code {proc.returncode}")

            try:
                resp = await client.get(
                    f"http://127.0.0.1:{port}/health",
                    timeout=2,
                )
                if resp.status_code == 200:
                    return
            except httpx.RequestError:
                pass

            await asyncio.sleep(1)

    raise TimeoutError(
        f"Model {pm.name} failed to become ready within {config.STARTUP_TIMEOUT}s"
    )


async def _evict_lru() -> bool:
    """Evict the least recently used ready model from the pool."""
    async with _load_lock:
        ready = [m for m in _pool.values() if m.state == PoolModelState.READY]
        if not ready:
            return False

        ready.sort(key=lambda m: m.last_used_at or datetime.min.replace(tzinfo=UTC))
        victim = ready[0]

    logger.info(f"Evicting LRU model: {victim.name} (port {victim.port})")
    await _terminate_subprocess(victim)

    async with _load_lock:
        if victim.name in _pool:
            del _pool[victim.name]

    events.emit(
        events.EventType.POOL_UNLOADED,
        model=victim.name,
        detail={"reason": "lru"},
    )
    return True


async def _terminate_subprocess(pm: PoolModel) -> None:
    """Terminate a pool model's subprocess."""
    if pm.subprocess and pm.subprocess.poll() is None:
        logger.info(f"Terminating subprocess for {pm.name} (pid={pm.pid})")
        pm.subprocess.terminate()
        try:
            await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, pm.subprocess.wait),
                timeout=5,
            )
        except TimeoutError:
            pm.subprocess.kill()
            await asyncio.get_running_loop().run_in_executor(None, pm.subprocess.wait)

    pm.state = PoolModelState.NOT_LOADED
    pm.subprocess = None
    pm.pid = None
    pm.port = None
    pm.error = None


async def unload_model(name: str) -> None:
    """Unload a specific model from the pool."""
    async with _load_lock:
        if name not in _pool:
            return
        pm = _pool[name]
        if pm.state == PoolModelState.NOT_LOADED:
            return
        logger.info(f"Unloading model {name} from pool...")
        await _terminate_subprocess(pm)
        del _pool[name]

    events.emit(
        events.EventType.POOL_UNLOADED,
        model=name,
        detail={"reason": "manual"},
    )


def get_pool_model_port(name: str) -> int | None:
    """Get the port for a pool model (for proxying requests)."""
    pm = _pool.get(name)
    if pm and pm.state == PoolModelState.READY:
        return pm.port
    return None


# ---------------------------------------------------------------------------
# Memory pressure
# ---------------------------------------------------------------------------


async def check_memory_pressure() -> None:
    """Check memory pressure and evict models if needed."""
    if not config.POOL_CONFIG.auto_evict or not _pool:
        return

    vm = psutil.virtual_memory()
    if vm.percent < config.POOL_CONFIG.memory_threshold * 100:
        return

    logger.warning(
        f"Memory pressure: {vm.percent}% used. Evicting LRU model..."
    )
    await _evict_lru()


async def start_memory_watcher() -> None:
    """Background task: check memory pressure every 10 seconds."""
    while True:
        await asyncio.sleep(10)
        try:
            await check_memory_pressure()
        except Exception:
            pass


# Resolve executable paths relative to the running venv so subprocesses
# inherit the same environment that has mlx_lm / mlx_vlm installed.
_VENV_BIN = pathlib.Path(sys.executable).parent
