"""
config.py — reads models.yaml and exposes typed configuration.

Config discovery order:
  1. MLX_SERVE_CONFIG env var (explicit override)
  2. ./models.yaml  (current working directory)
  3. ~/.mlx-serve/models.yaml  (user config directory)
  4. Bundled _default_models.yaml inside the package
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _find_config() -> Path:
    """Locate models.yaml using a fallback chain."""
    # 1. Explicit env var
    env_path = os.environ.get("MLX_SERVE_CONFIG")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"MLX_SERVE_CONFIG points to {p} which does not exist")

    # 2. Current working directory
    cwd_path = Path.cwd() / "models.yaml"
    if cwd_path.exists():
        return cwd_path

    # 3. User config directory
    user_path = Path.home() / ".mlx-serve" / "models.yaml"
    if user_path.exists():
        return user_path

    # 4. Bundled default
    bundled = Path(__file__).parent / "_default_models.yaml"
    if bundled.exists():
        return bundled

    raise FileNotFoundError(
        "No models.yaml found. Searched:\n"
        f"  - {cwd_path}\n"
        f"  - {user_path}\n"
        "Run 'mlx-serve init' to generate one."
    )


_CONFIG_PATH = _find_config()


@dataclass
class ModelConfig:
    name: str
    type: str  # "text", "vision", "embedding", "tts", or "stt"
    hf_path: str
    context_length: int = 0  # max output tokens per response (--max-tokens); 0 = server default
    max_kv_cache_size: int = (
        0  # KV cache token capacity for prompt caching (--max-kv-cache-size); 0 = model default
    )
    chat_template_args: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    prompt_cache_size: int | None = None
    extra_args: list[str] | None = None
    image_command: str | None = None
    base_model: str | None = None
    steps: int | None = None
    guidance: float | None = None
    quantize: int | None = None
    # Tool use: "auto" = client decides, "server" = mlx-serve executes tools
    tool_use: str | None = None  # "auto" | "server" | None
    # Multi-model pool: keep this model loaded alongside others
    keep_in_pool: bool = False  # if True, model stays loaded in model pool


@dataclass
class MonitoringConfig:
    log_dir: Path
    metrics_history_size: int = 500
    events_history_size: int = 1000
    memory_sample_interval: int = 10  # seconds, in-memory
    memory_log_interval: int = 60  # seconds, to disk
    log_retention_mb: int = 50  # per JSONL file


@dataclass
class ToolUseConfig:
    max_iterations: int = 10  # max agent loop iterations
    timeout_seconds: int = 60  # timeout per iteration


@dataclass
class ModelPoolConfig:
    max_models: int = 3  # max concurrent models in pool
    memory_threshold: float = 0.85  # unload lowest priority when RAM > this %
    auto_evict: bool = True  # automatically evict models when memory is tight


_VALID_TYPES = {"text", "vision", "embedding", "image", "tts", "stt"}


def _load() -> tuple[
    dict[str, ModelConfig],
    int,
    int,
    int,
    int,
    int,
    int,
    MonitoringConfig,
    ToolUseConfig,
    ModelPoolConfig,
]:
    with _CONFIG_PATH.open() as f:
        data = yaml.safe_load(f)

    models = {}
    for entry in data.get("models", []):
        if entry["type"] not in _VALID_TYPES:
            raise ValueError(
                f"Model '{entry['name']}' has invalid type '{entry['type']}'. "
                f"Must be one of: {sorted(_VALID_TYPES)}"
            )
        models[entry["name"]] = ModelConfig(
            name=entry["name"],
            type=entry["type"],
            hf_path=entry["hf_path"],
            context_length=entry.get("context_length", 0),
            max_kv_cache_size=entry.get("max_kv_cache_size", 0),
            chat_template_args=entry.get("chat_template_args"),
            temperature=entry.get("temperature"),
            top_p=entry.get("top_p"),
            top_k=entry.get("top_k"),
            min_p=entry.get("min_p"),
            prompt_cache_size=entry.get("prompt_cache_size"),
            extra_args=entry.get("extra_args"),
            image_command=entry.get("image_command"),
            base_model=entry.get("base_model"),
            steps=entry.get("steps"),
            guidance=entry.get("guidance"),
            quantize=entry.get("quantize"),
            tool_use=entry.get("tool_use"),
            keep_in_pool=entry.get("keep_in_pool", False),
        )

    # Monitoring settings (optional section in models.yaml)
    mon_raw = data.get("monitoring", {})
    default_log_dir = Path.home() / ".mlx-serve" / "logs"
    log_dir_str = mon_raw.get("log_dir", str(default_log_dir))
    monitoring = MonitoringConfig(
        log_dir=Path(log_dir_str).expanduser(),
        metrics_history_size=mon_raw.get("metrics_history_size", 500),
        events_history_size=mon_raw.get("events_history_size", 1000),
        memory_sample_interval=mon_raw.get("memory_sample_interval", 10),
        memory_log_interval=mon_raw.get("memory_log_interval", 60),
        log_retention_mb=mon_raw.get("log_retention_mb", 50),
    )

    # Tool use config (optional section in models.yaml)
    tool_raw = data.get("tool_use", {})
    tool_config = ToolUseConfig(
        max_iterations=tool_raw.get("max_iterations", 10),
        timeout_seconds=tool_raw.get("timeout_seconds", 60),
    )

    # Model pool config (optional section in models.yaml)
    pool_raw = data.get("model_pool", {})
    pool_config = ModelPoolConfig(
        max_models=pool_raw.get("max_models", 3),
        memory_threshold=pool_raw.get("memory_threshold", 0.85),
        auto_evict=pool_raw.get("auto_evict", True),
    )

    return (
        models,
        data.get("mlx_port", 8091),
        data.get("manager_port", 8095),
        data.get("inactivity_timeout_seconds", 600),
        data.get("startup_timeout_seconds", 120),
        data.get("download_timeout_seconds", 1800),
        data.get("failure_cooldown_seconds", 30),
        monitoring,
        tool_config,
        pool_config,
    )


(
    MODELS,
    MLX_PORT,
    MANAGER_PORT,
    INACTIVITY_TIMEOUT,
    STARTUP_TIMEOUT,
    DOWNLOAD_TIMEOUT,
    FAILURE_COOLDOWN,
    MONITORING,
    TOOL_CONFIG,
    POOL_CONFIG,
) = _load()

# Optional bearer token auth. Set MLX_API_KEY env var to enable.
# When empty, all requests are accepted (safe for localhost-only deployments).
API_KEY: str = os.environ.get("MLX_API_KEY", "")
