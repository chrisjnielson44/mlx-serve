"""
mcp_server.py — Expose MLX models as MCP (Model Context Protocol) tools/resources.

This module creates an MCP server that exposes:
  - Tools: Chat completions, embeddings, TTS, STT, image generation
  - Resources: Model info, status, metrics
  - Prompts: Pre-built prompt templates

Uses the official `mcp` Python SDK from modelcontextprotocol/python-sdk.
"""

import json
import logging
from typing import Any

import psutil

from . import config, inline_manager, metrics, model_pool, process_manager

logger = logging.getLogger("mlx-serve.mcp")

# Try to import mcp, provide helpful error if not installed
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning(
        "mcp package not installed. Install with: pip install mlx-serve[mcp]"
    )


def create_mcp_server(host: str = "0.0.0.0", port: int = 8096) -> Any:
    """
    Create and configure the MCP server.

    Args:
        host: Bind address for the MCP server.
        port: Port for the MCP server.

    Returns:
        FastMCP instance configured with MLX tools and resources.
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP server requires the 'mcp' package. Install with:\n"
            "  pip install mlx-serve[mcp]\n\n"
            "Or install the mcp Python SDK:\n"
            "  pip install mcp"
        )

    mcp = FastMCP(
        name="MLX Serve",
        instructions=(
            "You are an MLX inference server running on Apple Silicon. "
            "You can run chat completions, generate embeddings, create images, "
            "synthesize speech, and transcribe audio using locally loaded MLX models."
        ),
        host=host,
        port=port,
    )

    # Register tools
    _register_chat_tool(mcp)
    _register_embeddings_tool(mcp)
    _register_image_tool(mcp)
    _register_tts_tool(mcp)
    _register_stt_tool(mcp)

    # Register resources
    _register_model_resource(mcp)
    _register_status_resource(mcp)
    _register_metrics_resource(mcp)

    return mcp


def _register_chat_tool(mcp: Any) -> None:
    """Register the chat completions tool."""

    @mcp.tool()
    async def chat_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """
        Run a chat completion using an MLX model.

        Args:
            model: Model name (e.g., 'mlx-qwen2.5-7b')
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response (default: False)
            temperature: Sampling temperature (optional)
            max_tokens: Maximum tokens to generate (optional)
            tools: Tool definitions for function calling (optional)

        Returns:
            JSON string with the completion response
        """
        logger.info(f"MCP chat_completion: model={model}, messages={len(messages)}")

        # Ensure model is loaded
        if model not in config.MODELS:
            return json.dumps({"error": f"Model '{model}' not found in config"})

        model_cfg = config.MODELS[model]
        if model_cfg.type not in ("text", "vision"):
            return json.dumps({"error": f"Model '{model}' is type '{model_cfg.type}', not suitable for chat"})

        # Build request body
        body = {
            "model": model_cfg.hf_path,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools

        # Proxy to mlx_lm
        from .router import _HTTP_CLIENT

        target = f"http://127.0.0.1:{config.MLX_PORT}/v1/chat/completions"

        try:
            resp = await _HTTP_CLIENT.post(target, json=body, timeout=300)
            if resp.status_code != 200:
                return json.dumps({"error": f"mlx_lm returned {resp.status_code}"})

            response_data = resp.json()

            # Extract content from response
            choices = response_data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                tool_calls = message.get("tool_calls")

                result = {"content": content}
                if tool_calls:
                    result["tool_calls"] = tool_calls

                return json.dumps(result)

            return json.dumps({"error": "Empty response from mlx_lm"})

        except Exception as e:
            return json.dumps({"error": f"Request failed: {str(e)}"})


def _register_embeddings_tool(mcp: Any) -> None:
    """Register the embeddings tool."""

    @mcp.tool()
    async def generate_embeddings(
        model: str,
        texts: list[str],
    ) -> str:
        """
        Generate embeddings for a list of texts.

        Args:
            model: Embedding model name (e.g., 'mlx-qwen3-embedding')
            texts: List of text strings to embed

        Returns:
            JSON string with embedding vectors
        """
        logger.info(f"MCP generate_embeddings: model={model}, texts={len(texts)}")

        if model not in config.MODELS or config.MODELS[model].type != "embedding":
            return json.dumps({"error": f"Model '{model}' not found or not an embedding model"})

        inputs = [{"text": text} for text in texts]

        try:
            from .inline_manager import generate_embeddings as inline_embeddings

            vectors = await inline_embeddings(model, inputs)

            result = {
                "model": model,
                "embeddings": [
                    {"index": i, "embedding": vec}
                    for i, vec in enumerate(vectors)
                ],
            }
            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": f"Embedding failed: {str(e)}"})


def _register_image_tool(mcp: Any) -> None:
    """Register the image generation tool."""

    @mcp.tool()
    async def generate_image(
        model: str,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
    ) -> str:
        """
        Generate an image from a text prompt.

        Args:
            model: Image model name (e.g., 'mlx-flux2-klein-4b')
            prompt: Text description of the image
            width: Image width in pixels (default: 1024)
            height: Image height in pixels (default: 1024)

        Returns:
            JSON string with image path and base64 data
        """
        logger.info(f"MCP generate_image: model={model}, prompt={prompt[:50]}...")

        if model not in config.MODELS or config.MODELS[model].type != "image":
            return json.dumps({"error": f"Model '{model}' not found or not an image model"})

        try:
            from .router import _run_image_generation

            image_path = await _run_image_generation(
                model, prompt, width=width, height=height
            )

            import base64

            b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

            result = {
                "path": str(image_path),
                "b64_json": b64,
                "prompt": prompt,
                "size": f"{width}x{height}",
            }
            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": f"Image generation failed: {str(e)}"})


def _register_tts_tool(mcp: Any) -> None:
    """Register the TTS tool."""

    @mcp.tool()
    async def text_to_speech(
        model: str,
        text: str,
        speed: float = 1.0,
        language: str = "en",
    ) -> str:
        """
        Convert text to speech.

        Args:
            model: TTS model name (e.g., 'mlx-chatterbox')
            text: Text to synthesize
            speed: Speech speed multiplier (default: 1.0)
            language: Language code (default: 'en')

        Returns:
            Base64-encoded WAV audio data
        """
        logger.info(f"MCP text_to_speech: model={model}, chars={len(text)}")

        if model not in config.MODELS or config.MODELS[model].type != "tts":
            return json.dumps({"error": f"Model '{model}' not found or not a TTS model"})

        try:
            from .inline_manager import generate_tts

            wav_bytes = await generate_tts(model, text, speed, language)

            import base64

            b64 = base64.b64encode(wav_bytes).decode("ascii")

            result = {
                "audio_b64": b64,
                "format": "wav",
                "sample_rate": 24000,
                "text": text,
            }
            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": f"TTS failed: {str(e)}"})


def _register_stt_tool(mcp: Any) -> None:
    """Register the STT tool."""

    @mcp.tool()
    async def speech_to_text(
        model: str,
        audio_b64: str,
        language: str | None = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            model: STT model name (e.g., 'mlx-whisper-large')
            audio_b64: Base64-encoded audio data
            language: Language code hint (optional)

        Returns:
            JSON string with transcription text
        """
        logger.info(f"MCP speech_to_text: model={model}")

        if model not in config.MODELS or config.MODELS[model].type != "stt":
            return json.dumps({"error": f"Model '{model}' not found or not an STT model"})

        try:
            import base64

            audio_bytes = base64.b64decode(audio_b64)

            from .inline_manager import generate_stt

            text = await generate_stt(model, audio_bytes, language)

            result = {"text": text, "model": model}
            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": f"STT failed: {str(e)}"})


def _register_model_resource(mcp: Any) -> None:
    """Register the model info resource."""

    @mcp.resource("mlx://models")
    def get_models() -> str:
        """List all configured MLX models with their capabilities."""
        models_data = []
        for name, cfg in config.MODELS.items():
            models_data.append({
                "name": name,
                "hf_path": cfg.hf_path,
                "type": cfg.type,
                "capabilities": _get_capabilities(cfg.type),
            })

        return json.dumps(
            {"object": "list", "data": models_data},
            indent=2,
        )

    @mcp.resource("mlx://models/{model_name}")
    def get_model_info(model_name: str) -> str:
        """Get detailed info about a specific model."""
        if model_name not in config.MODELS:
            return json.dumps({"error": f"Model '{model_name}' not found"})

        cfg = config.MODELS[model_name]
        info = {
            "name": cfg.name,
            "hf_path": cfg.hf_path,
            "type": cfg.type,
            "capabilities": _get_capabilities(cfg.type),
            "context_length": cfg.context_length,
            "keep_in_pool": cfg.keep_in_pool,
        }
        return json.dumps(info, indent=2)


def _register_status_resource(mcp: Any) -> None:
    """Register the status resource."""

    @mcp.resource("mlx://status")
    def get_status() -> str:
        """Get current server status including model states and memory usage."""
        status = {
            "subprocess": process_manager.get_status(),
            "inline": inline_manager.get_status(),
            "pool": model_pool.get_pool_status(),
            "memory": _get_memory_stats(),
        }
        return json.dumps(status, indent=2)


def _register_metrics_resource(mcp: Any) -> None:
    """Register the metrics resource."""

    @mcp.resource("mlx://metrics")
    def get_metrics() -> str:
        """Get server metrics including request history and memory usage."""
        metal = metrics.get_metal_memory()
        vm = psutil.virtual_memory()

        metrics_data = {
            "uptime_seconds": process_manager.get_status()["uptime_seconds"],
            "models": metrics.get_aggregates(),
            "memory": {
                "ram_total_gb": round(vm.total / 1e9, 1),
                "ram_used_gb": round(vm.used / 1e9, 1),
                "ram_available_gb": round(vm.available / 1e9, 1),
                "ram_percent": vm.percent,
                "metal_active_mb": metal.get("active_mb"),
                "metal_peak_mb": metal.get("peak_mb"),
            },
            "pressure": metrics.check_memory_pressure(),
        }
        return json.dumps(metrics_data, indent=2)


def _get_capabilities(model_type: str) -> list[str]:
    """Get OpenAI-style capabilities for a model type."""
    capabilities_map = {
        "text": ["completion"],
        "vision": ["completion", "vision"],
        "embedding": ["embedding"],
        "image": ["image_generation"],
        "tts": ["audio_speech"],
        "stt": ["audio_transcription"],
    }
    return capabilities_map.get(model_type, [])


def _get_memory_stats() -> dict:
    """Return unified memory stats."""
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / 1_000_000_000, 1),
        "used_gb": round(vm.used / 1_000_000_000, 1),
        "available_gb": round(vm.available / 1_000_000_000, 1),
        "percent_used": vm.percent,
    }
