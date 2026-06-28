# Configuration Reference

All configuration lives in `models.yaml` at the project root. The file is read once at startup. To apply changes, restart the manager with `make mlx-start`.

---

## File Structure

```yaml
# Top-level runtime settings
mlx_port: 8091
manager_port: 8095
inactivity_timeout_seconds: 600
startup_timeout_seconds: 120

models:
  - name: <model-name>
    type: <text|vision|embedding|tts>
    hf_path: <huggingface-repo-id>
    context_length: 32768
    chat_template_args: {}
    temperature: 0.6
    top_p: 0.9
```

---

## Top-Level Settings

| Field | Default | Description |
|-------|---------|-------------|
| `mlx_port` | `8091` | Internal port used by the active subprocess model. Only one subprocess runs at a time; this port is always the same. Loopback only (`127.0.0.1`). |
| `manager_port` | `8095` | Port the FastAPI manager listens on. Clients (LiteLLM, curl) connect here. |
| `inactivity_timeout_seconds` | `600` | Seconds of no requests before an idle model is unloaded. Frees unified memory. |
| `startup_timeout_seconds` | `120` | Maximum seconds to wait for a subprocess model to become ready. If exceeded, the request returns 503 and the next request retries the load. |

---

## Model Fields

### `name` (required)

The identifier clients use in the `"model"` field of API requests. Must be unique within the file.

```yaml
name: mlx-qwen2.5-7b
```

This is what LiteLLM (or any OpenAI SDK) sends:
```json
{"model": "mlx-qwen2.5-7b", "messages": [...]}
```

### `type` (required)

Controls how the model is loaded and which inference path is used.

| Value | Execution | Inference path |
|-------|-----------|----------------|
| `text` | subprocess — `mlx_lm.server` | `/v1/chat/completions` proxy |
| `vision` | subprocess — `mlx_vlm.server` | `/v1/chat/completions` proxy |
| `embedding` | in-process — `mlx-embeddings` | `/v1/embeddings` |
| `tts` | in-process — `mlx-audio` | `/v1/audio/speech` |
| `stt` | in-process — `mlx-whisper` | `/v1/audio/transcriptions` |

Invalid types are rejected at startup with a clear error message.

### `hf_path` (required)

The HuggingFace repository ID used to load the model. For `text` and `vision` types, this is passed as `--model` to the subprocess. For `embedding` and `tts`, it is passed to the library's `load()` function.

```yaml
hf_path: mlx-community/Qwen2.5-7B-Instruct-4bit
```

The model must already be downloaded to the local HuggingFace cache before the manager can load it. Models are **not** downloaded automatically on first request.

---

## Model Type Examples

### Text model

Spawns `mlx_lm.server`. Used for standard chat and instruction-following.

```yaml
- name: mlx-qwen2.5-7b
  type: text
  hf_path: mlx-community/Qwen2.5-7B-Instruct-4bit
```

### Vision model

Spawns `mlx_vlm.server`. Accepts multimodal messages with images.

```yaml
- name: mlx-qwen3vl-8b
  type: vision
  hf_path: mlx-community/Qwen3-VL-8B-Instruct-8bit
```

### Embedding model

Loaded in-process via `mlx-embeddings`. Accepts text or multimodal inputs.

```yaml
- name: mlx-qwen3-embedding
  type: embedding
  hf_path: Qwen/Qwen3-Embedding-0.6B
```

### TTS model

Loaded in-process via `mlx-audio`. Returns 16-bit PCM WAV audio at 24 kHz mono.

```yaml
- name: mlx-chatterbox
  type: tts
  hf_path: mlx-community/chatterbox-fp16
```

Download before first use:
```bash
make download-tts
```

### STT model

Loaded in-process via `mlx-whisper`. Accepts any audio format (WAV, MP3, M4A, FLAC, etc.) and returns transcribed text.

```yaml
- name: mlx-whisper-turbo
  type: stt
  hf_path: mlx-community/whisper-large-v3-turbo
```

Download before first use:
```bash
make download-stt
```

### `context_length` (optional)

Maximum output tokens per response, passed to the subprocess as `--max-tokens`. Defaults to `0` (uses the server's built-in default of 512 tokens).

```yaml
- name: mlx-qwen2.5-7b
  type: text
  hf_path: mlx-community/Qwen2.5-7B-Instruct-4bit
  context_length: 32768    # allow responses up to 32768 tokens
```

Only applies to `text` and `vision` model types. Ignored for `embedding`, `tts`, and `stt`.

### `max_kv_cache_size` (optional)

KV cache token capacity for prompt caching, passed to the subprocess as `--max-kv-cache-size`. Controls how many tokens of KV state (attention key/value pairs) are retained in memory between requests. When a cached prefix is reused across requests, the model skips recomputing those tokens — reducing latency and improving throughput.

Defaults to `0` (no limit — the model retains as much KV state as fits in unified memory).

```yaml
- name: mlx-qwen2.5-7b
  type: text
  hf_path: mlx-community/Qwen2.5-7B-Instruct-4bit
  context_length: 32768
  max_kv_cache_size: 8192    # cap KV cache at 8192 tokens to bound memory usage
```

Only applies to `text` and `vision` model types. Setting a value limits the memory footprint of the KV cache — useful when running large models close to the unified memory limit.

### `chat_template_args` (optional)

JSON-serializable dictionary passed to the subprocess as `--chat-template-args`.
This is useful for models whose tokenizer chat template supports feature flags.

```yaml
- name: mlx-ornith
  type: text
  hf_path: mlx-community/Ornith-1.0-35B-bf16
  context_length: 8192
  chat_template_args:
    enable_thinking: false
```

This becomes:

```bash
mlx_lm.server ... --chat-template-args '{"enable_thinking": false}'
```

Only applies to `text` and `vision` model types.

### Sampling defaults (optional)

Per-model sampling defaults can be passed to `mlx_lm.server` / `mlx_vlm.server`:

| Field | Subprocess flag |
|-------|-----------------|
| `temperature` | `--temp` |
| `top_p` | `--top-p` |
| `top_k` | `--top-k` |
| `min_p` | `--min-p` |

```yaml
- name: mlx-qwen-coder
  type: text
  hf_path: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
  context_length: 8192
  temperature: 0.4
  top_p: 0.9
```

Only applies to `text` and `vision` model types.

### `prompt_cache_size` (optional)

Prompt cache count, passed to the subprocess as `--prompt-cache-size`.

```yaml
- name: mlx-qwen-coder
  type: text
  hf_path: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
  prompt_cache_size: 3
```

Only applies to `text` and `vision` model types.

### `extra_args` (optional)

Raw subprocess CLI arguments appended after the managed options. Use this for
new `mlx_lm.server` / `mlx_vlm.server` flags before the manager has first-class
fields for them.

```yaml
- name: mlx-qwen-coder
  type: text
  hf_path: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
  extra_args:
    - --log-level
    - DEBUG
```

Only applies to `text` and `vision` model types. Prefer first-class fields when
available so config stays portable across server versions.

### API key authentication (environment variable)

Set `MLX_API_KEY` in the environment to require bearer token auth on all `/v1/*` endpoints:

```bash
export MLX_API_KEY=my-secret-key
make mlx-start
```

When the variable is unset or empty, auth is disabled — all requests are accepted. This is the default and is appropriate for localhost-only deployments.

---

## Validation

The following are validated at startup. The service will not start if any check fails.

| Check | Error |
|-------|-------|
| `models.yaml` file exists | `FileNotFoundError` |
| Each model has a valid `type` | `ValueError: Model '...' has invalid type '...'. Must be one of: [...]` |
| Required fields present (`name`, `type`, `hf_path`) | `KeyError` |

---

## Log Files

When a subprocess model fails to load, its stderr is captured and written to:

```
/tmp/mlx-manager-logs/<model-name>.log
```

The path is printed to the manager's stdout on failure:

```
[mlx-manager] Model failed to load. Logs: /tmp/mlx-manager-logs/mlx-qwen2.5-7b.log
```

Each model gets its own log file, overwritten on each load attempt.
