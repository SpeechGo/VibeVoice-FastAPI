# VibeVoice FastAPI Service

FastAPI implementation of Microsoft VibeVoice (text-to-speech) with REST and WebSocket APIs. The service loads a VibeVoice model on startup and exposes endpoints for TTS generation, health/readiness, and Prometheus metrics.

## Overview

- **Framework:** FastAPI + Uvicorn
- **Model:** `microsoft/VibeVoice-1.5B` (default) or `microsoft/VibeVoice-Large`
- **APIs:** REST `/api/generate`, `/api/voices`; WebSocket `/ws/generate`
- **Health/Monitoring:** `/healthz`, `/readyz`, `/metrics`
- **Static UI:** Basic demo served from `/` if `static/index.html` exists

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (configured for `cuda:0`)
- PyTorch with CUDA support; FlashAttention 2 optional (falls back to SDPA automatically)

## Setup

Using uv (recommended):

```bash
# Install dependencies (dev optional)
uv pip install -e .[dev]

# Or, with lockfile sync
uv sync
```

Using pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

Environment variables (can be placed in `.env`):

- `MODEL_PATH`: `microsoft/VibeVoice-1.5B` (default) or `microsoft/VibeVoice-Large`
- `VOICES_DIR`: path to voice samples directory (default: `voices/`)
- `MAX_CONCURRENCY`: concurrent generations (default: `1`)
- `TIMEOUT_SEC`: per-request timeout seconds (default: `300`)
- `CORS_ALLOW_ORIGINS`: comma-separated origins (default: empty)
- `LOG_LEVEL`: `debug|info|warning|error|critical` (default: `info`)

## Run (Local)

```bash
# With uv
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Or with python environment
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open interactive docs at `http://localhost:8000/docs`.

Model downloads and initialization happen on first start; a CUDA-enabled GPU is required.

## Run (Docker)

Requires the NVIDIA Container Toolkit.

```bash
docker compose up --build
# or
docker run --gpus all -p 8000:8000 --env-file .env \
  -v $(pwd)/voices:/app/voices \
  vibevoice-fastapi:latest
```

## API

- `GET /healthz`: basic liveness probe
- `GET /readyz`: readiness + model/device info
- `GET /api/voices`: list available voices (scanned from `VOICES_DIR`)
- `POST /api/generate`: generate TTS
  - Content negotiation via `Accept`:
    - `audio/wav` returns binary WAV
    - `application/json` returns JSON with `audio_base64`
- `GET /metrics`: Prometheus exposition format
- `WS /ws/generate`: streaming PCM16 frames with progress/final messages

Example: request a WAV directly and save to file

```bash
curl -X POST http://localhost:8000/api/generate \
  -H 'Content-Type: application/json' \
  -H 'Accept: audio/wav' \
  -d '{
        "script": "Alice: Hello, this is a test.",
        "speakers": ["en-Alice_woman"],
        "cfg_scale": 1.3,
        "inference_steps": 5
      }' \
  --output out.wav
```

Example: request JSON output with base64-encoded audio

```bash
curl -X POST http://localhost:8000/api/generate \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -d '{
        "script": "Alice: Hello, this is a test.",
        "speakers": ["en-Alice_woman"],
        "cfg_scale": 1.3,
        "inference_steps": 5
      }'
```

### Voices

- Put reference audio files in `voices/` (default) and use the stem as the speaker ID.
- Naming like `en-Alice_woman.wav` is parsed into language/name/gender for `/api/voices`.
- Discover available voices:

```bash
curl http://localhost:8000/api/voices
```

## Testing

```bash
pytest tests/
# or
uv run pytest -q
```

## Notes

- This service is a FastAPI implementation of the Microsoft VibeVoice model. It requires a CUDA-capable GPU and downloads models from Hugging Face on first use.
- If FlashAttention 2 is not available, the service automatically falls back to SDPA attention.
