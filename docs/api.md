# SatQuery-RS — API Reference

## Base URL

```
http://localhost:8100    # Local development
http://<cloud-gpu>:8100  # Cloud GPU deployment
```

## Authentication

No authentication required for local/private deployment.
For production, configure API keys via environment variables.

---

## Endpoints

### `GET /health`

Check server status and model readiness.

**Response:**
```json
{
    "status": "healthy",
    "model": "SatQuery-RS",
    "loaded": true,
    "mode": "quantized-4bit"
}
```

---

### `POST /vqa`

Visual Question Answering on a satellite image.

**Parameters (multipart/form-data):**

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | ✅ | Satellite/RS image (PNG, JPG, TIFF) |
| `question` | string | ✅ | Natural language question |

**Example:**
```bash
curl -X POST http://localhost:8100/vqa \
    -F "image=@satellite.png" \
    -F "question=What land cover is visible?"
```

**Response:**
```json
{
    "task": "vqa",
    "model": "SatQuery-RS",
    "answer": "The image shows agricultural fields with mixed vegetation and a small water body.",
    "confidence": 0.87,
    "spatial_evidence": [],
    "artifacts": [],
    "metadata": {
        "image": "satellite.png",
        "question": "What land cover is visible?",
        "inference_time_s": 2.3
    },
    "warnings": []
}
```

---

### `POST /caption`

Generate a detailed scene description for a satellite image.

**Parameters (multipart/form-data):**

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | ✅ | Satellite/RS image |
| `instruction` | string | ❌ | Custom captioning prompt (default: "Describe this satellite image in detail.") |

**Example:**
```bash
curl -X POST http://localhost:8100/caption \
    -F "image=@satellite.png"
```

**Response:**
```json
{
    "task": "caption",
    "model": "SatQuery-RS",
    "answer": "An aerial view of a coastal region with sandy beaches, dense urban development near the shore, and agricultural fields extending inland.",
    "confidence": 0.85,
    "spatial_evidence": [],
    "artifacts": [],
    "metadata": {
        "image": "satellite.png",
        "instruction": "Describe this satellite image in detail.",
        "inference_time_s": 3.1
    },
    "warnings": []
}
```

---

### `POST /grounding`

Locate a specific object or feature in a satellite image with bounding boxes.

**Parameters (multipart/form-data):**

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | ✅ | Satellite/RS image |
| `query` | string | ✅ | What to locate (e.g., "water body", "buildings") |

**Example:**
```bash
curl -X POST http://localhost:8100/grounding \
    -F "image=@satellite.png" \
    -F "query=water body"
```

**Response:**
```json
{
    "task": "grounding",
    "model": "SatQuery-RS",
    "answer": "Water body identified in the southeast region of the image.",
    "confidence": 0.89,
    "spatial_evidence": [
        {
            "type": "bbox",
            "coordinates": [0.6, 0.7, 0.9, 0.95],
            "label": "water body"
        }
    ],
    "artifacts": [],
    "metadata": {
        "image": "satellite.png",
        "query": "water body",
        "num_regions": 1,
        "inference_time_s": 2.8
    },
    "warnings": []
}
```

---

### `POST /change-vqa`

Bi-temporal change interpretation using VLM + ChangeFormer.

**Parameters (multipart/form-data):**

| Field | Type | Required | Description |
|---|---|---|---|
| `image_before` | file | ✅ | Before (T1) satellite image |
| `image_after` | file | ✅ | After (T2) satellite image |
| `question` | string | ❌ | Question about the change (default provided) |
| `change_mask` | file | ❌ | Binary change mask from P3's ChangeFormer |

**Example:**
```bash
curl -X POST http://localhost:8100/change-vqa \
    -F "image_before=@before.png" \
    -F "image_after=@after.png" \
    -F "question=What type of development appeared?" \
    -F "change_mask=@mask.png"
```

**Response:**
```json
{
    "task": "change_vqa",
    "model": "SatQuery-RS",
    "answer": "New residential buildings have appeared in the southeast quadrant.",
    "confidence": 0.82,
    "spatial_evidence": [
        {
            "type": "bbox",
            "coordinates": [0.6, 0.7, 0.9, 0.95],
            "label": "change region"
        }
    ],
    "artifacts": ["evaluation/visualizations/change_overlay_after.png"],
    "metadata": {
        "image_before": "before.png",
        "image_after": "after.png",
        "question": "What type of development appeared?",
        "change_percentage": 12.3,
        "change_source": "changeformer",
        "inference_time_s": 3.5
    },
    "warnings": []
}
```

---

## Error Handling

All endpoints return errors in the standard ToolResult format:

```json
{
    "task": "vqa",
    "model": "SatQuery-RS",
    "answer": "",
    "confidence": 0.0,
    "spatial_evidence": [],
    "artifacts": [],
    "metadata": {"error": true},
    "warnings": ["Image file not found: path/to/missing.png"]
}
```

HTTP 500 errors include a `detail` field with the exception message.

---

## Python Client

```python
from backend.client import SatQueryClient

client = SatQueryClient("http://localhost:8100")

# VQA
result = client.vqa("satellite.png", "What land cover is present?")

# Caption
result = client.caption("satellite.png")

# Grounding
result = client.grounding("satellite.png", "water body")

# Change-VQA
result = client.change_vqa(
    "before.png", "after.png",
    question="What changed?",
    change_mask="mask.png"
)
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SATQUERY_MODE` | `cpu-offload` | Inference mode: `quantized-4bit`, `cpu-offload`, `full-fp16` |
| `GEOCHAT_BASE_MODEL` | `MBZUAI/geochat-7B` | Base model HuggingFace ID |
| `LORA_ADAPTER_PATH` | `models/checkpoints/satquery-rs-vlm` | Path to LoRA adapter weights |
| `VISION_TOWER` | `openai/clip-vit-large-patch14-336` | CLIP vision encoder |
| `SATQUERY_SERVER_URL` | `http://localhost:8100` | Server URL for client |
| `HF_TOKEN` | — | HuggingFace API token |
