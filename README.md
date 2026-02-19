# 🤖 AI Face Recognition & Face Swap

> A production-grade, modular AI pipeline for real-time **face detection**, **face recognition**, and **face swapping** — powered by **YOLOv8**, **InsightFace**, and **inswapper_128**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Development](#development)
- [Docker](#docker)
- [Roadmap](#roadmap)
- [Ethics & Safety](#ethics--safety)
- [License](#license)

---

## Overview

This project provides a complete, end-to-end AI pipeline that can:

1. **Detect** faces in images and videos using **YOLOv8** (real-time, GPU-accelerated)
2. **Recognize** and identify faces using **InsightFace ArcFace** embeddings
3. **Swap** faces between images/videos using the **inswapper_128** ONNX model
4. **Enhance** swapped faces using **GFPGAN** or **CodeFormer** for photorealistic results
5. Expose everything via a **FastAPI REST backend** and a **Streamlit web UI**

---

## Features

| Feature | Status |
|---|---|
| ⚡ YOLOv8 real-time face detection | ✅ Ready |
| 🧠 InsightFace ArcFace recognition + embeddings | ✅ Ready |
| 🔄 inswapper_128 face swap engine | ✅ Ready |
| ✨ GFPGAN face enhancement / restoration | ✅ Ready |
| 🎬 Image & Video processing | ✅ Ready |
| 📷 Webcam / live stream support | ✅ Ready |
| 🌐 FastAPI REST backend | ✅ Ready |
| 🖥️ Streamlit web UI | ✅ Ready |
| 🐳 Docker + docker-compose | ✅ Ready |
| 🖥️ GPU (CUDA) / CPU auto-selection | ✅ Ready |
| 🧪 Unit + Integration tests | ✅ Ready |
| 🔒 Ethics gate + output watermarking | ✅ Ready |
| 📦 Model auto-downloader | ✅ Ready |

---

## Architecture

```
Input Image / Video / Webcam
           │
           ▼
┌─────────────────────────┐
│  [1] YOLOv8 Detector    │  → Bounding boxes + confidence scores
│      yolov8n-face.pt    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  [2] InsightFace        │  → 512-dim ArcFace embeddings
│      Analyser           │  → 5-point facial landmarks
│      buffalo_l          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  [3] Face Recognizer    │  → Identity match (cosine similarity)
│      FaceDatabase       │  → identity name + score
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  [4] inswapper_128      │  → Source face identity injected
│      Face Swap Engine   │    into target frame (ONNX)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  [5] GFPGAN / CodeFormer│  → Artifact removal + upscaling
│      Face Enhancer      │  → Photorealistic output
└────────────┬────────────┘
             │
             ▼
     Output Image / Video
```

### Component Breakdown

| Component | File | Responsibility |
|---|---|---|
| Face Detector | `core/detector/yolo_detector.py` | YOLOv8 bounding box detection |
| Face Analyser | `core/recognizer/insightface_recognizer.py` | Embedding extraction + landmarks |
| Face Database | `core/recognizer/face_database.py` | Identity store, cosine search |
| Face Swapper | `core/swapper/inswapper.py` | ONNX face swap inference |
| Face Enhancer | `core/enhancer/gfpgan_enhancer.py` | GFPGAN restoration |
| Pipeline | `core/pipeline/face_pipeline.py` | Orchestrates all steps |
| Video Pipeline | `core/pipeline/video_pipeline.py` | Frame-by-frame video processing |
| REST API | `api/main.py` | FastAPI endpoints |
| Web UI | `ui/app.py` | Streamlit interface |

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Face Detection | [YOLOv8](https://github.com/ultralytics/ultralytics) | ≥ 8.2 |
| Face Analysis | [InsightFace](https://github.com/deepinsight/insightface) | ≥ 0.7.3 |
| Face Swap | inswapper_128.onnx | — |
| Face Enhancement | [GFPGAN](https://github.com/TencentARC/GFPGAN) | ≥ 1.3.8 |
| ONNX Runtime | [onnxruntime-gpu](https://onnxruntime.ai/) | ≥ 1.18 |
| Deep Learning | [PyTorch](https://pytorch.org/) | ≥ 2.2 |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) | ≥ 0.111 |
| Frontend UI | [Streamlit](https://streamlit.io/) | ≥ 1.35 |
| Image Processing | [OpenCV](https://opencv.org/) | ≥ 4.9 |
| Configuration | [Pydantic](https://docs.pydantic.dev/) | ≥ 2.7 |
| Logging | [Loguru](https://github.com/Delgan/loguru) | ≥ 0.7 |
| Packaging | Docker + docker-compose | — |

---

## Project Structure

```
ai-face-recognition/
│
├── core/                          # Core AI engine (framework-agnostic)
│   ├── detector/                  # YOLOv8 face detection
│   │   ├── base_detector.py       # Abstract base class
│   │   └── yolo_detector.py       # YOLOv8 implementation
│   │
│   ├── recognizer/                # Face recognition
│   │   ├── base_recognizer.py     # Abstract base class
│   │   ├── insightface_recognizer.py  # ArcFace embedding extraction
│   │   └── face_database.py       # Face identity store
│   │
│   ├── swapper/                   # Face swap engine
│   │   ├── base_swapper.py        # Abstract base class
│   │   └── inswapper.py           # inswapper_128.onnx wrapper
│   │
│   ├── enhancer/                  # Post-swap face enhancement
│   │   ├── base_enhancer.py       # Abstract base class
│   │   ├── gfpgan_enhancer.py     # GFPGAN restorer
│   │   └── codeformer_enhancer.py # CodeFormer alternative
│   │
│   └── pipeline/                  # Orchestration layer
│       ├── face_pipeline.py       # Image pipeline
│       └── video_pipeline.py      # Video pipeline
│
├── api/                           # FastAPI REST backend
│   ├── main.py                    # App entry point
│   ├── routers/
│   │   ├── health.py              # GET  /api/v1/health
│   │   ├── recognition.py         # POST /api/v1/recognize
│   │   └── swap.py                # POST /api/v1/swap
│   ├── schemas/
│   │   ├── requests.py            # Pydantic request models
│   │   └── responses.py           # Pydantic response models
│   └── middleware/
│       └── cors.py                # CORS + rate limiting
│
├── ui/                            # Streamlit web frontend
│   ├── app.py                     # Main app entry
│   └── pages/
│       ├── face_recognition.py    # Recognition page
│       └── face_swap.py           # Swap page
│
├── models/                        # AI model weights (git-ignored)
│   ├── yolov8n-face.pt
│   ├── buffalo_l/                 # InsightFace model pack
│   ├── inswapper_128.onnx
│   └── GFPGANv1.4.pth
│
├── utils/
│   ├── image_utils.py             # Image I/O, transforms, blending
│   ├── video_utils.py             # Video I/O, frame extraction
│   ├── mask_utils.py              # Face mask generation + blending
│   ├── download_models.py         # Auto model downloader
│   └── logger.py                  # Loguru-based logger
│
├── config/
│   ├── settings.py                # Pydantic BaseSettings
│   └── config.yaml                # Default config values
│
├── tests/
│   ├── unit/                      # Unit tests per module
│   └── integration/               # Full pipeline tests
│
├── docker/
│   ├── Dockerfile                 # CPU/GPU multi-stage image
│   └── docker-compose.yml         # API + UI services
│
├── scripts/
│   ├── setup.sh                   # Linux/macOS setup
│   └── setup.bat                  # Windows setup
│
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

---

## Quick Start

### Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | — (CPU ok) | 6 GB+ (CUDA 12.x) |
| Disk Space | 5 GB | 10 GB |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | — |

### 1-Command Setup (Linux / macOS)

```bash
git clone https://github.com/your-org/ai-face-recognition.git
cd ai-face-recognition
bash scripts/setup.sh         # CPU
bash scripts/setup.sh --gpu   # CUDA GPU
```

### 1-Command Setup (Windows)

```bat
git clone https://github.com/your-org/ai-face-recognition.git
cd ai-face-recognition
scripts\setup.bat
```

---

## Installation

### Manual Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/ai-face-recognition.git
cd ai-face-recognition

# 2. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch (choose ONE)
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 (GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install ONNX Runtime (choose ONE)
pip install onnxruntime          # CPU
pip install onnxruntime-gpu      # GPU

# 6. Install all other dependencies
pip install -r requirements.txt

# 7. Copy environment config
cp .env.example .env
# Edit .env with your settings

# 8. Download model weights
python utils/download_models.py --minimum  # Required models only
python utils/download_models.py --all      # All models (including enhancers)
```

### Windows Additional Requirements

InsightFace requires C++ build tools on Windows:

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Select: **Desktop development with C++**
2. Install [CMake](https://cmake.org/download/)
   - Add to PATH during installation

---

## Configuration

All settings are controlled via environment variables or the `.env` file.
Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

### Key Settings

```env
# Hardware
EXECUTION_PROVIDER=cuda       # cuda | cpu | mps (Apple Silicon)
DEVICE_ID=0                   # GPU index

# YOLOv8 Detection
DETECTOR_CONFIDENCE_THRESHOLD=0.5
DETECTOR_MAX_FACES=10

# Face Recognition
RECOGNIZER_SIMILARITY_THRESHOLD=0.45

# Face Enhancement
ENHANCER_BACKEND=gfpgan       # gfpgan | codeformer | none
ENHANCER_FIDELITY_WEIGHT=0.5

# API Server
API_PORT=8000
API_WORKERS=1

# Ethics
ETHICS_REQUIRE_CONSENT=true
ETHICS_WATERMARK_OUTPUT=true
```

See `.env.example` for the full list of options.

---

## Usage

### Start the API Server

```bash
# Activate virtual environment first
source .venv/bin/activate

# Development (with hot-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Start the Web UI

```bash
streamlit run ui/app.py --server.port 8501
```

Open your browser at: **http://localhost:8501**

### Download Models

```bash
# Minimum required models (detect + recognize + swap)
python utils/download_models.py --minimum

# All models (includes GFPGAN + CodeFormer enhancers)
python utils/download_models.py --all

# Single model
python utils/download_models.py --model inswapper_128

# Check what's installed
python utils/download_models.py --check

# Force re-download
python utils/download_models.py --all --force
```

### Python API (Programmatic Usage)

```python
from core.pipeline import FacePipeline
from utils import load_image, save_image

# Initialize the pipeline
pipeline = FacePipeline(
    detector_device="cuda",   # or "cpu"
    enhance=True,             # Enable GFPGAN post-processing
    enhancer_backend="gfpgan",
)

# Load images
source_image = load_image("path/to/source_face.jpg")   # Face to copy FROM
target_image = load_image("path/to/target.jpg")         # Image to swap INTO

# Run face swap
result = pipeline.swap(
    source=source_image,
    target=target_image,
    consent=True,             # Required: explicit consent flag
)

# Save result
save_image(result.output_image, "output/swapped.jpg")
print(f"Detected {result.num_faces} faces")
print(f"Processing time: {result.processing_time_ms:.0f}ms")
```

### Face Recognition

```python
from core.recognizer import InsightFaceRecognizer, FaceDatabase

# Initialize recognizer
recognizer = InsightFaceRecognizer(model_pack="buffalo_l")

# Build a face database
db = FaceDatabase()
db.register("Alice", recognizer.get_embedding(load_image("alice.jpg")))
db.register("Bob",   recognizer.get_embedding(load_image("bob.jpg")))
db.save("cache/face_db.pkl")

# Recognize a new face
query_image = load_image("unknown_person.jpg")
embedding = recognizer.get_embedding(query_image)
match = db.search(embedding, threshold=0.45)

if match:
    print(f"Recognized: {match.identity} (similarity={match.similarity:.3f})")
else:
    print("Unknown person")
```

---

## API Reference

Interactive docs available at: **http://localhost:8000/docs** (Swagger UI)

### Endpoints

#### `GET /api/v1/health`
Check API + model readiness.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "development",
  "uptime_seconds": 42.3,
  "components": {
    "detector":   {"status": "ok", "loaded": true, "detail": null},
    "recognizer": {"status": "ok", "loaded": true, "detail": null},
    "swapper":    {"status": "ok", "loaded": true, "detail": null},
    "enhancer":   {"status": "ok", "loaded": false, "detail": "disabled"}
  }
}
```

---

#### `POST /api/v1/recognize`
Detect and identify faces in an uploaded image.

**Request:** `multipart/form-data`
- `image` (file) — image file (JPEG/PNG/WebP/BMP)
- `consent` (bool, required) — must be `true`
- `top_k` (int, optional, default: `1`) — candidates per face
- `similarity_threshold` (float, optional) — override server default
- `return_attributes` (bool, optional) — include age/gender

**Response:**
```json
{
  "num_faces_detected": 1,
  "num_faces_recognized": 1,
  "faces": [
    {
      "face_index": 0,
      "bbox": {"x1": 100, "y1": 80, "x2": 300, "y2": 320, "confidence": 0.97},
      "landmarks": null,
      "attributes": {"age": 28.5, "gender": "F", "gender_score": 0.92},
      "match": {
        "identity_name": "Alice",
        "identity_id": "uuid-1234",
        "similarity": 0.87,
        "is_known": true,
        "threshold_used": 0.45
      },
      "embedding_norm": 1.0
    }
  ],
  "inference_time_ms": 34.2,
  "image_width": 640,
  "image_height": 480
}
```

---

#### `POST /api/v1/swap`
Swap faces between a source and target image.

**Request:** `multipart/form-data`
- `source_file` (file) — image containing the source face (donor identity)
- `target_file` (file) — image to swap the face into
- `consent` (bool, required) — must be `true`
- `blend_mode` (str, optional, default: `"poisson"`) — `poisson` | `alpha` | `masked_alpha`
- `enhance` (bool, optional, default: `false`) — apply GFPGAN/CodeFormer enhancement
- `source_face_index` (int, optional, default: `0`) — which face in source to use
- `target_face_index` (int, optional, default: `0`) — which face in target to replace
- `return_base64` (bool, optional, default: `false`) — return JSON with base64 image

**Response (default):** `image/png` — the swapped result image

**Response (`return_base64=true`):**
```json
{
  "output_url": "/api/v1/results/swap_abc123.png",
  "output_base64": "<base64 string>",
  "num_faces_swapped": 1,
  "num_faces_failed": 0,
  "faces": [
    {
      "face_index": 0,
      "bbox": {"x1": 100, "y1": 80, "x2": 300, "y2": 320, "confidence": 0.96},
      "success": true,
      "status": "success",
      "timing": {"align_ms": 2.1, "inference_ms": 18.4, "blend_ms": 3.7, "total_ms": 24.2},
      "error": null
    }
  ],
  "total_inference_ms": 24.2,
  "blend_mode": "poisson",
  "enhanced": false,
  "watermarked": true
}
```

---

#### `POST /api/v1/register`
Register a new face identity in the face database.

**Request:** `multipart/form-data`
- `image` (file) — image containing the face to register
- `name` (str, required) — identity label for this face
- `consent` (bool, required) — must be `true`
- `identity_id` (str, optional) — existing UUID to append embeddings to
- `overwrite` (bool, optional, default: `false`) — replace existing embeddings

**Response:**
```json
{
  "identity_id": "a1b2c3d4-1234-5678-abcd-ef0123456789",
  "identity_name": "Alice",
  "embeddings_added": 1,
  "total_embeddings": 3,
  "faces_detected": 1,
  "message": "Identity 'Alice' updated with 1 new embedding."
}
```

---

## Development

### Install Dev Dependencies

```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

### Code Formatting

```bash
# Format code
black .
isort .

# Lint
flake8 .

# Type checking
mypy .
```

### Adding a New Detector

The project uses an abstract `BaseDetector` class for easy extensibility:

```python
# core/detector/my_detector.py
from core.detector.base_detector import BaseDetector, DetectionResult

class MyCustomDetector(BaseDetector):
    def load_model(self) -> None:
        # Load your model here
        ...

    def detect(self, image: np.ndarray) -> DetectionResult:
        # Return DetectionResult with face boxes
        ...
```

---

## Docker

### Build & Run (CPU)

```bash
cd docker

# Build
docker-compose build

# Start all services (API + UI)
docker-compose up

# Download models (first run)
docker-compose --profile setup up model-downloader
```

### Build & Run (GPU)

```bash
# Requires: NVIDIA Container Toolkit
# Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Service URLs (Docker)

| Service | URL |
|---|---|
| FastAPI Backend | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |

---

## Roadmap

### Phase 1 — Environment & Scaffold ✅
- Project structure, requirements, config, logger, utilities

### Phase 2 — Face Detection ✅
- `YOLOFaceDetector` with `BaseDetector` abstraction
- Image + video + webcam input support

### Phase 3 — Face Recognition ✅
- `InsightFaceRecognizer` (ArcFace embeddings)
- `FaceDatabase` with cosine similarity search + persistence

### Phase 4 — Face Swap Engine ✅
- `InSwapper` wrapping `inswapper_128.onnx`
- Face alignment using 5-point landmarks
- Poisson blending for seamless compositing

### Phase 5 — Face Enhancement ✅
- `GFPGANEnhancer` for post-swap quality restoration
- `CodeFormerEnhancer` as alternative backend

### Phase 6 — Pipeline Orchestration ✅
- `FacePipeline` (image) + `VideoPipeline` (video)
- Progress tracking, error handling, telemetry

### Phase 7 — FastAPI Backend ✅
- REST endpoints: `/swap`, `/recognize`, `/register`, `/health`
- File upload validation, rate limiting, CORS

### Phase 8 — Streamlit UI ✅
- Side-by-side preview
- Face recognition results display
- Video processing with progress bar

### Phase 9 — Testing & Benchmarks ✅
- Unit tests for every module
- Integration tests for full pipeline
- FPS benchmark: CPU vs GPU

### Phase 10 — Docker & Deployment ✅
- Dockerfile (CPU + GPU), docker-compose, GPU override

---

## Ethics & Safety

This project is built with **responsible AI principles**:

- ✅ **Consent gate** — All swap API requests require `consent=true` flag
- ✅ **Output watermarking** — All swapped outputs are stamped "AI GENERATED"
- ✅ **Request logging** — All swap operations are logged (metadata only)
- ✅ **No cloud storage** — All processing is local; no data sent to third parties
- ⚠️ **Use responsibly** — Do NOT use this technology to create non-consensual deepfakes
- ⚠️ **Legal compliance** — Laws on deepfakes vary by jurisdiction; know your local laws

> **The authors are not responsible for misuse of this software.**
> This project is intended for education, research, and legitimate creative applications only.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

| Component | License |
|---|---|
| YOLOv8 (Ultralytics) | AGPL-3.0 |
| InsightFace | MIT |
| GFPGAN | Apache 2.0 |
| CodeFormer | S-Lab License 1.0 |
| inswapper_128 | Research / Non-commercial |
| PyTorch | BSD-3-Clause |
| FastAPI | MIT |
| Streamlit | Apache 2.0 |

> **Important licensing notes:**
> - **AGPL-3.0 (YOLOv8)**: If you deploy this application as a network service,
>   AGPL-3.0 requires you to release your source code to users of that service.
> - **inswapper_128.onnx** is subject to a non-commercial research license.
> - **CodeFormer** is licensed under S-Lab License 1.0 (non-commercial).
>
> Review all component licenses before any commercial deployment.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [InsightFace](https://github.com/deepinsight/insightface) by deepinsight
- [GFPGAN](https://github.com/TencentARC/GFPGAN) by TencentARC
- [CodeFormer](https://github.com/sczhou/CodeFormer) by S-Lab
- [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) for inspiration
- [akanametov/yolo-face](https://github.com/akanametov/yolo-face) for YOLOv8-face weights

---

<div align="center">
  <sub>Built with ❤️ by the AI Face Recognition team</sub>
</div>