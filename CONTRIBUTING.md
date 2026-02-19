# Contributing

Thank you for your interest in contributing to AI Face Recognition & Face Swap.

## Development Setup

```bash
# 1. Clone and create virtual environment
git clone https://github.com/your-org/ai-face-recognition.git
cd ai-face-recognition
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Install pre-commit hooks
pre-commit install

# 4. Download minimum models
python utils/download_models.py --minimum

# 5. Run tests
pytest tests/unit/ -v
```

## Architecture Overview

The project follows a layered architecture with abstract base classes for extensibility:

```
Utility Layer    →  utils/ (image, video, mask, logger, download)
Core Engine      →  core/ (detector, recognizer, swapper, enhancer)
Orchestration    →  core/pipeline/ (face_pipeline, video_pipeline)
API Layer        →  api/ (FastAPI routers, schemas, middleware)
UI Layer         →  ui/ (Streamlit pages)
```

### Abstract Base Classes

Each AI component has a `Base*` class defining the interface contract:

| Base Class | Location | Implementations |
|---|---|---|
| `BaseDetector` | `core/detector/base_detector.py` | `YOLOFaceDetector` |
| `BaseRecognizer` | `core/recognizer/base_recognizer.py` | `InsightFaceRecognizer` |
| `BaseSwapper` | `core/swapper/base_swapper.py` | `InSwapper` |
| `BaseEnhancer` | `core/enhancer/base_enhancer.py` | `GFPGANEnhancer`, `CodeFormerEnhancer` |

## Adding a New Backend

### Example: Adding a new face detector

1. Create `core/detector/my_detector.py`
2. Subclass `BaseDetector` and implement `load_model()`, `detect()`, and `release()`
3. Add corresponding settings to `config/settings.py` if needed
4. Add unit tests in `tests/unit/test_detector.py`
5. Register it as an option in the pipeline / settings factory

### Example: Adding a new enhancer

1. Create `core/enhancer/my_enhancer.py`
2. Subclass `BaseEnhancer` and implement `load_model()`, `enhance()`, and `release()`
3. Add it as a `backend` option in `EnhancerSettings`
4. Add tests in `tests/unit/test_enhancer.py`

## Test Structure

```
tests/
  unit/               # Fast tests, no models required, full mock isolation
    test_detector.py
    test_recognizer.py
    test_swapper.py
    test_enhancer.py
  integration/        # Requires downloaded models
    test_api.py
    test_pipeline.py
```

Run specific test categories:
```bash
pytest tests/unit/ -v                          # Unit tests only
pytest tests/integration/ -v                   # Integration tests
pytest tests/ -m "not integration" -v          # Skip integration
pytest tests/ --cov=core --cov-report=html     # With coverage
```

## Code Style

- **Formatter**: Black (line length 100)
- **Import sorting**: isort (black profile)
- **Linting**: flake8
- **Type checking**: mypy

All checks run automatically via pre-commit hooks. You can also run manually:
```bash
black .
isort .
flake8 .
mypy .
```

## PR Checklist

Before submitting a pull request, ensure:

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] Coverage is maintained at >= 80%
- [ ] New code has corresponding unit tests
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] Docstrings are present on public functions/classes
- [ ] No hardcoded paths or credentials
- [ ] `consent=true` gate is preserved on any new swap/recognition endpoint
