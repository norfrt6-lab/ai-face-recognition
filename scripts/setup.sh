#!/usr/bin/env bash
# ============================================================
# AI Face Recognition & Face Swap - Environment Setup Script
# ============================================================
# Usage:
#   bash scripts/setup.sh            # Full setup (CPU)
#   bash scripts/setup.sh --gpu      # Full setup (CUDA GPU)
#   bash scripts/setup.sh --minimal  # Minimum deps only
#   bash scripts/setup.sh --no-models # Skip model downloads
# ============================================================

set -euo pipefail

# ── Colours ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Helpers ──────────────────────────────────────────────────
info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
section() { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${RESET}"; \
            echo -e "${BOLD}${CYAN}  $*${RESET}"; \
            echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}"; }

# ── Defaults ─────────────────────────────────────────────────
GPU_MODE=false
MINIMAL=false
SKIP_MODELS=false
PYTHON_MIN_VERSION="3.10"
VENV_DIR=".venv"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse arguments ──────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --gpu)        GPU_MODE=true ;;
    --minimal)    MINIMAL=true ;;
    --no-models)  SKIP_MODELS=true ;;
    --help|-h)
      echo "Usage: bash scripts/setup.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --gpu         Install GPU (CUDA) dependencies"
      echo "  --minimal     Install only minimum required packages"
      echo "  --no-models   Skip downloading model weights"
      echo "  --help        Show this help message"
      exit 0
      ;;
    *)
      warn "Unknown argument: $arg (ignored)"
      ;;
  esac
done

# ── Banner ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║     AI Face Recognition & Face Swap              ║"
echo "  ║     Environment Setup                            ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${RESET}"
info "Project root : $PROJECT_ROOT"
info "GPU mode     : $GPU_MODE"
info "Minimal      : $MINIMAL"
info "Skip models  : $SKIP_MODELS"

cd "$PROJECT_ROOT"

# ============================================================
# STEP 1 — Check OS & system dependencies
# ============================================================
section "Step 1 — System checks"

OS="$(uname -s)"
ARCH="$(uname -m)"
info "Operating system : $OS ($ARCH)"

# Detect Windows (Git Bash / MSYS2 / WSL)
IS_WINDOWS=false
if [[ "$OS" == MINGW* ]] || [[ "$OS" == MSYS* ]] || [[ "$OS" == CYGWIN* ]]; then
  IS_WINDOWS=true
  warn "Running on Windows shell (Git Bash / MSYS2). Some features may differ."
fi

# Check for required tools
check_cmd() {
  if command -v "$1" &>/dev/null; then
    success "$1 found: $(command -v "$1")"
    return 0
  else
    return 1
  fi
}

# git
if ! check_cmd git; then
  error "git is not installed. Please install git first."
  exit 1
fi

# cmake (needed by dlib / insightface on some platforms)
if ! check_cmd cmake; then
  warn "cmake not found. It may be required by insightface on some platforms."
  warn "Install: https://cmake.org/download/"
fi

# ffmpeg (needed for video audio merge)
if ! check_cmd ffmpeg; then
  warn "ffmpeg not found. Video audio preservation will be disabled."
  if [[ "$IS_WINDOWS" == true ]]; then
    warn "Install ffmpeg: winget install ffmpeg"
  elif [[ "$OS" == "Darwin" ]]; then
    warn "Install ffmpeg: brew install ffmpeg"
  else
    warn "Install ffmpeg: sudo apt-get install ffmpeg"
  fi
fi

# ============================================================
# STEP 2 — Check Python version
# ============================================================
section "Step 2 — Python version check"

PYTHON_CMD=""
for cmd in python3.11 python3.10 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    version=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)
    if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
      PYTHON_CMD="$cmd"
      success "Python $version found at: $(command -v "$cmd")"
      break
    fi
  fi
done

if [[ -z "$PYTHON_CMD" ]]; then
  error "Python $PYTHON_MIN_VERSION+ is required but not found."
  error "Download Python: https://www.python.org/downloads/"
  exit 1
fi

# ============================================================
# STEP 3 — Create virtual environment
# ============================================================
section "Step 3 — Virtual environment"

if [[ -d "$VENV_DIR" ]]; then
  warn "Virtual environment already exists at $VENV_DIR — skipping creation."
  warn "To recreate it, delete the directory first: rm -rf $VENV_DIR"
else
  info "Creating virtual environment in $VENV_DIR ..."
  "$PYTHON_CMD" -m venv "$VENV_DIR"
  success "Virtual environment created."
fi

# Activate venv
if [[ "$IS_WINDOWS" == true ]]; then
  VENV_ACTIVATE="$VENV_DIR/Scripts/activate"
else
  VENV_ACTIVATE="$VENV_DIR/bin/activate"
fi

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  error "Could not find venv activation script at: $VENV_ACTIVATE"
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"
success "Virtual environment activated."

# Resolve pip inside the venv
PIP_CMD="python -m pip"

# ============================================================
# STEP 4 — Upgrade pip, setuptools, wheel
# ============================================================
section "Step 4 — Upgrade pip / setuptools / wheel"

$PIP_CMD install --upgrade pip setuptools wheel
success "pip, setuptools, wheel up to date."

# ============================================================
# STEP 5 — Install PyTorch
# ============================================================
section "Step 5 — Install PyTorch"

# Check if PyTorch is already installed
if python -c "import torch; print(torch.__version__)" &>/dev/null; then
  TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
  success "PyTorch already installed: $TORCH_VERSION"
else
  if [[ "$GPU_MODE" == true ]]; then
    info "Installing PyTorch with CUDA 12.1 support..."
    $PIP_CMD install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu121
  else
    info "Installing PyTorch (CPU only) ..."
    $PIP_CMD install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cpu
  fi
  TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
  success "PyTorch installed: $TORCH_VERSION"
fi

# ============================================================
# STEP 6 — Install ONNX Runtime
# ============================================================
section "Step 6 — Install ONNX Runtime"

if [[ "$GPU_MODE" == true ]]; then
  info "Installing onnxruntime-gpu ..."
  $PIP_CMD install onnxruntime-gpu
else
  info "Installing onnxruntime (CPU) ..."
  # Uninstall GPU version if present to avoid conflicts
  $PIP_CMD uninstall -y onnxruntime-gpu 2>/dev/null || true
  $PIP_CMD install onnxruntime
fi
success "ONNX Runtime installed."

# ============================================================
# STEP 7 — Install core requirements
# ============================================================
section "Step 7 — Install core dependencies"

if [[ "$MINIMAL" == true ]]; then
  info "Installing minimal requirements only ..."
  $PIP_CMD install \
    "ultralytics>=8.2.0" \
    "insightface>=0.7.3" \
    "opencv-python>=4.9.0.80" \
    "numpy>=1.26.0" \
    "Pillow>=10.3.0" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-multipart>=0.0.9" \
    "pydantic>=2.7.0" \
    "pydantic-settings>=2.2.0" \
    "tqdm>=4.66.0" \
    "loguru>=0.7.2" \
    "PyYAML>=6.0.1" \
    "python-dotenv>=1.0.1" \
    "requests>=2.31.0" \
    "aiofiles>=23.2.1" \
    "streamlit>=1.35.0"
else
  info "Installing full requirements from requirements.txt ..."
  if [[ -f "requirements.txt" ]]; then
    # Remove onnxruntime-gpu from requirements.txt if CPU mode to avoid conflict
    if [[ "$GPU_MODE" == false ]]; then
      grep -v "onnxruntime-gpu" requirements.txt > /tmp/requirements_cpu.txt || true
      $PIP_CMD install -r /tmp/requirements_cpu.txt
      rm -f /tmp/requirements_cpu.txt
    else
      $PIP_CMD install -r requirements.txt
    fi
  else
    error "requirements.txt not found at project root."
    exit 1
  fi
fi

success "Core dependencies installed."

# ============================================================
# STEP 8 — Install insightface extra (C++ build)
# ============================================================
section "Step 8 — InsightFace build check"

if python -c "import insightface" &>/dev/null; then
  IF_VERSION=$(python -c "import insightface; print(insightface.__version__)" 2>/dev/null || echo "unknown")
  success "InsightFace already installed: $IF_VERSION"
else
  warn "InsightFace not installed yet — attempting install ..."
  warn "This may require C++ build tools:"
  if [[ "$IS_WINDOWS" == true ]]; then
    warn "  Windows: install Visual Studio Build Tools from"
    warn "  https://visualstudio.microsoft.com/visual-cpp-build-tools/"
  elif [[ "$OS" == "Darwin" ]]; then
    warn "  macOS: xcode-select --install"
  else
    warn "  Linux: sudo apt-get install build-essential cmake"
  fi

  $PIP_CMD install insightface || {
    error "InsightFace installation failed."
    error "Please install C++ build tools (see warnings above) and re-run."
    exit 1
  }
  success "InsightFace installed."
fi

# ============================================================
# STEP 9 — Install development dependencies (optional)
# ============================================================
section "Step 9 — Dev dependencies"

if [[ "$MINIMAL" == false ]] && [[ -f "requirements-dev.txt" ]]; then
  read -r -p "Install dev/test dependencies? [y/N] " response
  if [[ "$response" =~ ^[Yy]$ ]]; then
    $PIP_CMD install -r requirements-dev.txt
    success "Dev dependencies installed."
  else
    info "Skipping dev dependencies."
  fi
else
  info "Skipping dev dependencies (minimal mode or file not found)."
fi

# ============================================================
# STEP 10 — Create required directories
# ============================================================
section "Step 10 — Create runtime directories"

DIRS=(
  "models"
  "models/buffalo_l"
  "models/facexlib"
  "uploads"
  "output"
  "cache"
  "logs"
  "tmp"
  "tmp/frames"
  "data"
)

for dir in "${DIRS[@]}"; do
  if [[ ! -d "$dir" ]]; then
    mkdir -p "$dir"
    success "Created: $dir/"
  else
    info "Exists:  $dir/"
  fi
done

# Create .gitkeep files so empty dirs are tracked by git
for dir in uploads output cache logs tmp data; do
  touch "$dir/.gitkeep" 2>/dev/null || true
done

# ============================================================
# STEP 11 — Copy .env.example → .env (if not present)
# ============================================================
section "Step 11 — Environment file"

if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    cp ".env.example" ".env"
    success ".env created from .env.example"
    info "  → Review and edit .env before running the application."
    if [[ "$GPU_MODE" == true ]]; then
      # Update execution provider to cuda
      if [[ "$OS" == "Darwin" ]]; then
        sed -i '' 's/EXECUTION_PROVIDER=cpu/EXECUTION_PROVIDER=cuda/' .env 2>/dev/null || true
      else
        sed -i 's/EXECUTION_PROVIDER=cpu/EXECUTION_PROVIDER=cuda/' .env 2>/dev/null || true
      fi
      info "  → Set EXECUTION_PROVIDER=cuda in .env"
    fi
  else
    warn ".env.example not found — skipping .env creation."
  fi
else
  info ".env already exists — skipping."
fi

# ============================================================
# STEP 12 — Download model weights
# ============================================================
section "Step 12 — Model weights"

if [[ "$SKIP_MODELS" == true ]]; then
  warn "Skipping model downloads (--no-models flag set)."
  warn "Run later: python utils/download_models.py --minimum"
else
  info "Downloading minimum required model weights ..."
  info "(This may take several minutes on first run)"

  python utils/download_models.py --minimum || {
    warn "Model download encountered errors."
    warn "You can retry with: python utils/download_models.py --minimum"
    warn "Or download all models: python utils/download_models.py --all"
  }
fi

# ============================================================
# STEP 13 — GPU verification (if --gpu)
# ============================================================
if [[ "$GPU_MODE" == true ]]; then
  section "Step 13 — GPU verification"

  python - <<'EOF'
import torch
import onnxruntime as ort

print(f"PyTorch version  : {torch.__version__}")
print(f"CUDA available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version     : {torch.version.cuda}")
    print(f"GPU device       : {torch.cuda.get_device_name(0)}")
    print(f"GPU memory       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: CUDA not available — will fall back to CPU inference.")

print(f"\nONNX Runtime     : {ort.__version__}")
print(f"Execution providers: {ort.get_available_providers()}")
EOF
fi

# ============================================================
# STEP 14 — Verify core imports
# ============================================================
section "Step 14 — Verify core imports"

python - <<'EOF'
import sys

checks = {
    "numpy":       "import numpy as np; print(f'  numpy      {np.__version__}')",
    "cv2":         "import cv2; print(f'  opencv     {cv2.__version__}')",
    "PIL":         "from PIL import Image; import PIL; print(f'  Pillow     {PIL.__version__}')",
    "ultralytics": "import ultralytics; print(f'  ultralytics {ultralytics.__version__}')",
    "insightface": "import insightface; print(f'  insightface {insightface.__version__}')",
    "onnxruntime": "import onnxruntime as ort; print(f'  onnxruntime {ort.__version__}')",
    "fastapi":     "import fastapi; print(f'  fastapi    {fastapi.__version__}')",
    "streamlit":   "import streamlit as st; print(f'  streamlit  {st.__version__}')",
    "loguru":      "from loguru import logger; print(f'  loguru     OK')",
    "pydantic":    "import pydantic; print(f'  pydantic   {pydantic.__version__}')",
    "tqdm":        "from tqdm import tqdm; print(f'  tqdm       OK')",
}

all_ok = True
for name, code in checks.items():
    try:
        exec(code)
    except ImportError as e:
        print(f"  MISSING    {name}: {e}", file=sys.stderr)
        all_ok = False

if not all_ok:
    print("\nSome imports failed. Run setup again or install missing packages.", file=sys.stderr)
    sys.exit(1)
else:
    print("\nAll core imports OK!")
EOF

success "Import verification passed."

# ============================================================
# DONE
# ============================================================
section "Setup Complete"

echo -e "${GREEN}${BOLD}"
echo "  ✓ Environment ready!"
echo -e "${RESET}"
echo -e "  Virtual env  : ${CYAN}${VENV_DIR}/${RESET}"
echo -e "  Activate     : ${CYAN}source ${VENV_DIR}/bin/activate${RESET}"
if [[ "$IS_WINDOWS" == true ]]; then
  echo -e "  (Windows)    : ${CYAN}.\\${VENV_DIR}\\Scripts\\activate${RESET}"
fi
echo ""
echo -e "  Start API    : ${CYAN}python -m uvicorn api.main:app --reload${RESET}"
echo -e "  Start UI     : ${CYAN}streamlit run ui/app.py${RESET}"
echo -e "  Download all : ${CYAN}python utils/download_models.py --all${RESET}"
echo -e "  Run tests    : ${CYAN}pytest tests/${RESET}"
echo ""
echo -e "${YELLOW}  Remember: review and edit ${BOLD}.env${RESET}${YELLOW} before first run!${RESET}"
echo ""
```

Now let me continue creating the remaining files — the Docker setup, README, and `pytest.ini`:
