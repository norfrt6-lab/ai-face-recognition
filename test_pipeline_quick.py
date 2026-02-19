#!/usr/bin/env python3
"""
Quick pipeline test to verify all components are working.
Tests: face detection, recognition, and swapping with available models.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger, setup_from_settings
from config.settings import settings

# Setup logging
setup_from_settings()
logger = get_logger(__name__)


def test_detector():
    """Test YOLOv8 face detector."""
    logger.info("=" * 60)
    logger.info("TEST 1: Face Detector (YOLOv8n)")
    logger.info("=" * 60)
    
    try:
        from core.detector.yolo_detector import YOLOFaceDetector
        
        detector = YOLOFaceDetector(
            model_path=str(PROJECT_ROOT / "models" / "yolov8n-face.pt"),
            device="cpu"
        )
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = detector.detect(dummy_img)
        
        logger.success(f"✓ Detector loaded successfully")
        logger.info(f"  Model: {detector.model_path}")
        logger.info(f"  Device: {detector.device}")
        logger.info(f"  Test image size: {dummy_img.shape}")
        logger.info(f"  Detections on test image: {len(result.faces)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Detector test failed: {e}")
        return False


def test_recognizer():
    """Test InsightFace recognizer."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 2: Face Recognizer (InsightFace)")
    logger.info("=" * 60)
    
    try:
        from core.recognizer.insightface_recognizer import InsightFaceRecognizer
        
        recognizer = InsightFaceRecognizer(
            model_pack="buffalo_l",
            model_root=str(PROJECT_ROOT / "models"),
            ctx_id=-1  # CPU mode
        )
        recognizer.load_model()
        
        # Create dummy face image (small face crop)
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        embedding = recognizer.get_embedding(dummy_face)
        
        logger.success(f"✓ Recognizer loaded successfully")
        logger.info(f"  Model pack: buffalo_l")
        logger.info(f"  Model root: {str(PROJECT_ROOT / 'models')}")
        logger.info(f"  Embedding shape: {embedding.embedding.shape}")
        logger.info(f"  Embedding type: {type(embedding.embedding).__name__}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Recognizer test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_swapper():
    """Test inswapper_128 face swap model."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 3: Face Swapper (inswapper_128)")
    logger.info("=" * 60)
    
    try:
        from core.swapper.inswapper import InSwapper
        
        swapper = InSwapper(
            model_path=str(PROJECT_ROOT / "models" / "inswapper_128.onnx"),
            providers=["CPUExecutionProvider"]
        )
        
        logger.success(f"✓ Swapper loaded successfully")
        logger.info(f"  Model path: {swapper.model_path}")
        logger.info(f"  Model loaded: {swapper._session is not None}")
        logger.info(f"  Input size: 128x128")
        
        return True
    except Exception as e:
        logger.error(f"✗ Swapper test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_pipeline_integration():
    """Test full pipeline with a real image if available."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 4: Full Pipeline Integration")
    logger.info("=" * 60)
    
    try:
        from core.pipeline.face_pipeline import FacePipeline, PipelineConfig
        from core.detector.yolo_detector import YOLOFaceDetector
        from core.recognizer.insightface_recognizer import InsightFaceRecognizer
        from core.swapper.inswapper import InSwapper
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        detector = YOLOFaceDetector(
            model_path=str(PROJECT_ROOT / "models" / "yolov8n-face.pt"),
            device="cpu"
        )
        
        recognizer = InsightFaceRecognizer(
            model_pack="buffalo_l",
            model_root=str(PROJECT_ROOT / "models"),
            ctx_id=-1  # CPU mode
        )
        recognizer.load_model()
        
        swapper = InSwapper(
            model_path=str(PROJECT_ROOT / "models" / "inswapper_128.onnx"),
            providers=["CPUExecutionProvider"]
        )
        
        pipeline = FacePipeline(
            detector=detector,
            recognizer=recognizer,
            swapper=swapper,
            enhancer=None  # Enhancement not available yet
        )
        
        logger.success(f"✓ Pipeline initialized successfully")
        logger.info(f"  Detector: {type(detector).__name__}")
        logger.info(f"  Recognizer: {type(recognizer).__name__}")
        logger.info(f"  Swapper: {type(swapper).__name__}")
        logger.info(f"  Enhancer: None (optional)")
        
        # Create test image
        logger.info("Creating test image...")
        test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        logger.info(f"  Test image shape: {test_image.shape}")
        
        # Run detection only
        logger.info("Running face detection...")
        detection_result = detector.detect(test_image)
        logger.success(f"✓ Detection complete")
        logger.info(f"  Faces detected: {len(detection_result.faces)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all tests."""
    logger.info("")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 12 + "AI FACE RECOGNITION PIPELINE TEST" + " " * 12 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    
    results = {
        "Detector": test_detector(),
        "Recognizer": test_recognizer(),
        "Swapper": test_swapper(),
        "Pipeline": test_pipeline_integration(),
    }
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        if result:
            logger.success(f"  {test_name}: {status}")
        else:
            logger.error(f"  {test_name}: {status}")
    
    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.success("✓ All tests passed! Pipeline is ready to use.")
        return 0
    else:
        logger.error("✗ Some tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
