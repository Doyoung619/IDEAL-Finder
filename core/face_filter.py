from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class QualityEstimate:
    score: float
    accepted: bool
    face_detected: bool
    backend: str


class FaceQualityFilter:
    def __init__(self, minimum_quality: float = 0.18) -> None:
        self.minimum_quality = minimum_quality
        self.face_detector = None
        try:
            import cv2

            detector_path = Path(cv2.data.haarcascades) / (
                "haarcascade_frontalface_default.xml"
            )
            detector_paths = [detector_path]
            # OpenCV's Windows file loader can fail for cascade files below a
            # path containing non-ASCII characters (for example, a Korean
            # Windows username). Retry through an ASCII Windows temp path.
            if os.name == "nt" and any(ord(char) > 127 for char in str(detector_path)):
                temp_root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Temp"
                fallback_path = temp_root / "ideal_finder_cv2" / detector_path.name
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                if (
                    not fallback_path.exists()
                    or fallback_path.stat().st_size != detector_path.stat().st_size
                ):
                    shutil.copyfile(detector_path, fallback_path)
                detector_paths = [fallback_path]

            for candidate_path in detector_paths:
                detector = cv2.CascadeClassifier(str(candidate_path))
                if not detector.empty():
                    self.face_detector = detector
                    break
        except (ImportError, AttributeError, OSError):
            self.face_detector = None

    def evaluate(
        self,
        image: Image.Image,
        require_face_detection: bool = True,
    ) -> QualityEstimate:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        contrast = min(float(gray.std()) / 0.24, 1.0)
        exposure = 1.0 - min(abs(float(gray.mean()) - 0.5) / 0.5, 1.0)
        edges = np.asarray(
            image.convert("L").filter(ImageFilter.FIND_EDGES),
            dtype=np.float32,
        )
        sharpness = min(float(edges.std()) / 60.0, 1.0)
        color_range = min(float(np.ptp(rgb, axis=(0, 1)).mean()) / 0.45, 1.0)
        score = float(
            0.35 * contrast
            + 0.25 * exposure
            + 0.25 * sharpness
            + 0.15 * color_range
        )
        face_detected = True
        backend = "contrast_exposure_sharpness"
        if require_face_detection and self.face_detector is not None:
            gray_uint8 = (gray * 255).astype(np.uint8)
            minimum_side = max(40, min(image.size) // 5)
            faces = self.face_detector.detectMultiScale(
                gray_uint8,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(minimum_side, minimum_side),
            )
            face_detected = len(faces) > 0
            backend = "opencv_haar+contrast_exposure_sharpness"
        elif not require_face_detection:
            backend = "demo_bypass+contrast_exposure_sharpness"
        accepted = face_detected and score >= self.minimum_quality
        return QualityEstimate(
            score=score if face_detected else 0.0,
            accepted=accepted,
            face_detected=face_detected,
            backend=backend,
        )
