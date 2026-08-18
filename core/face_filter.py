from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class QualityEstimate:
    score: float
    accepted: bool
    face_detected: bool
    face_count: int
    backend: str


class FaceQualityFilter:
    def __init__(
        self,
        minimum_quality: float = 0.18,
        yunet_path: str | Path | None = None,
        face_confidence_threshold: float = 0.65,
    ) -> None:
        self.minimum_quality = minimum_quality
        self.face_detector = None
        self.detector_backend = "none"
        self.face_confidence_threshold = float(face_confidence_threshold)
        try:
            import cv2

            if yunet_path is not None:
                detector_path = Path(yunet_path)
                if not detector_path.exists():
                    raise FileNotFoundError(f"YuNet checkpoint not found: {detector_path}")
                self.face_detector = cv2.FaceDetectorYN.create(
                    str(detector_path), "", (256, 256), 0.55, 0.3, 5000
                )
                self.detector_backend = "opencv_yunet"
            else:
                detector_path = (
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                detector = cv2.CascadeClassifier(detector_path)
                if not detector.empty():
                    self.face_detector = detector
                    self.detector_backend = "opencv_haar"
        except (ImportError, AttributeError):
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
        face_count = 0 if require_face_detection else 1
        face_detected = not require_face_detection
        backend = "contrast_exposure_sharpness"
        if require_face_detection and self.face_detector is not None:
            gray_uint8 = (gray * 255).astype(np.uint8)
            if self.detector_backend == "opencv_yunet":
                import cv2

                rgb_uint8 = (rgb * 255).round().astype(np.uint8)
                self.face_detector.setInputSize(image.size)
                _, detected = self.face_detector.detect(
                    cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
                )
                faces = [] if detected is None else [
                    face for face in detected
                    if float(face[-1]) >= self.face_confidence_threshold
                    and float(face[2] * face[3]) / float(image.width * image.height) >= 0.045
                ]
            else:
                minimum_side = max(40, min(image.size) // 5)
                faces = self.face_detector.detectMultiScale(
                    gray_uint8,
                    scaleFactor=1.08,
                    minNeighbors=3,
                    minSize=(minimum_side, minimum_side),
                )
            face_detected = len(faces) > 0
            face_count = len(faces)
            backend = f"{self.detector_backend}+contrast_exposure_sharpness"
        elif not require_face_detection:
            backend = "demo_bypass+contrast_exposure_sharpness"
        else:
            backend = "face_detector_unavailable+contrast_exposure_sharpness"
        accepted = face_detected and score >= self.minimum_quality
        return QualityEstimate(
            score=score if face_detected else 0.0,
            accepted=accepted,
            face_detected=face_detected,
            face_count=face_count,
            backend=backend,
        )
