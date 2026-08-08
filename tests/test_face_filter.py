from PIL import Image

from core.face_filter import FaceQualityFilter


def test_non_face_image_is_rejected_by_hard_detector():
    image = Image.new("RGB", (256, 256), color=(0, 255, 255))
    estimate = FaceQualityFilter().evaluate(image)

    assert not estimate.accepted
    assert not estimate.face_detected
    assert estimate.score == 0.0
    assert estimate.backend.startswith(
        ("opencv_haar", "face_detector_unavailable")
    )
