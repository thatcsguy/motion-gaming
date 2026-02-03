"""Hand detection module using MediaPipe."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


@dataclass
class HandData:
    """Data for a detected hand."""

    landmarks: npt.NDArray[np.float32]  # Shape: (21, 3) - x, y, z for each landmark
    handedness: str  # "Left" or "Right"
    confidence: float


class HandDetector:
    """Detects hands in video frames using MediaPipe."""

    def __init__(self, max_hands: int = 2, min_detection_confidence: float = 0.7) -> None:
        """Initialize the hand detector.

        Args:
            max_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for detection.
        """
        # Lazy import to allow testing without mediapipe
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            model_complexity=0,  # Fastest model for real-time
        )

    def detect(self, frame: npt.NDArray[np.uint8]) -> list[HandData]:
        """Detect hands in a video frame.

        Args:
            frame: BGR image from OpenCV (H, W, 3).

        Returns:
            List of detected hands with landmarks and handedness.
        """
        import cv2

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        hands: list[HandData] = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # Extract landmarks as numpy array
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )

                # Get handedness label and confidence
                classification = handedness_info.classification[0]
                handedness = classification.label
                confidence = classification.score

                hands.append(HandData(landmarks, handedness, confidence))

        return hands

    def close(self) -> None:
        """Release resources."""
        self._hands.close()
