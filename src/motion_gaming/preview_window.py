"""Preview window for displaying camera feed and detection overlay."""

from typing import Optional, Sequence

import cv2
import numpy as np
import numpy.typing as npt

from motion_gaming.gesture_recognizer import Direction
from motion_gaming.hand_detector import HandData


# MediaPipe hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm
]


class PreviewWindow:
    """Displays camera preview with hand detection overlay."""

    WINDOW_NAME = "Motion Gaming"

    def __init__(self) -> None:
        """Initialize the preview window."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 640, 480)

    def update(
        self,
        frame: npt.NDArray[np.uint8],
        hands: Sequence[HandData],
        direction: Optional[Direction],
        finger_count: int,
        is_tracking: bool,
    ) -> None:
        """Update the preview window with new frame and overlays.

        Args:
            frame: BGR image from camera.
            hands: Detected hands to overlay.
            direction: Current pointing direction (right hand).
            finger_count: Current finger count (left hand).
            is_tracking: Whether gesture tracking is active.
        """
        # Make a copy to draw on
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw hand landmarks
        for hand in hands:
            self._draw_hand(display, hand, w, h)

        # Draw status overlay
        self._draw_status(display, direction, finger_count, is_tracking)

        cv2.imshow(self.WINDOW_NAME, display)

    def _draw_hand(
        self,
        frame: npt.NDArray[np.uint8],
        hand: HandData,
        width: int,
        height: int,
    ) -> None:
        """Draw hand landmarks and connections."""
        # Choose color based on handedness
        color = (0, 255, 0) if hand.handedness == "Right" else (255, 0, 0)

        # Convert normalized coordinates to pixel coordinates
        landmarks_px = hand.landmarks[:, :2] * np.array([width, height])
        landmarks_px = landmarks_px.astype(np.int32)

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = tuple(landmarks_px[start_idx])
            end = tuple(landmarks_px[end_idx])
            cv2.line(frame, start, end, color, 2)

        # Draw landmarks
        for point in landmarks_px:
            cv2.circle(frame, tuple(point), 4, color, -1)

    def _draw_status(
        self,
        frame: npt.NDArray[np.uint8],
        direction: Optional[Direction],
        finger_count: int,
        is_tracking: bool,
    ) -> None:
        """Draw status overlay."""
        # Tracking status
        status_color = (0, 255, 0) if is_tracking else (0, 0, 255)
        status_text = "TRACKING" if is_tracking else "PAUSED (F12)"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        if not is_tracking:
            return

        # Direction
        dir_text = f"Direction: {direction.value if direction else 'None'}"
        cv2.putText(frame, dir_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Finger count
        finger_text = f"Fingers: {finger_count}"
        cv2.putText(frame, finger_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Keys pressed
        keys: list[str] = []
        if direction:
            from motion_gaming.input_controller import DIRECTION_KEYS
            keys.extend(DIRECTION_KEYS.get(direction, []))
        if finger_count > 0:
            keys.append(str(finger_count))

        if keys:
            keys_text = f"Keys: {' + '.join(keys)}"
            cv2.putText(frame, keys_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def is_open(self) -> bool:
        """Check if window is still open."""
        return bool(cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1)

    def close(self) -> None:
        """Close the window."""
        cv2.destroyWindow(self.WINDOW_NAME)
