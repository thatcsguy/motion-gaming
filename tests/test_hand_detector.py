"""Tests for hand detector."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestHandDetector:
    """Tests for HandDetector."""

    @pytest.fixture
    def mock_mediapipe(self):
        """Create mocked mediapipe module."""
        mock_mp = MagicMock()
        mock_hands_instance = MagicMock()
        mock_mp.solutions.hands.Hands.return_value = mock_hands_instance
        return mock_mp, mock_hands_instance

    @pytest.fixture
    def mock_cv2(self):
        """Create mocked cv2 module."""
        mock = MagicMock()
        # cvtColor returns the frame unchanged for testing
        mock.cvtColor.side_effect = lambda frame, code: frame
        mock.COLOR_BGR2RGB = 4  # Actual OpenCV constant
        return mock

    def make_landmark(self, x: float, y: float, z: float) -> MagicMock:
        """Create a mock landmark with x, y, z coordinates."""
        lm = MagicMock()
        lm.x = x
        lm.y = y
        lm.z = z
        return lm

    def make_hand_result(
        self, handedness: str, confidence: float, landmarks: list[tuple[float, float, float]]
    ) -> tuple[MagicMock, MagicMock]:
        """Create mock hand landmarks and handedness info.

        Args:
            handedness: "Left" or "Right"
            confidence: Detection confidence 0-1
            landmarks: List of 21 (x, y, z) tuples

        Returns:
            Tuple of (hand_landmarks, handedness_info) mocks
        """
        hand_landmarks = MagicMock()
        hand_landmarks.landmark = [self.make_landmark(x, y, z) for x, y, z in landmarks]

        classification = MagicMock()
        classification.label = handedness
        classification.score = confidence

        handedness_info = MagicMock()
        handedness_info.classification = [classification]

        return hand_landmarks, handedness_info

    def make_dummy_landmarks(self) -> list[tuple[float, float, float]]:
        """Create 21 dummy landmarks at increasing positions."""
        return [(i * 0.04, i * 0.04, 0.0) for i in range(21)]

    def test_detects_no_hands_when_none_present(
        self, mock_mediapipe: tuple, mock_cv2: MagicMock
    ) -> None:
        """detect() should return empty list when no hands detected."""
        mock_mp, mock_hands = mock_mediapipe

        # MediaPipe returns None for both when no hands
        results = MagicMock()
        results.multi_hand_landmarks = None
        results.multi_handedness = None
        mock_hands.process.return_value = results

        with patch.dict("sys.modules", {"mediapipe": mock_mp, "cv2": mock_cv2}):
            from motion_gaming.hand_detector import HandDetector

            # Need to reload to pick up mock
            import importlib
            import motion_gaming.hand_detector

            importlib.reload(motion_gaming.hand_detector)
            from motion_gaming.hand_detector import HandDetector

            detector = HandDetector()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            hands = detector.detect(frame)

            assert hands == []

    def test_detects_single_right_hand(
        self, mock_mediapipe: tuple, mock_cv2: MagicMock
    ) -> None:
        """detect() should return one HandData for a single right hand."""
        mock_mp, mock_hands = mock_mediapipe

        landmarks = self.make_dummy_landmarks()
        hand_lm, handedness = self.make_hand_result("Right", 0.95, landmarks)

        results = MagicMock()
        results.multi_hand_landmarks = [hand_lm]
        results.multi_handedness = [handedness]
        mock_hands.process.return_value = results

        with patch.dict("sys.modules", {"mediapipe": mock_mp, "cv2": mock_cv2}):
            import importlib
            import motion_gaming.hand_detector

            importlib.reload(motion_gaming.hand_detector)
            from motion_gaming.hand_detector import HandDetector

            detector = HandDetector()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            hands = detector.detect(frame)

            assert len(hands) == 1
            assert hands[0].handedness == "Right"
            assert hands[0].confidence == 0.95

    def test_detects_single_left_hand(
        self, mock_mediapipe: tuple, mock_cv2: MagicMock
    ) -> None:
        """detect() should return one HandData for a single left hand."""
        mock_mp, mock_hands = mock_mediapipe

        landmarks = self.make_dummy_landmarks()
        hand_lm, handedness = self.make_hand_result("Left", 0.85, landmarks)

        results = MagicMock()
        results.multi_hand_landmarks = [hand_lm]
        results.multi_handedness = [handedness]
        mock_hands.process.return_value = results

        with patch.dict("sys.modules", {"mediapipe": mock_mp, "cv2": mock_cv2}):
            import importlib
            import motion_gaming.hand_detector

            importlib.reload(motion_gaming.hand_detector)
            from motion_gaming.hand_detector import HandDetector

            detector = HandDetector()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            hands = detector.detect(frame)

            assert len(hands) == 1
            assert hands[0].handedness == "Left"
            assert hands[0].confidence == 0.85

    def test_detects_two_hands(
        self, mock_mediapipe: tuple, mock_cv2: MagicMock
    ) -> None:
        """detect() should return two HandData when both hands present."""
        mock_mp, mock_hands = mock_mediapipe

        landmarks = self.make_dummy_landmarks()
        left_lm, left_handed = self.make_hand_result("Left", 0.9, landmarks)
        right_lm, right_handed = self.make_hand_result("Right", 0.88, landmarks)

        results = MagicMock()
        results.multi_hand_landmarks = [left_lm, right_lm]
        results.multi_handedness = [left_handed, right_handed]
        mock_hands.process.return_value = results

        with patch.dict("sys.modules", {"mediapipe": mock_mp, "cv2": mock_cv2}):
            import importlib
            import motion_gaming.hand_detector

            importlib.reload(motion_gaming.hand_detector)
            from motion_gaming.hand_detector import HandDetector

            detector = HandDetector()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            hands = detector.detect(frame)

            assert len(hands) == 2
            handedness_set = {h.handedness for h in hands}
            assert handedness_set == {"Left", "Right"}

    def test_landmarks_converted_to_numpy_array(
        self, mock_mediapipe: tuple, mock_cv2: MagicMock
    ) -> None:
        """Landmarks should be converted to (21, 3) numpy float32 array."""
        mock_mp, mock_hands = mock_mediapipe

        # Specific landmarks to verify conversion
        landmarks = [(0.1 * i, 0.2 * i, 0.01 * i) for i in range(21)]
        hand_lm, handedness = self.make_hand_result("Right", 0.9, landmarks)

        results = MagicMock()
        results.multi_hand_landmarks = [hand_lm]
        results.multi_handedness = [handedness]
        mock_hands.process.return_value = results

        with patch.dict("sys.modules", {"mediapipe": mock_mp, "cv2": mock_cv2}):
            import importlib
            import motion_gaming.hand_detector

            importlib.reload(motion_gaming.hand_detector)
            from motion_gaming.hand_detector import HandDetector

            detector = HandDetector()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            hands = detector.detect(frame)

            assert hands[0].landmarks.shape == (21, 3)
            assert hands[0].landmarks.dtype == np.float32

            # Verify actual values
            for i in range(21):
                assert abs(hands[0].landmarks[i, 0] - 0.1 * i) < 0.0001
                assert abs(hands[0].landmarks[i, 1] - 0.2 * i) < 0.0001
                assert abs(hands[0].landmarks[i, 2] - 0.01 * i) < 0.0001

    def test_passes_frame_through_color_conversion(
        self, mock_mediapipe: tuple, mock_cv2: MagicMock
    ) -> None:
        """Frame should be converted from BGR to RGB before processing."""
        mock_mp, mock_hands = mock_mediapipe

        results = MagicMock()
        results.multi_hand_landmarks = None
        results.multi_handedness = None
        mock_hands.process.return_value = results

        with patch.dict("sys.modules", {"mediapipe": mock_mp, "cv2": mock_cv2}):
            import importlib
            import motion_gaming.hand_detector

            importlib.reload(motion_gaming.hand_detector)
            from motion_gaming.hand_detector import HandDetector

            detector = HandDetector()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detector.detect(frame)

            mock_cv2.cvtColor.assert_called_once()
            call_args = mock_cv2.cvtColor.call_args
            assert np.array_equal(call_args[0][0], frame)
            assert call_args[0][1] == mock_cv2.COLOR_BGR2RGB

    def test_close_releases_resources(self, mock_mediapipe: tuple) -> None:
        """close() should call close on the MediaPipe hands object."""
        mock_mp, mock_hands = mock_mediapipe

        with patch.dict("sys.modules", {"mediapipe": mock_mp}):
            import importlib
            import motion_gaming.hand_detector

            importlib.reload(motion_gaming.hand_detector)
            from motion_gaming.hand_detector import HandDetector

            detector = HandDetector()
            detector.close()

            mock_hands.close.assert_called_once()

    def test_init_configures_mediapipe_correctly(self, mock_mediapipe: tuple) -> None:
        """HandDetector should configure MediaPipe with correct parameters."""
        mock_mp, mock_hands = mock_mediapipe

        with patch.dict("sys.modules", {"mediapipe": mock_mp}):
            import importlib
            import motion_gaming.hand_detector

            importlib.reload(motion_gaming.hand_detector)
            from motion_gaming.hand_detector import HandDetector

            HandDetector(max_hands=1, min_detection_confidence=0.8)

            mock_mp.solutions.hands.Hands.assert_called_once_with(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.5,
                model_complexity=0,
            )
