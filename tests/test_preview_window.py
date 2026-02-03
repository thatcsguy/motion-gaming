"""Tests for preview window."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPreviewWindow:
    """Smoke tests for PreviewWindow."""

    @pytest.fixture
    def mock_cv2(self):
        """Create mocked cv2 module."""
        mock = MagicMock()
        mock.WINDOW_NORMAL = 0
        mock.FONT_HERSHEY_SIMPLEX = 0
        mock.getWindowProperty.return_value = 1.0  # Window is open
        mock.WND_PROP_VISIBLE = 0
        return mock

    @pytest.fixture
    def preview_window(self, mock_cv2: MagicMock):
        """Create PreviewWindow with mocked cv2."""
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            import importlib
            import motion_gaming.preview_window

            importlib.reload(motion_gaming.preview_window)
            from motion_gaming.preview_window import PreviewWindow

            window = PreviewWindow()
            yield window, mock_cv2

    def make_hand_data(
        self, handedness: str = "Right"
    ) -> "MagicMock":
        """Create a mock HandData with valid landmarks."""
        from motion_gaming.hand_detector import HandData

        landmarks = np.array(
            [[i * 0.04, i * 0.04, 0.0] for i in range(21)],
            dtype=np.float32,
        )
        return HandData(
            landmarks=landmarks,
            handedness=handedness,
            confidence=0.9,
        )

    def test_init_creates_window(self, mock_cv2: MagicMock) -> None:
        """PreviewWindow should create and resize an OpenCV window."""
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            import importlib
            import motion_gaming.preview_window

            importlib.reload(motion_gaming.preview_window)
            from motion_gaming.preview_window import PreviewWindow

            PreviewWindow()

            mock_cv2.namedWindow.assert_called_once_with(
                "Motion Gaming", mock_cv2.WINDOW_NORMAL
            )
            mock_cv2.resizeWindow.assert_called_once_with("Motion Gaming", 640, 480)

    def test_update_with_no_hands(
        self, preview_window: tuple[MagicMock, MagicMock]
    ) -> None:
        """update() should not crash with empty hands list."""
        window, mock_cv2 = preview_window
        from motion_gaming.gesture_recognizer import Direction

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        window.update(frame, [], None, 0, True)

        mock_cv2.imshow.assert_called_once()

    def test_update_with_one_hand(
        self, preview_window: tuple[MagicMock, MagicMock]
    ) -> None:
        """update() should not crash with one hand."""
        window, mock_cv2 = preview_window
        from motion_gaming.gesture_recognizer import Direction

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = self.make_hand_data("Right")

        window.update(frame, [hand], Direction.NORTH, 0, True)

        mock_cv2.imshow.assert_called_once()
        # Should have drawn lines and circles for hand
        assert mock_cv2.line.call_count > 0
        assert mock_cv2.circle.call_count > 0

    def test_update_with_two_hands(
        self, preview_window: tuple[MagicMock, MagicMock]
    ) -> None:
        """update() should not crash with two hands."""
        window, mock_cv2 = preview_window

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        left_hand = self.make_hand_data("Left")
        right_hand = self.make_hand_data("Right")

        window.update(frame, [left_hand, right_hand], None, 3, True)

        mock_cv2.imshow.assert_called_once()
        # Should have drawn both hands
        # 21 connections * 2 hands = 42 lines
        assert mock_cv2.line.call_count >= 42

    def test_update_tracking_disabled(
        self, preview_window: tuple[MagicMock, MagicMock]
    ) -> None:
        """update() should show paused status when tracking disabled."""
        window, mock_cv2 = preview_window

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        window.update(frame, [], None, 0, False)

        # Should still call imshow
        mock_cv2.imshow.assert_called_once()

        # Check that putText was called with PAUSED
        put_text_calls = mock_cv2.putText.call_args_list
        assert any("PAUSED" in str(call) for call in put_text_calls)

    def test_update_shows_direction(
        self, preview_window: tuple[MagicMock, MagicMock]
    ) -> None:
        """update() should display direction when tracking."""
        window, mock_cv2 = preview_window
        from motion_gaming.gesture_recognizer import Direction

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        window.update(frame, [], Direction.NORTH, 0, True)

        put_text_calls = mock_cv2.putText.call_args_list
        assert any("N" in str(call) for call in put_text_calls)

    def test_update_shows_finger_count(
        self, preview_window: tuple[MagicMock, MagicMock]
    ) -> None:
        """update() should display finger count when tracking."""
        window, mock_cv2 = preview_window

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        window.update(frame, [], None, 3, True)

        put_text_calls = mock_cv2.putText.call_args_list
        assert any("Fingers: 3" in str(call) for call in put_text_calls)

    def test_update_shows_keys_pressed(
        self, preview_window: tuple[MagicMock, MagicMock]
    ) -> None:
        """update() should display active keys when tracking."""
        window, mock_cv2 = preview_window
        from motion_gaming.gesture_recognizer import Direction

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        window.update(frame, [], Direction.NORTH, 2, True)

        put_text_calls = mock_cv2.putText.call_args_list
        # Should show W for north and 2 for finger count
        assert any("Keys:" in str(call) for call in put_text_calls)

    def test_is_open_returns_true_when_visible(
        self, preview_window: tuple[MagicMock, MagicMock]
    ) -> None:
        """is_open() should return True when window is visible."""
        window, mock_cv2 = preview_window
        mock_cv2.getWindowProperty.return_value = 1.0

        assert window.is_open() is True
        mock_cv2.getWindowProperty.assert_called_with(
            "Motion Gaming", mock_cv2.WND_PROP_VISIBLE
        )

    def test_is_open_returns_false_when_closed(
        self, preview_window: tuple[MagicMock, MagicMock]
    ) -> None:
        """is_open() should return False when window is closed."""
        window, mock_cv2 = preview_window
        mock_cv2.getWindowProperty.return_value = 0.0

        assert window.is_open() is False

    def test_close_destroys_window(
        self, preview_window: tuple[MagicMock, MagicMock]
    ) -> None:
        """close() should destroy the OpenCV window."""
        window, mock_cv2 = preview_window

        window.close()

        mock_cv2.destroyWindow.assert_called_once_with("Motion Gaming")

    def test_hand_colors_by_handedness(
        self, preview_window: tuple[MagicMock, MagicMock]
    ) -> None:
        """Right hand should be green, left hand should be blue."""
        window, mock_cv2 = preview_window

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        right_hand = self.make_hand_data("Right")
        window.update(frame, [right_hand], None, 0, True)

        # Check that green (0, 255, 0) was used for right hand
        line_calls = mock_cv2.line.call_args_list
        assert any(call[0][3] == (0, 255, 0) for call in line_calls)

        mock_cv2.reset_mock()

        left_hand = self.make_hand_data("Left")
        window.update(frame, [left_hand], None, 0, True)

        # Check that blue (255, 0, 0) was used for left hand
        line_calls = mock_cv2.line.call_args_list
        assert any(call[0][3] == (255, 0, 0) for call in line_calls)

    def test_does_not_modify_original_frame(
        self, preview_window: tuple[MagicMock, MagicMock]
    ) -> None:
        """update() should not modify the original frame."""
        window, mock_cv2 = preview_window

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_frame = frame.copy()
        hand = self.make_hand_data("Right")

        window.update(frame, [hand], None, 0, True)

        # Original frame should be unchanged
        assert np.array_equal(frame, original_frame)
