"""Tests for gesture recognition."""

import numpy as np
import pytest

from motion_gaming.gesture_recognizer import (
    Direction,
    Landmark,
    count_fingers,
    recognize_pointing,
)


def make_landmarks() -> np.ndarray:
    """Create a base landmarks array with all points at center."""
    return np.full((21, 3), 0.5, dtype=np.float32)


class TestRecognizePointing:
    """Tests for pointing direction recognition."""

    def test_pointing_north(self) -> None:
        """Pointing up should return NORTH."""
        landmarks = make_landmarks()
        # Index finger base at center, tip above (lower y)
        landmarks[Landmark.INDEX_FINGER_MCP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.INDEX_FINGER_TIP] = [0.5, 0.3, 0.0]

        result = recognize_pointing(landmarks)
        assert result == Direction.NORTH

    def test_pointing_south(self) -> None:
        """Pointing down should return SOUTH."""
        landmarks = make_landmarks()
        landmarks[Landmark.INDEX_FINGER_MCP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.INDEX_FINGER_TIP] = [0.5, 0.7, 0.0]

        result = recognize_pointing(landmarks)
        assert result == Direction.SOUTH

    def test_pointing_east(self) -> None:
        """Pointing right should return EAST."""
        landmarks = make_landmarks()
        landmarks[Landmark.INDEX_FINGER_MCP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.INDEX_FINGER_TIP] = [0.7, 0.5, 0.0]

        result = recognize_pointing(landmarks)
        assert result == Direction.EAST

    def test_pointing_west(self) -> None:
        """Pointing left should return WEST."""
        landmarks = make_landmarks()
        landmarks[Landmark.INDEX_FINGER_MCP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.INDEX_FINGER_TIP] = [0.3, 0.5, 0.0]

        result = recognize_pointing(landmarks)
        assert result == Direction.WEST

    def test_pointing_northeast(self) -> None:
        """Pointing up-right should return NORTHEAST."""
        landmarks = make_landmarks()
        landmarks[Landmark.INDEX_FINGER_MCP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.INDEX_FINGER_TIP] = [0.65, 0.35, 0.0]

        result = recognize_pointing(landmarks)
        assert result == Direction.NORTHEAST

    def test_pointing_northwest(self) -> None:
        """Pointing up-left should return NORTHWEST."""
        landmarks = make_landmarks()
        landmarks[Landmark.INDEX_FINGER_MCP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.INDEX_FINGER_TIP] = [0.35, 0.35, 0.0]

        result = recognize_pointing(landmarks)
        assert result == Direction.NORTHWEST

    def test_pointing_southeast(self) -> None:
        """Pointing down-right should return SOUTHEAST."""
        landmarks = make_landmarks()
        landmarks[Landmark.INDEX_FINGER_MCP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.INDEX_FINGER_TIP] = [0.65, 0.65, 0.0]

        result = recognize_pointing(landmarks)
        assert result == Direction.SOUTHEAST

    def test_pointing_southwest(self) -> None:
        """Pointing down-left should return SOUTHWEST."""
        landmarks = make_landmarks()
        landmarks[Landmark.INDEX_FINGER_MCP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.INDEX_FINGER_TIP] = [0.35, 0.65, 0.0]

        result = recognize_pointing(landmarks)
        assert result == Direction.SOUTHWEST

    def test_dead_zone(self) -> None:
        """Finger not extended enough should return None."""
        landmarks = make_landmarks()
        landmarks[Landmark.INDEX_FINGER_MCP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.INDEX_FINGER_TIP] = [0.51, 0.51, 0.0]  # Barely moved

        result = recognize_pointing(landmarks)
        assert result is None


class TestCountFingers:
    """Tests for finger counting."""

    def test_zero_fingers_fist(self) -> None:
        """Closed fist should return 0."""
        landmarks = make_landmarks()
        # All fingertips below their PIP joints (curled)
        for tip, pip in [
            (Landmark.INDEX_FINGER_TIP, Landmark.INDEX_FINGER_PIP),
            (Landmark.MIDDLE_FINGER_TIP, Landmark.MIDDLE_FINGER_PIP),
            (Landmark.RING_FINGER_TIP, Landmark.RING_FINGER_PIP),
            (Landmark.PINKY_TIP, Landmark.PINKY_PIP),
        ]:
            landmarks[pip] = [0.5, 0.4, 0.0]
            landmarks[tip] = [0.5, 0.6, 0.0]  # Below PIP

        # Thumb curled (tip left of IP for left hand)
        landmarks[Landmark.THUMB_IP] = [0.6, 0.5, 0.0]
        landmarks[Landmark.THUMB_TIP] = [0.5, 0.5, 0.0]

        result = count_fingers(landmarks)
        assert result == 0

    def test_one_finger(self) -> None:
        """One extended finger should return 1."""
        landmarks = make_landmarks()
        # Index finger extended (tip above PIP)
        landmarks[Landmark.INDEX_FINGER_PIP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.INDEX_FINGER_TIP] = [0.5, 0.3, 0.0]

        # Other fingers curled
        for tip, pip in [
            (Landmark.MIDDLE_FINGER_TIP, Landmark.MIDDLE_FINGER_PIP),
            (Landmark.RING_FINGER_TIP, Landmark.RING_FINGER_PIP),
            (Landmark.PINKY_TIP, Landmark.PINKY_PIP),
        ]:
            landmarks[pip] = [0.5, 0.4, 0.0]
            landmarks[tip] = [0.5, 0.6, 0.0]

        # Thumb curled
        landmarks[Landmark.THUMB_IP] = [0.6, 0.5, 0.0]
        landmarks[Landmark.THUMB_TIP] = [0.5, 0.5, 0.0]

        result = count_fingers(landmarks)
        assert result == 1

    def test_two_fingers(self) -> None:
        """Two extended fingers should return 2."""
        landmarks = make_landmarks()
        # Index and middle extended
        landmarks[Landmark.INDEX_FINGER_PIP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.INDEX_FINGER_TIP] = [0.5, 0.3, 0.0]
        landmarks[Landmark.MIDDLE_FINGER_PIP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.MIDDLE_FINGER_TIP] = [0.5, 0.3, 0.0]

        # Ring and pinky curled
        for tip, pip in [
            (Landmark.RING_FINGER_TIP, Landmark.RING_FINGER_PIP),
            (Landmark.PINKY_TIP, Landmark.PINKY_PIP),
        ]:
            landmarks[pip] = [0.5, 0.4, 0.0]
            landmarks[tip] = [0.5, 0.6, 0.0]

        # Thumb curled
        landmarks[Landmark.THUMB_IP] = [0.6, 0.5, 0.0]
        landmarks[Landmark.THUMB_TIP] = [0.5, 0.5, 0.0]

        result = count_fingers(landmarks)
        assert result == 2

    def test_five_fingers(self) -> None:
        """All fingers extended should return 5."""
        landmarks = make_landmarks()
        # All fingers extended (tips above PIPs)
        for tip, pip in [
            (Landmark.INDEX_FINGER_TIP, Landmark.INDEX_FINGER_PIP),
            (Landmark.MIDDLE_FINGER_TIP, Landmark.MIDDLE_FINGER_PIP),
            (Landmark.RING_FINGER_TIP, Landmark.RING_FINGER_PIP),
            (Landmark.PINKY_TIP, Landmark.PINKY_PIP),
        ]:
            landmarks[pip] = [0.5, 0.5, 0.0]
            landmarks[tip] = [0.5, 0.3, 0.0]

        # Thumb extended (tip right of IP for left hand)
        landmarks[Landmark.THUMB_IP] = [0.5, 0.5, 0.0]
        landmarks[Landmark.THUMB_TIP] = [0.7, 0.5, 0.0]

        result = count_fingers(landmarks)
        assert result == 5
