"""Gesture recognition from hand landmarks."""

from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt


class Direction(Enum):
    """Pointing directions."""

    NORTH = "N"
    NORTHEAST = "NE"
    EAST = "E"
    SOUTHEAST = "SE"
    SOUTH = "S"
    SOUTHWEST = "SW"
    WEST = "W"
    NORTHWEST = "NW"


# MediaPipe hand landmark indices
class Landmark:
    """MediaPipe hand landmark indices."""

    WRIST = 0
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_TIP = 16
    PINKY_TIP = 20
    THUMB_TIP = 4
    # Knuckle (PIP) indices for finger counting
    INDEX_FINGER_PIP = 6
    MIDDLE_FINGER_PIP = 10
    RING_FINGER_PIP = 14
    PINKY_PIP = 18
    THUMB_IP = 3


def recognize_pointing(landmarks: npt.NDArray[np.float32]) -> Optional[Direction]:
    """Recognize pointing direction from right hand landmarks.

    Uses the vector from index finger base (MCP) to tip to determine direction.

    Args:
        landmarks: Hand landmarks array of shape (21, 3).

    Returns:
        Direction if pointing gesture detected, None otherwise.
    """
    # Get index finger base and tip
    base = landmarks[Landmark.INDEX_FINGER_MCP]
    tip = landmarks[Landmark.INDEX_FINGER_TIP]

    # Calculate direction vector (in image coordinates, y increases downward)
    dx = tip[0] - base[0]
    dy = tip[1] - base[1]

    # Calculate magnitude
    magnitude = np.sqrt(dx * dx + dy * dy)

    # Dead zone threshold - finger must be extended enough
    if magnitude < 0.05:
        return None

    # Calculate angle (note: y is inverted in image coordinates)
    angle = np.arctan2(-dy, dx)  # Negate dy so up is positive
    angle_deg = np.degrees(angle)

    # Normalize to 0-360
    if angle_deg < 0:
        angle_deg += 360

    # Map angle to 8 directions (each sector is 45 degrees)
    # East = 0, North = 90, West = 180, South = 270
    if 337.5 <= angle_deg or angle_deg < 22.5:
        return Direction.EAST
    elif 22.5 <= angle_deg < 67.5:
        return Direction.NORTHEAST
    elif 67.5 <= angle_deg < 112.5:
        return Direction.NORTH
    elif 112.5 <= angle_deg < 157.5:
        return Direction.NORTHWEST
    elif 157.5 <= angle_deg < 202.5:
        return Direction.WEST
    elif 202.5 <= angle_deg < 247.5:
        return Direction.SOUTHWEST
    elif 247.5 <= angle_deg < 292.5:
        return Direction.SOUTH
    elif 292.5 <= angle_deg < 337.5:
        return Direction.SOUTHEAST

    return None


def count_fingers(landmarks: npt.NDArray[np.float32]) -> int:
    """Count extended fingers on left hand.

    A finger is considered extended if its tip is above (lower y) its PIP joint.
    Thumb uses different logic based on x-coordinate spread.

    Args:
        landmarks: Hand landmarks array of shape (21, 3).

    Returns:
        Number of extended fingers (0-5).
    """
    count = 0

    # Check thumb - tip should be to the left of IP joint for left hand
    # (in image coordinates, left hand's thumb extends to the right side of image)
    thumb_tip_x = landmarks[Landmark.THUMB_TIP][0]
    thumb_ip_x = landmarks[Landmark.THUMB_IP][0]
    if thumb_tip_x > thumb_ip_x:  # Extended for left hand (mirrored view)
        count += 1

    # Check other fingers - tip should be above (lower y) PIP joint
    finger_tips = [
        Landmark.INDEX_FINGER_TIP,
        Landmark.MIDDLE_FINGER_TIP,
        Landmark.RING_FINGER_TIP,
        Landmark.PINKY_TIP,
    ]
    finger_pips = [
        Landmark.INDEX_FINGER_PIP,
        Landmark.MIDDLE_FINGER_PIP,
        Landmark.RING_FINGER_PIP,
        Landmark.PINKY_PIP,
    ]

    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip_y = landmarks[tip_idx][1]
        pip_y = landmarks[pip_idx][1]
        if tip_y < pip_y:  # Tip is above PIP (lower y = higher on screen)
            count += 1

    return count
